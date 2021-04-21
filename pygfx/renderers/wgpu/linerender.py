import wgpu  # only for flags/enums
import pyshader
from pyshader import Struct, i32, f32, vec2, vec3, vec4, ivec4, Array

from . import register_wgpu_render_function, stdinfo_uniform_type
from ...utils import array_from_shadertype
from ...resources import Buffer
from ...objects import Line
from ...materials import (
    LineMaterial,
    LineThinMaterial,
    LineThinSegmentMaterial,
    LineSegmentMaterial,
    LineArrowMaterial,
)


# ## Notes
#
# Rendering lines on the GPU is hard! The approach used here uses
# VertexId and storage buffers instead of vertex buffers. That way we
# can create multiple vertices for each point on the line. These
# vertices are positioned such that rendering with triangle_strip
# results in a tick line. More info below.
#
# An alternative (used by e.g. ThreeJS) is to draw the geometry of a
# line segment many times using instancing. Note that some Intel drivers
# limit the number of instances to about 5-10k, which is not much considering
# that lines consisting of millions of points can normally run realtime.
# https://wwwtyro.net/2019/11/18/instanced-lines.html
#
# Another alternative is a geometry shader, but wgpu does not have that.
# But we have compute shaders, which can be used to prepare the triangle-based
# geometry and store it in a buffer, which can be used as a vertex buffer in
# the render pass. The downside of this approach is the memory consumption.
# But a hybrid approach may be viable: the current approach using VertexId,
# but preparing some data in a buffer.

# todo: we can learn about dashing, unfolding and more at http://jcgt.org/published/0002/02/08/paper.pdf


renderer_uniform_type = Struct(last_i=i32)

# %% Shaders


# This compute shader is wrong and unused, but left as an example for what
# such a "geometry shader pass" could look like.
@pyshader.python2shader
def compute_shader(
    index_xyz: (pyshader.RES_INPUT, "GlobalInvocationId", "ivec3"),
    pos1: (pyshader.RES_BUFFER, (0, 0), Array(f32)),
    pos2: (pyshader.RES_BUFFER, (0, 1), Array(f32)),
    u_material: (pyshader.RES_UNIFORM, (0, 2), LineMaterial.uniform_type),
):
    index = index_xyz.x
    p = vec3(pos1[index * 3 + 0], pos1[index * 3 + 1], pos1[index * 3 + 2])
    dz = u_material.thickness
    pos2[index * 6 + 0] = p.x
    pos2[index * 6 + 1] = p.y + dz
    pos2[index * 6 + 2] = p.z
    pos2[index * 6 + 3] = p.x
    pos2[index * 6 + 4] = p.y - dz
    pos2[index * 6 + 5] = p.x


@pyshader.python2shader
def vertex_shader_thin(
    in_pos: (pyshader.RES_INPUT, 0, vec3),
    u_stdinfo: (pyshader.RES_UNIFORM, (0, 0), stdinfo_uniform_type),
    u_wobject: (pyshader.RES_UNIFORM, (0, 1), Line.uniform_type),
    out_pos: (pyshader.RES_OUTPUT, "Position", vec4),
):
    wpos = u_wobject.world_transform * vec4(in_pos.xyz, 1.0)
    npos = u_stdinfo.projection_transform * u_stdinfo.cam_transform * wpos
    out_pos = npos  # noqa


@pyshader.python2shader
def vertex_shader_thin_vtxclr(
    in_pos: (pyshader.RES_INPUT, 0, vec3),
    in_clr: (pyshader.RES_INPUT, 1, vec4),
    u_stdinfo: (pyshader.RES_UNIFORM, (0, 0), stdinfo_uniform_type),
    u_wobject: (pyshader.RES_UNIFORM, (0, 1), Line.uniform_type),
    out_pos: (pyshader.RES_OUTPUT, "Position", vec4),
    v_clr: (pyshader.RES_OUTPUT, 0, vec4),
):
    wpos = u_wobject.world_transform * vec4(in_pos.xyz, 1.0)
    npos = u_stdinfo.projection_transform * u_stdinfo.cam_transform * wpos
    out_pos = npos  # noqa
    v_clr = in_clr  # noqa


@pyshader.python2shader
def fragment_shader_thin_vtxclr(
    v_clr: (pyshader.RES_INPUT, 0, vec4),
    out_color: (pyshader.RES_OUTPUT, 0, vec4),
):
    out_color = v_clr  # noqa - shader assign


@pyshader.python2shader
def fragment_shader_thin(
    u_material: (pyshader.RES_UNIFORM, (0, 2), LineMaterial.uniform_type),
    out_color: (pyshader.RES_OUTPUT, 0, vec4),
):
    out_color = u_material.color  # noqa - shader assign


@pyshader.python2shader
def vertex_shader(
    index: (pyshader.RES_INPUT, "VertexId", "i32"),
    u_stdinfo: (pyshader.RES_UNIFORM, (0, 0), stdinfo_uniform_type),
    u_wobject: (pyshader.RES_UNIFORM, (0, 1), Line.uniform_type),
    u_material: (pyshader.RES_UNIFORM, (0, 2), LineMaterial.uniform_type),
    u_renderer: (pyshader.RES_UNIFORM, (0, 3), renderer_uniform_type),
    buf_pos: (pyshader.RES_BUFFER, (1, 0), Array(f32)),
    out_pos: (pyshader.RES_OUTPUT, "Position", vec4),
    v_line_width_p: (pyshader.RES_OUTPUT, 0, f32),
    v_vec_from_node_p: (pyshader.RES_OUTPUT, 1, vec2),
    v_vertex_idx: (pyshader.RES_OUTPUT, 2, vec2),
):
    # This vertex shader uses VertexId and storage buffers instead of
    # vertex buffers. It creates 5 vertices for each point on the line.
    # You could do with 4 to get bevel joins, but round joins are so
    # much nicer. The extra vertex is used to cover more fragments at
    # the joins (and caps). In the fragment shader we discard fragments
    # that are "out of range", based on a varying that represents the
    # vector from the node to the vertex.
    #
    # Basic algorithm and definitions:
    #
    # - We read the positions of three nodes, the current, previous, and next.
    # - These are converted to logical pixel screen space.
    # - We define four normal vectors (na, nb, nc, nd) which represent the
    #   vertices. Two for the previous segment, two for the next. One extra
    #   normal/vertex is defined at the join.
    # - These calculations are done for each vertex (yeah, bit of a waste),
    #   we select just one as output.
    #
    #            /  o     node 3
    #           /  /  /
    #          d  /  /
    #   - - - b  /  /     corners rectangles a, b, c, d
    #   o-------o  /      the vertex e is extra
    #   - - - - a c
    #                node 2
    #  node 1
    #
    # Note that at the inside of a join, the normals (b and d above)
    # move past each-other, causing the rectangles of both segments to
    # overlap. We could prevent this, but that would screw up
    # v_vec_from_node_p. Furthermore, consider a thick line with dense
    # nodes jumping all over the place, causing a lot of overlap anyway.
    # In summary, this viz relies on depth testing with "less" (or
    # semi-tranparent lines would break).
    #
    # Possible improvents:
    #
    # - can we do dashes/stipling?
    # - also implement bevel and miter joins
    # - also implement different caps
    # - we can prepare the nodes' screen coordinates in a compute shader.

    # Prepare some numbers
    screen_factor = u_stdinfo.logical_size.xy / 2.0
    l2p = u_stdinfo.physical_size.x / u_stdinfo.logical_size.x
    half_line_width = u_material.thickness * 0.5  # in logical pixels

    # What i in the node list (point on the line) is this?
    i = index // 5

    # Sample the current node and it's two neighbours, and convert to NDC
    pos1 = vec3(buf_pos[i * 3 - 3], buf_pos[i * 3 - 2], buf_pos[i * 3 - 1])
    pos2 = vec3(buf_pos[i * 3 + 0], buf_pos[i * 3 + 1], buf_pos[i * 3 + 2])
    pos3 = vec3(buf_pos[i * 3 + 3], buf_pos[i * 3 + 4], buf_pos[i * 3 + 5])
    wpos1 = u_wobject.world_transform * vec4(pos1.xyz, 1.0)
    wpos2 = u_wobject.world_transform * vec4(pos2.xyz, 1.0)
    wpos3 = u_wobject.world_transform * vec4(pos3.xyz, 1.0)
    npos1 = u_stdinfo.projection_transform * u_stdinfo.cam_transform * wpos1
    npos2 = u_stdinfo.projection_transform * u_stdinfo.cam_transform * wpos2
    npos3 = u_stdinfo.projection_transform * u_stdinfo.cam_transform * wpos3
    npos2 = npos2 / npos2.w
    npos3 = npos3 / npos3.w

    # Convert to logical screen coordinates, because that's were the lines work
    ppos1 = (npos1.xy + 1.0) * screen_factor
    ppos2 = (npos2.xy + 1.0) * screen_factor
    ppos3 = (npos3.xy + 1.0) * screen_factor

    # Get vectors normal to the line segments
    v1, v2 = ppos2.xy - ppos1.xy, ppos3.xy - ppos2.xy

    if i == 0:
        # This is the first point on the line: create a cap.
        v1 = v2
        nc = normalize(vec2(+v2.y, -v2.x))
        nd = -nc
        na = nd
        nb = nd - normalize(v2)
        ne = nc - normalize(v2)
    elif i == u_renderer.last_i:
        # This is the last point on the line: create a cap.
        v2 = v1
        na = normalize(vec2(+v1.y, -v1.x))
        nb = -na
        ne = nb + normalize(v1)
        nc = na + normalize(v1)
        nd = na
    else:
        # Create a join
        na = normalize(vec2(+v1.y, -v1.x))
        nb = -na
        nc = normalize(vec2(+v2.y, -v2.x))
        nd = -nc

        # Determine the angle between two of the normals. If this angle is smaller
        # than zero, the inside of the join is at nb/nd, otherwise it is at na/nc.
        angle = atan2(na.y, na.x) - atan2(nc.y, nc.x)
        angle = (angle + math.pi) % (2.0 * math.pi) - math.pi

        # From the angle we can also calculate the intersection of the lines.
        # We express it in a vector magnifier, and limit it to a factor 2,
        # since when the angle is ~pi, the intersection is near infinity.
        # For a bevel join we can omit ne (or set vec_mag to 1.0).
        # For a miter join we'd need an extra vertex to smoothly transition
        # from a miter to a bevel when the angle is too small.
        # Note that ne becomes inf if v1 == v2, but that's ok, because the
        # triangles in which ne takes part are degenerate for this use-case.
        vec_mag = 1.0 / max(0.25, cos(0.5 * angle))
        ne = normalize(normalize(v1) - normalize(v2)) * vec_mag

    # Select the correct vector, note that all except ne are unit.
    vectors = [na, nb, ne, nc, nd]
    the_vec = vectors[index % 5] * half_line_width

    # Outputs
    out_pos = vec4((ppos2 + the_vec) / screen_factor - 1.0, npos2.zw)  # noqa
    v_line_width_p = half_line_width * 2.0 * l2p  # noqa
    v_vec_from_node_p = the_vec * l2p  # noqa
    v_vertex_idx = vec2(i // 10000, i % 10000)  # noqa


@pyshader.python2shader
def fragment_shader(
    in_coord: (pyshader.RES_INPUT, "FragCoord", vec4),
    v_line_width_p: (pyshader.RES_INPUT, 0, f32),
    v_vec_from_node_p: (pyshader.RES_INPUT, 1, vec2),
    v_vertex_idx: (pyshader.RES_INPUT, 2, vec2),
    u_wobject: (pyshader.RES_UNIFORM, (0, 1), Line.uniform_type),
    u_material: (pyshader.RES_UNIFORM, (0, 2), LineMaterial.uniform_type),
    out_color: (pyshader.RES_OUTPUT, 0, vec4),
    out_pick: (pyshader.RES_OUTPUT, 1, ivec4),
    out_depth: (pyshader.RES_OUTPUT, "FragDepth", f32),
):
    # Discard fragments outside of the radius. This is what makes round
    # joins and caps. If we ever want bevel or miter joins, we should
    # change the vertex positions a bit, and drop these lines below.
    dist_to_node_p = length(v_vec_from_node_p)
    if dist_to_node_p > v_line_width_p * 0.5:
        return  # discard

    alpha = 1.0

    # Anti-aliasing. Note that because of the discarding above, we cannot
    # use MSAA for aa. But maybe we use another generic approach to aa. We'll see.
    # todo: because of this, our line gets a wee bit thinner, so we have to
    # output ticker lines in the vertex shader!
    aa_width = 1.2
    alpha = ((0.5 * v_line_width_p) - abs(dist_to_node_p)) / aa_width
    alpha = min(1.0, alpha) ** 2

    # The outer edges with lower alpha for aa are pushed a bit back to avoid artifacts.
    out_depth = in_coord.z + 0.0001 * (0.8 - min(0.8, alpha))  # noqa

    # Set color
    color = u_material.color
    out_color = vec4(color.rgb, min(1.0, color.a) * alpha)  # noqa - shader assign
    # Set picking info. Yes, the vertex_id interpolates correctly in encoded form.
    vf = f32(v_vertex_idx.x * 10000.0 + v_vertex_idx.y)
    vi = i32(vf + 0.5)
    out_pick = ivec4(u_wobject.id, 0, vi, (vf - f32(vi)) * 1048576.0)  # noqa


@pyshader.python2shader
def vertex_shader_segment(
    index: (pyshader.RES_INPUT, "VertexId", "i32"),
    u_stdinfo: (pyshader.RES_UNIFORM, (0, 0), stdinfo_uniform_type),
    u_wobject: (pyshader.RES_UNIFORM, (0, 1), Line.uniform_type),
    u_material: (pyshader.RES_UNIFORM, (0, 2), LineMaterial.uniform_type),
    buf_pos: (pyshader.RES_BUFFER, (1, 0), Array(f32)),
    out_pos: (pyshader.RES_OUTPUT, "Position", vec4),
    v_line_width_p: (pyshader.RES_OUTPUT, 0, f32),
    v_vec_from_node_p: (pyshader.RES_OUTPUT, 1, vec2),
    v_vertex_idx: (pyshader.RES_OUTPUT, 2, vec2),
):
    # Similar to the normal vertex shader, except we only draw segments,
    # using 5 vertices per node. Four for the segments, and 1 to create
    # a degenerate triangle for the space in between. So we only draw
    # caps, no joins.

    # Prepare some numbers
    screen_factor = u_stdinfo.logical_size.xy / 2.0
    l2p = u_stdinfo.physical_size.x / u_stdinfo.logical_size.x
    half_line_width = u_material.thickness * 0.5  # in logical pixels
    # What i in the node list (point on the line) is this?
    i = index // 5
    # Sample the current node and either of its neighbours
    i3 = i + 1 - (i % 2) * 2  # (i + 1) if i is even else (i - 1)
    pos2 = vec3(buf_pos[i * 3 + 0], buf_pos[i * 3 + 1], buf_pos[i * 3 + 2])
    pos3 = vec3(buf_pos[i3 * 3 + 0], buf_pos[i3 * 3 + 1], buf_pos[i3 * 3 + 2])
    # Convert to ndc
    wpos2 = u_wobject.world_transform * vec4(pos2.xyz, 1.0)
    wpos3 = u_wobject.world_transform * vec4(pos3.xyz, 1.0)
    npos2 = u_stdinfo.projection_transform * u_stdinfo.cam_transform * wpos2
    npos3 = u_stdinfo.projection_transform * u_stdinfo.cam_transform * wpos3
    npos2 = npos2 / npos2.w
    npos3 = npos3 / npos3.w
    # Convert to logical screen coordinates, because that's were the lines work
    ppos2 = (npos2.xy + 1.0) * screen_factor
    ppos3 = (npos3.xy + 1.0) * screen_factor

    # Get vectors normal to the line segments
    if (i % 2) == 0:
        # A left-cap
        v = normalize(ppos3.xy - ppos2.xy)
        nc = vec2(+v.y, -v.x)
        nd = -nc
        na = nc - v
        nb = nd - v
    else:
        # A right cap
        v = normalize(ppos2.xy - ppos3.xy)
        na = vec2(+v.y, -v.x)
        nb = -na
        nc = na + v
        nd = nb + v

    # Select the correct vector
    # Note the replicated vertices to create degenerate triangles
    vectors = [na, na, nb, nc, nd, na, nb, nc, nd, nd]
    the_vec = vectors[index % 10] * half_line_width

    # Outputs
    out_pos = vec4((ppos2 + the_vec) / screen_factor - 1.0, npos2.zw)  # noqa
    v_line_width_p = half_line_width * 2.0 * l2p  # noqa
    v_vec_from_node_p = the_vec * l2p  # noqa
    v_vertex_idx = vec2(i // 10000, i % 10000)  # noqa


@pyshader.python2shader
def vertex_shader_arrow(
    index: (pyshader.RES_INPUT, "VertexId", "i32"),
    u_stdinfo: (pyshader.RES_UNIFORM, (0, 0), stdinfo_uniform_type),
    u_wobject: (pyshader.RES_UNIFORM, (0, 1), Line.uniform_type),
    u_material: (pyshader.RES_UNIFORM, (0, 2), LineMaterial.uniform_type),
    buf_pos: (pyshader.RES_BUFFER, (1, 0), Array(f32)),
    out_pos: (pyshader.RES_OUTPUT, "Position", vec4),
    v_line_width_p: (pyshader.RES_OUTPUT, 0, f32),
    v_vec_from_node_p: (pyshader.RES_OUTPUT, 1, vec2),
    v_vertex_idx: (pyshader.RES_OUTPUT, 2, vec2),
):
    # Similar to the normal vertex shader, except we only draw segments,
    # using 3 vertices per node: 6 per segment. 4 for the arrow, and 2
    # to create a degenerate triangle for the space in between. So we
    # only draw caps, no joins.

    # Prepare some numbers
    screen_factor = u_stdinfo.logical_size.xy / 2.0
    l2p = u_stdinfo.physical_size.x / u_stdinfo.logical_size.x
    half_line_width = u_material.thickness * 0.5  # in logical pixels
    # What i in the node list (point on the line) is this?
    i = index // 3
    # Sample the current node and either of its neighbours
    i3 = i + 1 - (i % 2) * 2  # (i + 1) if i is even else (i - 1)
    pos2 = vec3(buf_pos[i * 3 + 0], buf_pos[i * 3 + 1], buf_pos[i * 3 + 2])
    pos3 = vec3(buf_pos[i3 * 3 + 0], buf_pos[i3 * 3 + 1], buf_pos[i3 * 3 + 2])
    # Convert to ndc
    wpos2 = u_wobject.world_transform * vec4(pos2.xyz, 1.0)
    wpos3 = u_wobject.world_transform * vec4(pos3.xyz, 1.0)
    npos2 = u_stdinfo.projection_transform * u_stdinfo.cam_transform * wpos2
    npos3 = u_stdinfo.projection_transform * u_stdinfo.cam_transform * wpos3
    npos2 = npos2 / npos2.w
    npos3 = npos3 / npos3.w
    # Convert to logical screen coordinates, because that's were the lines work
    ppos2 = (npos2.xy + 1.0) * screen_factor
    ppos3 = (npos3.xy + 1.0) * screen_factor

    # Get vectors normal to the line segments
    if (i % 2) == 0:
        # A left-cap
        v = ppos3.xy - ppos2.xy
        na = normalize(vec2(+v.y, -v.x)) * half_line_width
        nb = v
    else:
        # A right cap
        v = ppos2.xy - ppos3.xy
        na = -0.75 * v
        nb = normalize(vec2(-v.y, +v.x)) * half_line_width - v

    # Select the correct vector
    # Note the replicated vertices to create degenerate triangles
    vectors = [na, na, nb, na, nb, nb]
    the_vec = vectors[index % 6]

    # Outputs
    out_pos = vec4((ppos2 + the_vec) / screen_factor - 1.0, npos2.zw)  # noqa
    v_line_width_p = half_line_width * 2.0 * l2p  # noqa
    v_vec_from_node_p = vec2(0.0, 0.0)  # noqa
    v_vertex_idx = vec2(i // 10000, i % 10000)  # noqa


@pyshader.python2shader
def vertex_shader_arrow_vtxclr(
    index: (pyshader.RES_INPUT, "VertexId", "i32"),
    u_stdinfo: (pyshader.RES_UNIFORM, (0, 0), stdinfo_uniform_type),
    u_wobject: (pyshader.RES_UNIFORM, (0, 1), Line.uniform_type),
    u_material: (pyshader.RES_UNIFORM, (0, 2), LineMaterial.uniform_type),
    buf_pos: (pyshader.RES_BUFFER, (1, 0), Array(f32)),
    buf_clr: (pyshader.RES_BUFFER, (1, 1), Array(f32)),
    out_pos: (pyshader.RES_OUTPUT, "Position", vec4),
    v_line_width_p: (pyshader.RES_OUTPUT, 0, f32),
    v_vec_from_node_p: (pyshader.RES_OUTPUT, 1, vec2),
    v_vertex_idx: (pyshader.RES_OUTPUT, 2, vec2),
    v_clr: (pyshader.RES_OUTPUT, 3, vec4),
):
    # Similar to the normal vertex shader, except we only draw segments,
    # using 3 vertices per node: 6 per segment. 4 for the arrow, and 2
    # to create a degenerate triangle for the space in between. So we
    # only draw caps, no joins.

    # Prepare some numbers
    screen_factor = u_stdinfo.logical_size.xy / 2.0
    l2p = u_stdinfo.physical_size.x / u_stdinfo.logical_size.x
    half_line_width = u_material.thickness * 0.5  # in logical pixels
    # What i in the node list (point on the line) is this?
    i = index // 3
    # Sample the current node's color
    color = vec4(
        buf_clr[i * 4 + 0], buf_clr[i * 4 + 1], buf_clr[i * 4 + 2], buf_clr[i * 4 + 3]
    )
    # Sample the current node and either of its neighbours
    i3 = i + 1 - (i % 2) * 2  # (i + 1) if i is even else (i - 1)
    pos2 = vec3(buf_pos[i * 3 + 0], buf_pos[i * 3 + 1], buf_pos[i * 3 + 2])
    pos3 = vec3(buf_pos[i3 * 3 + 0], buf_pos[i3 * 3 + 1], buf_pos[i3 * 3 + 2])
    # Convert to ndc
    wpos2 = u_wobject.world_transform * vec4(pos2.xyz, 1.0)
    wpos3 = u_wobject.world_transform * vec4(pos3.xyz, 1.0)
    npos2 = u_stdinfo.projection_transform * u_stdinfo.cam_transform * wpos2
    npos3 = u_stdinfo.projection_transform * u_stdinfo.cam_transform * wpos3
    npos2 = npos2 / npos2.w
    npos3 = npos3 / npos3.w
    # Convert to logical screen coordinates, because that's were the lines work
    ppos2 = (npos2.xy + 1.0) * screen_factor
    ppos3 = (npos3.xy + 1.0) * screen_factor

    # Get vectors normal to the line segments
    if (i % 2) == 0:
        # A left-cap
        v = ppos3.xy - ppos2.xy
        na = normalize(vec2(+v.y, -v.x)) * half_line_width
        nb = v
    else:
        # A right cap
        v = ppos2.xy - ppos3.xy
        na = -0.75 * v
        nb = normalize(vec2(-v.y, +v.x)) * half_line_width - v

    # Select the correct vector
    # Note the replicated vertices to create degenerate triangles
    vectors = [na, na, nb, na, nb, nb]
    the_vec = vectors[index % 6]

    # Outputs
    out_pos = vec4((ppos2 + the_vec) / screen_factor - 1.0, npos2.zw)  # noqa
    v_line_width_p = half_line_width * 2.0 * l2p  # noqa
    v_vec_from_node_p = vec2(0.0, 0.0)  # noqa
    v_vertex_idx = vec2(i // 10000, i % 10000)  # noqa
    v_clr = color  # noqa


@pyshader.python2shader
def vertex_shader_vtxclr(
    index: (pyshader.RES_INPUT, "VertexId", "i32"),
    u_stdinfo: (pyshader.RES_UNIFORM, (0, 0), stdinfo_uniform_type),
    u_wobject: (pyshader.RES_UNIFORM, (0, 1), Line.uniform_type),
    u_material: (pyshader.RES_UNIFORM, (0, 2), LineMaterial.uniform_type),
    u_renderer: (pyshader.RES_UNIFORM, (0, 3), renderer_uniform_type),
    buf_pos: (pyshader.RES_BUFFER, (1, 0), Array(f32)),
    buf_clr: (pyshader.RES_BUFFER, (1, 1), Array(f32)),
    out_pos: (pyshader.RES_OUTPUT, "Position", vec4),
    v_line_width_p: (pyshader.RES_OUTPUT, 0, f32),
    v_vec_from_node_p: (pyshader.RES_OUTPUT, 1, vec2),
    v_vertex_idx: (pyshader.RES_OUTPUT, 2, vec2),
    v_clr: (pyshader.RES_OUTPUT, 3, vec4),
):
    # This vertex shader uses VertexId and storage buffers instead of
    # vertex buffers. It creates 5 vertices for each point on the line.
    # You could do with 4 to get bevel joins, but round joins are so
    # much nicer. The extra vertex is used to cover more fragments at
    # the joins (and caps). In the fragment shader we discard fragments
    # that are "out of range", based on a varying that represents the
    # vector from the node to the vertex.
    #
    # Basic algorithm and definitions:
    #
    # - We read the positions of three nodes, the current, previous, and next.
    # - These are converted to logical pixel screen space.
    # - We define four normal vectors (na, nb, nc, nd) which represent the
    #   vertices. Two for the previous segment, two for the next. One extra
    #   normal/vertex is defined at the join.
    # - These calculations are done for each vertex (yeah, bit of a waste),
    #   we select just one as output.
    #
    #            /  o     node 3
    #           /  /  /
    #          d  /  /
    #   - - - b  /  /     corners rectangles a, b, c, d
    #   o-------o  /      the vertex e is extra
    #   - - - - a c
    #                node 2
    #  node 1
    #
    # Note that at the inside of a join, the normals (b and d above)
    # move past each-other, causing the rectangles of both segments to
    # overlap. We could prevent this, but that would screw up
    # v_vec_from_node_p. Furthermore, consider a thick line with dense
    # nodes jumping all over the place, causing a lot of overlap anyway.
    # In summary, this viz relies on depth testing with "less" (or
    # semi-tranparent lines would break).
    #
    # Possible improvents:
    #
    # - can we do dashes/stipling?
    # - also implement bevel and miter joins
    # - also implement different caps
    # - we can prepare the nodes' screen coordinates in a compute shader.

    # Prepare some numbers
    screen_factor = u_stdinfo.logical_size.xy / 2.0
    l2p = u_stdinfo.physical_size.x / u_stdinfo.logical_size.x
    half_line_width = u_material.thickness * 0.5  # in logical pixels

    # What i in the node list (point on the line) is this?
    i = index // 5

    # Sample the current node's color
    color = vec4(
        buf_clr[i * 4 + 0], buf_clr[i * 4 + 1], buf_clr[i * 4 + 2], buf_clr[i * 4 + 3]
    )

    # Sample the current node and it's two neighbours, and convert to NDC
    pos1 = vec3(buf_pos[i * 3 - 3], buf_pos[i * 3 - 2], buf_pos[i * 3 - 1])
    pos2 = vec3(buf_pos[i * 3 + 0], buf_pos[i * 3 + 1], buf_pos[i * 3 + 2])
    pos3 = vec3(buf_pos[i * 3 + 3], buf_pos[i * 3 + 4], buf_pos[i * 3 + 5])
    wpos1 = u_wobject.world_transform * vec4(pos1.xyz, 1.0)
    wpos2 = u_wobject.world_transform * vec4(pos2.xyz, 1.0)
    wpos3 = u_wobject.world_transform * vec4(pos3.xyz, 1.0)
    npos1 = u_stdinfo.projection_transform * u_stdinfo.cam_transform * wpos1
    npos2 = u_stdinfo.projection_transform * u_stdinfo.cam_transform * wpos2
    npos3 = u_stdinfo.projection_transform * u_stdinfo.cam_transform * wpos3
    npos2 = npos2 / npos2.w
    npos3 = npos3 / npos3.w

    # Convert to logical screen coordinates, because that's were the lines work
    ppos1 = (npos1.xy + 1.0) * screen_factor
    ppos2 = (npos2.xy + 1.0) * screen_factor
    ppos3 = (npos3.xy + 1.0) * screen_factor

    # Get vectors normal to the line segments
    v1, v2 = ppos2.xy - ppos1.xy, ppos3.xy - ppos2.xy

    if i == 0:
        # This is the first point on the line: create a cap.
        v1 = v2
        nc = normalize(vec2(+v2.y, -v2.x))
        nd = -nc
        na = nd
        nb = nd - normalize(v2)
        ne = nc - normalize(v2)
    elif i == u_renderer.last_i:
        # This is the last point on the line: create a cap.
        v2 = v1
        na = normalize(vec2(+v1.y, -v1.x))
        nb = -na
        ne = nb + normalize(v1)
        nc = na + normalize(v1)
        nd = na
    else:
        # Create a join
        na = normalize(vec2(+v1.y, -v1.x))
        nb = -na
        nc = normalize(vec2(+v2.y, -v2.x))
        nd = -nc

        # Determine the angle between two of the normals. If this angle is smaller
        # than zero, the inside of the join is at nb/nd, otherwise it is at na/nc.
        angle = atan2(na.y, na.x) - atan2(nc.y, nc.x)
        angle = (angle + math.pi) % (2.0 * math.pi) - math.pi

        # From the angle we can also calculate the intersection of the lines.
        # We express it in a vector magnifier, and limit it to a factor 2,
        # since when the angle is ~pi, the intersection is near infinity.
        # For a bevel join we can omit ne (or set vec_mag to 1.0).
        # For a miter join we'd need an extra vertex to smoothly transition
        # from a miter to a bevel when the angle is too small.
        # Note that ne becomes inf if v1 == v2, but that's ok, because the
        # triangles in which ne takes part are degenerate for this use-case.
        vec_mag = 1.0 / max(0.25, cos(0.5 * angle))
        ne = normalize(normalize(v1) - normalize(v2)) * vec_mag

    # Select the correct vector, note that all except ne are unit.
    vectors = [na, nb, ne, nc, nd]
    the_vec = vectors[index % 5] * half_line_width

    # Outputs
    out_pos = vec4((ppos2 + the_vec) / screen_factor - 1.0, npos2.zw)  # noqa
    v_line_width_p = half_line_width * 2.0 * l2p  # noqa
    v_vec_from_node_p = the_vec * l2p  # noqa
    v_vertex_idx = vec2(i // 10000, i % 10000)  # noqa
    v_clr = color  # noqa


@pyshader.python2shader
def vertex_shader_vtxclr_segment(
    index: (pyshader.RES_INPUT, "VertexId", "i32"),
    u_stdinfo: (pyshader.RES_UNIFORM, (0, 0), stdinfo_uniform_type),
    u_wobject: (pyshader.RES_UNIFORM, (0, 1), Line.uniform_type),
    u_material: (pyshader.RES_UNIFORM, (0, 2), LineMaterial.uniform_type),
    buf_pos: (pyshader.RES_BUFFER, (1, 0), Array(f32)),
    buf_clr: (pyshader.RES_BUFFER, (1, 1), Array(f32)),
    out_pos: (pyshader.RES_OUTPUT, "Position", vec4),
    v_line_width_p: (pyshader.RES_OUTPUT, 0, f32),
    v_vec_from_node_p: (pyshader.RES_OUTPUT, 1, vec2),
    v_vertex_idx: (pyshader.RES_OUTPUT, 2, vec2),
    v_clr: (pyshader.RES_OUTPUT, 3, vec4),
):
    # Similar to the normal vertex shader, except we only draw segments,
    # using 5 vertices per node. Four for the segments, and 1 to create
    # a degenerate triangle for the space in between. So we only draw
    # caps, no joins.

    # Prepare some numbers
    screen_factor = u_stdinfo.logical_size.xy / 2.0
    l2p = u_stdinfo.physical_size.x / u_stdinfo.logical_size.x
    half_line_width = u_material.thickness * 0.5  # in logical pixels
    # What i in the node list (point on the line) is this?
    i = index // 5
    # Sample the current node's color
    color = vec4(
        buf_clr[i * 4 + 0], buf_clr[i * 4 + 1], buf_clr[i * 4 + 2], buf_clr[i * 4 + 3]
    )
    # Sample the current node and either of its neighbours
    i3 = i + 1 - (i % 2) * 2  # (i + 1) if i is even else (i - 1)
    pos2 = vec3(buf_pos[i * 3 + 0], buf_pos[i * 3 + 1], buf_pos[i * 3 + 2])
    pos3 = vec3(buf_pos[i3 * 3 + 0], buf_pos[i3 * 3 + 1], buf_pos[i3 * 3 + 2])
    # Convert to ndc
    wpos2 = u_wobject.world_transform * vec4(pos2.xyz, 1.0)
    wpos3 = u_wobject.world_transform * vec4(pos3.xyz, 1.0)
    npos2 = u_stdinfo.projection_transform * u_stdinfo.cam_transform * wpos2
    npos3 = u_stdinfo.projection_transform * u_stdinfo.cam_transform * wpos3
    npos2 = npos2 / npos2.w
    npos3 = npos3 / npos3.w
    # Convert to logical screen coordinates, because that's were the lines work
    ppos2 = (npos2.xy + 1.0) * screen_factor
    ppos3 = (npos3.xy + 1.0) * screen_factor

    # Get vectors normal to the line segments
    if (i % 2) == 0:
        # A left-cap
        v = normalize(ppos3.xy - ppos2.xy)
        nc = vec2(+v.y, -v.x)
        nd = -nc
        na = nc - v
        nb = nd - v
    else:
        # A right cap
        v = normalize(ppos2.xy - ppos3.xy)
        na = vec2(+v.y, -v.x)
        nb = -na
        nc = na + v
        nd = nb + v

    # Select the correct vector
    # Note the replicated vertices to create degenerate triangles
    vectors = [na, na, nb, nc, nd, na, nb, nc, nd, nd]
    the_vec = vectors[index % 10] * half_line_width

    # Outputs
    out_pos = vec4((ppos2 + the_vec) / screen_factor - 1.0, npos2.zw)  # noqa
    v_line_width_p = half_line_width * 2.0 * l2p  # noqa
    v_vec_from_node_p = the_vec * l2p  # noqa
    v_vertex_idx = vec2(i // 10000, i % 10000)  # noqa
    v_clr = color  # noqa


@pyshader.python2shader
def fragment_shader_vtxclr(
    in_coord: (pyshader.RES_INPUT, "FragCoord", vec4),
    v_line_width_p: (pyshader.RES_INPUT, 0, f32),
    v_vec_from_node_p: (pyshader.RES_INPUT, 1, vec2),
    v_vertex_idx: (pyshader.RES_INPUT, 2, vec2),
    v_clr: (pyshader.RES_INPUT, 3, vec4),
    u_wobject: (pyshader.RES_UNIFORM, (0, 1), Line.uniform_type),
    out_color: (pyshader.RES_OUTPUT, 0, vec4),
    out_pick: (pyshader.RES_OUTPUT, 1, ivec4),
    out_depth: (pyshader.RES_OUTPUT, "FragDepth", f32),
):
    # Discard fragments outside of the radius. This is what makes round
    # joins and caps. If we ever want bevel or miter joins, we should
    # change the vertex positions a bit, and drop these lines below.
    dist_to_node_p = length(v_vec_from_node_p)
    if dist_to_node_p > v_line_width_p * 0.5:
        return  # discard

    alpha = 1.0

    # Anti-aliasing. Note that because of the discarding above, we cannot
    # use MSAA for aa. But maybe we use another generic approach to aa. We'll see.
    # todo: because of this, our line gets a wee bit thinner, so we have to
    # output ticker lines in the vertex shader!
    aa_width = 1.2
    alpha = ((0.5 * v_line_width_p) - abs(dist_to_node_p)) / aa_width
    alpha = min(1.0, alpha) ** 2

    # The outer edges with lower alpha for aa are pushed a bit back to avoid artifacts.
    out_depth = in_coord.z + 0.0001 * (0.8 - min(0.8, alpha))  # noqa

    # Set color
    color = v_clr
    out_color = vec4(color.rgb, min(1.0, color.a) * alpha)  # noqa - shader assign

    # Set picking info.
    vf = f32(v_vertex_idx.x * 10000.0 + v_vertex_idx.y)
    vi = i32(vf + 0.5)
    out_pick = ivec4(u_wobject.id, 0, vi, (vf - f32(vi)) * 1048576.0)  # noqa


# %% Render functions


@register_wgpu_render_function(Line, LineThinSegmentMaterial)
@register_wgpu_render_function(Line, LineThinMaterial)
def thin_line_renderer(wobject, render_info):
    """Render function capable of rendering lines."""

    material = wobject.material
    geometry = wobject.geometry

    positions1 = geometry.positions

    primitive = wgpu.PrimitiveTopology.line_strip
    if isinstance(material, LineThinSegmentMaterial):
        primitive = wgpu.PrimitiveTopology.line_list

    vertex_buffers = {0: positions1}
    vert_shader = vertex_shader_thin
    frag_shader = fragment_shader_thin

    if material.vertex_colors:
        colors1 = geometry.colors
        if colors1.data.shape[1] != 4:
            raise ValueError(
                "For rendering (thick) lines, the geometry.colors must be Nx4."
            )
        vertex_buffers[1] = colors1
        vert_shader = vertex_shader_thin_vtxclr
        frag_shader = fragment_shader_thin_vtxclr

    return [
        {
            "vertex_shader": vert_shader,
            "fragment_shader": frag_shader,
            "primitive_topology": primitive,
            "indices": (positions1.nitems, 1),
            "vertex_buffers": vertex_buffers,
            "bindings0": {
                0: ("buffer/uniform", render_info.stdinfo_uniform),
                1: ("buffer/uniform", wobject.uniform_buffer),
                2: ("buffer/uniform", material.uniform_buffer),
            },
            "target": None,  # default
        },
    ]


@register_wgpu_render_function(Line, LineMaterial)
def line_renderer(wobject, render_info):
    """Render function capable of rendering lines."""

    material = wobject.material
    geometry = wobject.geometry

    assert isinstance(material, LineMaterial)

    positions1 = geometry.positions

    # With vertex buffers, if a shader input is vec4, and the vbo has
    # Nx2, the z and w element will be zero. This works, because for
    # vertex buffers we provide additional information about the
    # striding of the data.
    # With storage buffers (aka SSBO) we just have some bytes that we
    # read from/write to in the shader. This is more free, but it means
    # that the data in the buffer must match with what the shader
    # expects. In addition to that, there's this thing with vec3's which
    # are padded to 16 bytes. So we either have to require our users
    # to provide Nx4 data, or read them as an array of f32.
    # Anyway, extra check here to make sure the data matches!
    # todo: data.something in here, which means we assume numpy-ish arrays
    if positions1.data.shape[1] != 3:
        raise ValueError(
            "For rendering (thick) lines, the geometry.positions must be Nx3."
        )

    bindings0 = {
        0: ("buffer/uniform", render_info.stdinfo_uniform),
        1: ("buffer/uniform", wobject.uniform_buffer),
        2: ("buffer/uniform", material.uniform_buffer),
    }

    bindings1 = {
        0: ("buffer/read_only_storage", positions1),
    }

    if material.vertex_colors:
        colors1 = geometry.colors
        if colors1.data.shape[1] != 4:
            raise ValueError(
                "For rendering (thick) lines, the geometry.colors must be Nx4."
            )
        bindings1[1] = ("buffer/read_only_storage", colors1)

    if isinstance(material, LineArrowMaterial):
        vert_shader = vertex_shader_arrow
        if material.vertex_colors:
            vert_shader = vertex_shader_arrow_vtxclr
        n = (positions1.nitems // 2) * 2 * 4
    elif isinstance(material, LineSegmentMaterial):
        vert_shader = vertex_shader_segment
        if material.vertex_colors:
            vert_shader = vertex_shader_vtxclr_segment
        n = (positions1.nitems // 2) * 2 * 5
    else:
        vert_shader = vertex_shader
        if material.vertex_colors:
            vert_shader = vertex_shader_vtxclr
        n = positions1.nitems * 5
        uniform_buffer = Buffer(
            array_from_shadertype(renderer_uniform_type), usage="UNIFORM"
        )
        uniform_buffer.data["last_i"] = positions1.nitems - 1
        bindings0[3] = "buffer/uniform", uniform_buffer

    frag_shader = fragment_shader
    if material.vertex_colors:
        frag_shader = fragment_shader_vtxclr

    return [
        {
            "vertex_shader": vert_shader,
            "fragment_shader": frag_shader,
            "primitive_topology": wgpu.PrimitiveTopology.triangle_strip,
            "indices": (n, 1),
            "bindings0": bindings0,
            "bindings1": bindings1,
            "target": None,  # default
        },
    ]
