import wgpu  # only for flags/enums
import pyshader
from pyshader import f32, vec2, vec4, Array

from . import register_wgpu_render_function, stdinfo_uniform_type
from ...objects import Line
from ...materials import (
    LineMaterial,
    LineThinMaterial,
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


# %% Shaders


# This compute shader is wrong and unused, but left as an example for what
# such a "geometry shader pass" could look like.
@pyshader.python2shader
def compute_shader(
    index: (pyshader.RES_INPUT, "GlobalInvocationId", "i32"),
    pos1: (pyshader.RES_BUFFER, (0, 0), Array(vec4)),
    pos2: (pyshader.RES_BUFFER, (0, 1), Array(vec4)),
    material: (pyshader.RES_UNIFORM, (0, 2), LineMaterial.uniform_type),
):
    p = pos1[index] * 1.0
    dz = material.thickness
    pos2[index * 2 + 0] = vec4(p.x, p.y + dz, p.z, 1.0)
    pos2[index * 2 + 1] = vec4(p.x, p.y - dz, p.z, 1.0)


@pyshader.python2shader
def vertex_shader_thin(
    in_pos: (pyshader.RES_INPUT, 0, vec4),
    stdinfo: (pyshader.RES_UNIFORM, (0, 0), stdinfo_uniform_type),
    out_pos: (pyshader.RES_OUTPUT, "Position", vec4),
):
    wpos = stdinfo.world_transform * in_pos
    npos = stdinfo.projection_transform * stdinfo.cam_transform * wpos
    out_pos = npos  # noqa


@pyshader.python2shader
def fragment_shader_thin(
    material: (pyshader.RES_UNIFORM, (0, 1), LineMaterial.uniform_type),
    out_color: (pyshader.RES_OUTPUT, 0, vec4),
):
    out_color = material.color  # noqa - shader assign


@pyshader.python2shader
def vertex_shader(
    index: (pyshader.RES_INPUT, "VertexId", "i32"),
    buf_pos: (pyshader.RES_BUFFER, (0, 2), Array(vec4)),
    stdinfo: (pyshader.RES_UNIFORM, (0, 0), stdinfo_uniform_type),
    material: (pyshader.RES_UNIFORM, (0, 1), LineMaterial.uniform_type),
    out_pos: (pyshader.RES_OUTPUT, "Position", vec4),
    v_line_width_p: (pyshader.RES_OUTPUT, 0, f32),
    v_vec_from_node_p: (pyshader.RES_OUTPUT, 1, vec2),
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
    screen_factor = stdinfo.logical_size.xy / 2.0
    l2p = stdinfo.physical_size.x / stdinfo.logical_size.x
    half_line_width = material.thickness * 0.5  # in logical pixels
    half_line_width_p = half_line_width * l2p  # in physical pixels

    # What i in the node list (point on the line) is this?
    i = index // 5

    # Sample the current node and it's two neighbours, and convert to NDC
    pos1, pos2, pos3 = buf_pos[i - 1], buf_pos[i], buf_pos[i + 1]
    wpos1 = stdinfo.world_transform * vec4(pos1.xyz, 1.0)
    wpos2 = stdinfo.world_transform * vec4(pos2.xyz, 1.0)
    wpos3 = stdinfo.world_transform * vec4(pos3.xyz, 1.0)
    npos1 = stdinfo.projection_transform * stdinfo.cam_transform * wpos1
    npos2 = stdinfo.projection_transform * stdinfo.cam_transform * wpos2
    npos3 = stdinfo.projection_transform * stdinfo.cam_transform * wpos3

    # Convert to logical screen coordinates, because that's were the lines work
    ppos1 = (npos1.xy + 1.0) * screen_factor
    ppos2 = (npos2.xy + 1.0) * screen_factor
    ppos3 = (npos3.xy + 1.0) * screen_factor

    # Get vectors normal to the line segments
    v1, v2 = ppos2.xy - ppos1.xy, ppos3.xy - ppos2.xy

    if pos1.w == 0.0:
        # This is the first point on the line: create a cap.
        v1 = v2
        nc = normalize(vec2(+v2.y, -v2.x))
        nd = -nc
        na = nd
        nb = nd - normalize(v2)
        ne = nc - normalize(v2)
    elif pos3.w == 0.0:
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
    v_line_width_p = half_line_width_p * 2.0  # noqa
    v_vec_from_node_p = the_vec * l2p  # noqa


@pyshader.python2shader
def fragment_shader(
    in_coord: (pyshader.RES_INPUT, "FragCoord", vec4),
    v_line_width_p: (pyshader.RES_INPUT, 0, f32),
    v_vec_from_node_p: (pyshader.RES_INPUT, 1, vec2),
    stdinfo: (pyshader.RES_UNIFORM, (0, 0), stdinfo_uniform_type),
    material: (pyshader.RES_UNIFORM, (0, 1), LineMaterial.uniform_type),
    out_color: (pyshader.RES_OUTPUT, 0, vec4),
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
    color = material.color
    out_color = vec4(color.rgb, min(1.0, color.a) * alpha)  # noqa - shader assign


@pyshader.python2shader
def vertex_shader_segment(
    index: (pyshader.RES_INPUT, "VertexId", "i32"),
    buf_pos: (pyshader.RES_BUFFER, (0, 2), Array(vec4)),
    stdinfo: (pyshader.RES_UNIFORM, (0, 0), stdinfo_uniform_type),
    material: (pyshader.RES_UNIFORM, (0, 1), LineMaterial.uniform_type),
    out_pos: (pyshader.RES_OUTPUT, "Position", vec4),
    v_line_width_p: (pyshader.RES_OUTPUT, 0, f32),
    v_vec_from_node_p: (pyshader.RES_OUTPUT, 1, vec2),
):
    # Similar to the normal vertex shader, except we only draw segments,
    # using 5 vertices per node. Four for the segments, and 1 to create
    # a degenerate triangle for the space in between. So we only draw
    # caps, no joins.

    # Prepare some numbers
    screen_factor = stdinfo.logical_size.xy / 2.0
    l2p = stdinfo.physical_size.x / stdinfo.logical_size.x
    half_line_width = material.thickness * 0.5  # in logical pixels
    half_line_width_p = half_line_width * l2p  # in physical pixels
    # What i in the node list (point on the line) is this?
    i = index // 5
    # Sample the current node and either of its neighbours
    pos2 = buf_pos[i]
    pos3 = buf_pos[i + 1 - (i % 2) * 2]  # (i + 1) if i is even else (i - 1)
    # Convert to ndc
    wpos2 = stdinfo.world_transform * vec4(pos2.xyz, 1.0)
    wpos3 = stdinfo.world_transform * vec4(pos3.xyz, 1.0)
    npos2 = stdinfo.projection_transform * stdinfo.cam_transform * wpos2
    npos3 = stdinfo.projection_transform * stdinfo.cam_transform * wpos3
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
    v_line_width_p = half_line_width_p * 2.0  # noqa
    v_vec_from_node_p = the_vec * l2p  # noqa


@pyshader.python2shader
def vertex_shader_arrow(
    index: (pyshader.RES_INPUT, "VertexId", "i32"),
    buf_pos: (pyshader.RES_BUFFER, (0, 2), Array(vec4)),
    stdinfo: (pyshader.RES_UNIFORM, (0, 0), stdinfo_uniform_type),
    material: (pyshader.RES_UNIFORM, (0, 1), LineMaterial.uniform_type),
    out_pos: (pyshader.RES_OUTPUT, "Position", vec4),
    v_line_width_p: (pyshader.RES_OUTPUT, 0, f32),
    v_vec_from_node_p: (pyshader.RES_OUTPUT, 1, vec2),
):
    # Similar to the normal vertex shader, except we only draw segments,
    # using 3 vertices per node: 6 per segment. 4 for the arrow, and 2
    # to create a degenerate triangle for the space in between. So we
    # only draw caps, no joins.

    # Prepare some numbers
    screen_factor = stdinfo.logical_size.xy / 2.0
    l2p = stdinfo.physical_size.x / stdinfo.logical_size.x
    half_line_width = material.thickness * 0.5  # in logical pixels
    half_line_width_p = half_line_width * l2p  # in physical pixels
    # What i in the node list (point on the line) is this?
    i = index // 3
    # Sample the current node and either of its neighbours
    pos2 = buf_pos[i]
    pos3 = buf_pos[i + 1 - (i % 2) * 2]  # (i + 1) if i is even else (i - 1)
    # Convert to ndc
    wpos2 = stdinfo.world_transform * vec4(pos2.xyz, 1.0)
    wpos3 = stdinfo.world_transform * vec4(pos3.xyz, 1.0)
    npos2 = stdinfo.projection_transform * stdinfo.cam_transform * wpos2
    npos3 = stdinfo.projection_transform * stdinfo.cam_transform * wpos3
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
    v_line_width_p = half_line_width_p * 2.0  # noqa
    v_vec_from_node_p = vec2(0.0, 0.0)  # noqa


# %% Render functions


@register_wgpu_render_function(Line, LineThinMaterial)
def thin_line_renderer(wobject, render_info):
    """ Render function capable of rendering lines.
    """

    material = wobject.material
    geometry = wobject.geometry

    positions1 = geometry.positions

    return [
        {
            "vertex_shader": vertex_shader_thin,
            "fragment_shader": fragment_shader_thin,
            "primitive_topology": wgpu.PrimitiveTopology.line_strip,
            "indices": (positions1.nitems, 1),
            "vertex_buffers": [positions1],
            "bindings0": {
                0: (wgpu.BindingType.uniform_buffer, render_info.stdinfo),
                1: (wgpu.BindingType.uniform_buffer, material.uniform_buffer),
            },
            # "bindings1": {},
            "target": None,  # default
        },
    ]


@register_wgpu_render_function(Line, LineMaterial)
def line_renderer(wobject, render_info):
    """ Render function capable of rendering lines.
    """

    material = wobject.material
    geometry = wobject.geometry

    assert isinstance(material, LineMaterial)

    positions1 = geometry.positions

    if isinstance(material, LineArrowMaterial):
        vert_shader = vertex_shader_arrow
        n = (positions1.nitems // 2) * 2 * 4
    elif isinstance(material, LineSegmentMaterial):
        vert_shader = vertex_shader_segment
        n = (positions1.nitems // 2) * 2 * 5
    else:
        vert_shader = vertex_shader
        n = positions1.nitems * 5

    return [
        # {
        #     "compute_shader": compute_shader,
        #     "indices": (positions1.nitems, 1, 1),
        #     "bindings0": {
        #         0: (wgpu.BindingType.storage_buffer, positions1),
        #         1: (wgpu.BindingType.storage_buffer, positions2),
        #         2: (wgpu.BindingType.uniform_buffer, material.uniform_buffer),
        #     },
        # },
        {
            "vertex_shader": vert_shader,
            "fragment_shader": fragment_shader,
            "primitive_topology": wgpu.PrimitiveTopology.triangle_strip,
            "indices": (n, 1),
            "bindings0": {
                0: (wgpu.BindingType.uniform_buffer, render_info.stdinfo),
                1: (wgpu.BindingType.uniform_buffer, material.uniform_buffer),
                2: (wgpu.BindingType.storage_buffer, positions1),
            },
            # "bindings1": {},
            "target": None,  # default
        },
    ]
