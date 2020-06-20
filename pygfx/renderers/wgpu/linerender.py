import wgpu  # only for flags/enums
import pyshader
from pyshader import f32, vec2, vec4, Array

from . import register_wgpu_render_function, stdinfo_uniform_type
from ...objects import Line
from ...materials import LineStripMaterial

# from ...datawrappers import Buffer

# ## Notes
#
# Rendering lines on the GPU is hard! The approach used here uses VertexId
# and storage buffers instead of vertex buffers. That way we can create
# 5 vertices for each point on the line. These vertices are positioned such
# that rendering with triangle_strip results in a tick line. More info below.
#
# An alternative (used by e.g. ThreeJS) is to draw the geometry of a
# line segment many times using instancing. Note that some Intel drivers
# limit the number of instances to about 5-10k, which is not much considering
# that lines consisting of millions of points can normally run realtime.
# https://wwwtyro.net/2019/11/18/instanced-lines.html
#
# Another alternative is a geometry shader, but wgpu does not have that.
#
# Another alternative is to use a geometry shader to prepare the triangle-based
# geometry and store it in a buffer, which can be used as a vertex buffer in
# the render pass. The downside of this approach is the memory consumption.
# But a hybrid approach may be viable: the current approach using VertexId,
# but preparing some data in a buffer.


# This compute shader is wrong and unused, but left as an example for what
# such a "geometry shader pass" could look like.
@pyshader.python2shader
def compute_shader(
    index: (pyshader.RES_INPUT, "GlobalInvocationId", "i32"),
    pos1: (pyshader.RES_BUFFER, (0, 0), Array(vec4)),
    pos2: (pyshader.RES_BUFFER, (0, 1), Array(vec4)),
    material: (pyshader.RES_UNIFORM, (0, 2), LineStripMaterial.uniform_type),
):
    p = pos1[index] * 1.0
    dz = material.thickness
    pos2[index * 2 + 0] = vec4(p.x, p.y + dz, p.z, 1.0)
    pos2[index * 2 + 1] = vec4(p.x, p.y - dz, p.z, 1.0)


# todo: Figure out what to do with transparency ...
# - make the actual ine thicker to account for aa


@pyshader.python2shader
def vertex_shader(
    index: (pyshader.RES_INPUT, "VertexId", "i32"),
    buf_pos: (pyshader.RES_BUFFER, (0, 2), Array(vec4)),
    stdinfo: (pyshader.RES_UNIFORM, (0, 0), stdinfo_uniform_type),
    material: (pyshader.RES_UNIFORM, (0, 1), LineStripMaterial.uniform_type),
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
    # - These calculations are done for 5x (yeah, bit of a waste), we select
    #   just one as output.
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
    # - also implement bevel joins, maybe miters too?
    # - also implement different caps.
    # - we can prepare the nodes' screen coordinates in a compute shader.

    # Prepare some numbers
    eps = 0.00001
    screen_factor = stdinfo.logical_size.xy / 2.0
    l2p = stdinfo.physical_size.x / stdinfo.logical_size.x
    half_line_width = material.thickness * 0.5  # in logical pixels
    half_line_width_p = half_line_width * l2p  # in physical pixels

    # What i in the node list (point on the line) is this?
    i = i32(f32(index) / 5.0)

    # Sample the vertex and it's two neighbours, and convert to NDC
    pos1 = buf_pos[i - 1]
    pos2 = buf_pos[i]
    pos3 = buf_pos[i + 1]
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
    v1 = ppos2.xy - ppos1.xy
    v2 = ppos3.xy - ppos2.xy

    if pos1.w == 0.0:
        # This is the first point on the line: create a cap.
        v1 = v2
        nc = normalize(vec2(v2.y, 0.0 - v2.x))
        nd = vec2(0.0, 0.0) - nc
        na = nd
        nb = nd - normalize(v2)
        ne = nc - normalize(v2)
    elif pos3.w == 0.0:
        # This is the last point on the line: create a cap.
        v2 = v1
        na = normalize(vec2(v1.y, 0.0 - v1.x))
        nb = vec2(0.0, 0.0) - na
        ne = nb + normalize(v1)
        nc = na + normalize(v1)
        nd = na
    else:
        # Create a join
        na = normalize(vec2(v1.y, 0.0 - v1.x))
        nb = vec2(0.0, 0.0) - na
        nc = normalize(vec2(v2.y, 0.0 - v2.x))
        nd = vec2(0.0, 0.0) - nc

        # Determine the angle between two of the normals. If this angle is smaller
        # than zero, the inside of the join is at nb/nd, otherwise it is at na/nc.
        angle = atan2(na.y, na.x) - atan2(nc.y, nc.x)
        angle = (angle + math.pi) % (2.0 * math.pi) - math.pi

        vec_mag_inv = cos(0.5 * angle)
        if angle < 0.0:
            # n_longest = mix(nb, nd, length(v2) / (length(v1) + length(v2) + eps))
            # n_between = normalize((nb + nd) * 0.5) / (vec_mag_inv + eps)
            # n_between = mix(n_longest, n_between, min(vec_mag_inv, 0.2) * 5.0)
            n_between = normalize(0.5 * (nb + nd)) / (vec_mag_inv + eps)
            ne = vec2(0.0, 0.0) - n_between  # the extra point
        else:
            # n_longest = mix(na, nc, length(v2) / (length(v1) + length(v2) + eps))
            # n_between = normalize((na + nc) * 0.5)  / (vec_mag_inv + eps)
            # n_between = mix(n_longest, n_between, min(vec_mag_inv, 0.2) * 5.0)
            n_between = normalize(0.5 * (na + nc)) / (vec_mag_inv + eps)
            ne = vec2(0.0, 0.0) - n_between  # the extra point

    # Select the correct vector, and the corresponding vertex pos.
    # Note that all except ne are unit.
    vectors = [na, nb, ne, nc, nd]
    the_vec = vectors[index % 5]
    ppos = ppos2 + the_vec * half_line_width
    npos = vec4(ppos / screen_factor - 1.0, npos2.zw)

    # Outputs
    out_pos = npos  # noqa - shader assign
    v_line_width_p = half_line_width_p * 2.0  # noqa - shader assign
    v_vec_from_node_p = the_vec * half_line_width * l2p  # noqa


@pyshader.python2shader
def fragment_shader(
    in_coord: (pyshader.RES_INPUT, "FragCoord", vec4),
    v_line_width_p: (pyshader.RES_INPUT, 0, f32),
    v_vec_from_node_p: (pyshader.RES_INPUT, 1, vec2),
    stdinfo: (pyshader.RES_UNIFORM, (0, 0), stdinfo_uniform_type),
    material: (pyshader.RES_UNIFORM, (0, 1), LineStripMaterial.uniform_type),
    out_color: (pyshader.RES_OUTPUT, 0, vec4),
    out_depth: (pyshader.RES_OUTPUT, "FragDepth", f32),
):
    dist_to_node_p = length(v_vec_from_node_p)
    if dist_to_node_p > v_line_width_p * 0.5:
        return  # discard

    alpha = 1.0
    aa_width = 1.2
    alpha = ((0.5 * v_line_width_p) - abs(dist_to_node_p)) / aa_width
    alpha = min(1.0, alpha) ** 2

    color = material.color
    out_color = vec4(color.rgb, min(1.0, color.a) * alpha)  # noqa - shader assign

    # The outer edges with lower alpha for aa are pushed a bit back to avoid artifacts.
    out_depth = in_coord.z + 0.0001 * (0.8 - min(0.8, alpha))  # noqa


@register_wgpu_render_function(Line, LineStripMaterial)
def line_renderer(wobject, render_info):
    """ Render function capable of rendering lines.
    """

    material = wobject.material
    geometry = wobject.geometry

    assert isinstance(material, LineStripMaterial)

    positions1 = geometry.positions

    # positions2_nitems = -1
    # if hasattr(geometry, "_wgpu_line_renderer_positions2"):
    #     positions2_nitems = geometry._wgpu_line_renderer_positions2.nitems
    # if positions2_nitems != positions1.nitems * 2:
    #     # Create buffer that only exist on the GPU, we provide stub data so that
    #     # the buffer knows type and strides (needed when used as vertex buffer)
    #     geometry._wgpu_line_renderer_positions2 = Buffer(
    #         nbytes=positions1.nitems * 2 * 4 * 4,
    #         nitems=positions1.nitems * 2,
    #         format="float4",
    #         usage="vertex|storage",
    #     )
    # positions2 = geometry._wgpu_line_renderer_positions2

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
            "vertex_shader": vertex_shader,
            "fragment_shader": fragment_shader,
            "primitive_topology": wgpu.PrimitiveTopology.triangle_strip,
            "indices": (positions1.nitems * 5, 1),
            "bindings0": {
                0: (wgpu.BindingType.uniform_buffer, render_info.stdinfo),
                1: (wgpu.BindingType.uniform_buffer, material.uniform_buffer),
                2: (wgpu.BindingType.storage_buffer, positions1),
            },
            # "bindings1": {},
            "target": None,  # default
        },
    ]
