import wgpu  # only for flags/enums
import pyshader
from pyshader import f32, vec4, Array

from . import register_wgpu_render_function, stdinfo_uniform_type
from ...objects import Line
from ...materials import LineStripMaterial
# from ...datawrappers import Buffer

# todo: use a compute shader or instancing?
# https://wwwtyro.net/2019/11/18/instanced-lines.html
# * what happens to transparency when using instancing (as segments overlap)?


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


@pyshader.python2shader
def vertex_shader(
    index: (pyshader.RES_INPUT, "VertexId", "i32"),
    buf_pos: (pyshader.RES_BUFFER, (0, 2), Array(vec4)),
    stdinfo: (pyshader.RES_UNIFORM, (0, 0), stdinfo_uniform_type),
    material: (pyshader.RES_UNIFORM, (0, 1), LineStripMaterial.uniform_type),
    out_pos: (pyshader.RES_OUTPUT, "Position", vec4),
    v_line_width_p: (pyshader.RES_OUTPUT, 0, f32),
    v_dist2center_p: (pyshader.RES_OUTPUT, 1, f32),
):
    # world_pos = stdinfo.world_transform * vec4(in_pos.xyz, 1.0)
    # ndc_pos = stdinfo.projection_transform * stdinfo.cam_transform * world_pos

    #
    #            /  o     vertex 3
    #           /  /  /
    #          d  /  /
    #   - - - b  /  /     corners rectangles a, b, c, d
    #   o-------o  /
    #   - - - - a c
    #                vertex 2
    #  vertex 1

    i = i32(f32(index) / 4.0)
    j = index % 4

    screen_factor = stdinfo.logical_size.xy / 2.0
    half_line_width = material.thickness * 0.5  # in logical pixels
    half_line_width_p = (
        half_line_width * stdinfo.physical_size.x / stdinfo.logical_size.x
    )

    # Sample the vertex and it's two neighbours, and convert to NDC
    # todo: this would be more efficient if we also had a a combined transform in the uniform
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
    v1 = normalize(ppos2.xy - ppos1.xy)
    v2 = normalize(ppos3.xy - ppos2.xy)
    if pos1.w == 0.0:  # This is the first vertex
        v1 = v2
    elif pos3.w == 0.0:  # This is the last vertex
        v2 = v1
    na = vec2(v1.y, 0.0 - v1.x)
    nb = vec2(0.0, 0.0) - na
    nc = vec2(v2.y, 0.0 - v2.x)
    nd = vec2(0.0, 0.0) - nc

    # Determine if this is the inside or ourside of the corner between the segments
    # For unit vectors, cos(angle) = dot(na, nc)
    angle = atan2(na.y, na.x) - atan2(nc.y, nc.x)
    angle = angle + 2.0 * math.pi if angle < 0.0 else angle
    if angle < 1.57079632:
        na = normalize((na + nc) * 0.5)
        nc = na
    else:
        nb = normalize((nb + nd) * 0.5)
        nd = nb

    # Depending on what vertex index this is, we emit the appropriate vertex pos
    if j == 0:
        ppos = ppos2 + na * half_line_width
        npos = vec4(ppos / screen_factor - 1.0, npos1.zw)
        dist = half_line_width_p
    elif j == 1:
        ppos = ppos2 + nb * half_line_width
        npos = vec4(ppos / screen_factor - 1.0, npos1.zw)
        dist = 0.0 - half_line_width_p
    elif j == 2:
        ppos = ppos2 + nc * half_line_width
        npos = vec4(ppos / screen_factor - 1.0, npos2.zw)
        dist = half_line_width_p
    else:
        ppos = ppos2 + nd * half_line_width
        npos = vec4(ppos / screen_factor - 1.0, npos2.zw)
        dist = 0.0 - half_line_width_p

    out_pos = npos  # noqa - shader assign
    v_line_width_p = half_line_width_p * 2.0  # noqa - shader assign
    v_dist2center_p = dist  # noqa - shader assign


@pyshader.python2shader
def fragment_shader(
    v_line_width_p: (pyshader.RES_INPUT, 0, f32),
    v_dist2center_p: (pyshader.RES_INPUT, 1, f32),
    out_color: (pyshader.RES_OUTPUT, 0, vec4),
    stdinfo: (pyshader.RES_UNIFORM, (0, 0), stdinfo_uniform_type),
    material: (pyshader.RES_UNIFORM, (0, 1), LineStripMaterial.uniform_type),
):
    aa_width = 1.2
    alpha = ((0.5 * v_line_width_p) - abs(v_dist2center_p)) / aa_width
    alpha = min(1.0, alpha) ** 2

    color = material.color
    out_color = vec4(color.rgb, min(1.0, color.a) * alpha)  # noqa - shader assign


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
            "indices": (positions1.nitems * 4, 1),
            "bindings0": {
                0: (wgpu.BindingType.uniform_buffer, render_info.stdinfo),
                1: (wgpu.BindingType.uniform_buffer, material.uniform_buffer),
                2: (wgpu.BindingType.storage_buffer, positions1),
            },
            # "bindings1": {},
            "target": None,  # default
        },
    ]
