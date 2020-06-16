import wgpu  # only for flags/enums
import pyshader
from pyshader import vec4, Array

from . import register_wgpu_render_function, stdinfo_uniform_type
from ...objects import Line
from ...materials import LineStripMaterial
from ...datawrappers import Buffer

# todo: use a compute shader or instancing?
# https://wwwtyro.net/2019/11/18/instanced-lines.html


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
    # io
    in_pos: (pyshader.RES_INPUT, 0, vec4),
    out_pos: (pyshader.RES_OUTPUT, "Position", vec4),
    stdinfo: (pyshader.RES_UNIFORM, (0, 0), stdinfo_uniform_type),
    # resources
):
    world_pos = stdinfo.world_transform * vec4(in_pos.xyz, 1.0)
    ndc_pos = stdinfo.projection_transform * stdinfo.cam_transform * world_pos

    out_pos = ndc_pos  # noqa - shader assign to input arg


@pyshader.python2shader
def fragment_shader(
    out_color: (pyshader.RES_OUTPUT, 0, vec4),
    stdinfo: (pyshader.RES_UNIFORM, (0, 0), stdinfo_uniform_type),
    material: (pyshader.RES_UNIFORM, (0, 1), LineStripMaterial.uniform_type),
):
    out_color = vec4(material.color.rgb, 1.0)  # noqa - shader assign to input arg


@register_wgpu_render_function(Line, LineStripMaterial)
def line_renderer(wobject, render_info):
    """ Render function capable of rendering lines.
    """

    material = wobject.material
    geometry = wobject.geometry

    assert isinstance(material, LineStripMaterial)

    positions1 = geometry.positions

    positions2_nitems = -1
    if hasattr(geometry, "_wgpu_line_renderer_positions2"):
        positions2_nitems = geometry._wgpu_line_renderer_positions2.nitems
    if positions2_nitems != positions1.nitems * 2:
        # Create buffer that only exist on the GPU, we provide stub data so that
        # the buffer knows type and strides (needed when used as vertex buffer)
        geometry._wgpu_line_renderer_positions2 = Buffer(
            nbytes=positions1.nitems * 2 * 4 * 4,
            nitems=positions1.nitems * 2,
            format="float4",
            usage="vertex|storage",
        )
    positions2 = geometry._wgpu_line_renderer_positions2

    return [
        {
            "compute_shader": compute_shader,
            "indices": (positions1.nitems, 1, 1),
            "bindings0": {
                0: (wgpu.BindingType.storage_buffer, positions1),
                1: (wgpu.BindingType.storage_buffer, positions2),
                2: (wgpu.BindingType.uniform_buffer, material.uniform_buffer),
            },
        },
        {
            "vertex_shader": vertex_shader,
            "fragment_shader": fragment_shader,
            "primitive_topology": wgpu.PrimitiveTopology.triangle_strip,
            "indices": positions1.nitems * 2,
            "vertex_buffers": [positions2],
            "bindings0": {
                0: (wgpu.BindingType.uniform_buffer, render_info.stdinfo),
                1: (wgpu.BindingType.uniform_buffer, material.uniform_buffer),
            },
            "target": None,  # default
        },
    ]
