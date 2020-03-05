import numpy as np
import wgpu  # only for flags/enums
import python_shader
from python_shader import vec4, Array

from . import register_wgpu_render_function, stdinfo_uniform_type
from ...objects import Mesh  # todo -> Line
from ...material import LineStripMaterial
from ..._wrappers import BufferWrapper

# todo: use a compute shader or instancing?
# https://wwwtyro.net/2019/11/18/instanced-lines.html


@python_shader.python2shader
def compute_shader(
    index: (python_shader.RES_INPUT, "GlobalInvocationId", "i32"),
    pos1: (python_shader.RES_BUFFER, (0, 0), Array(vec4)),
    pos2: (python_shader.RES_BUFFER, (0, 1), Array(vec4)),
    material: (python_shader.RES_UNIFORM, (0, 2), LineStripMaterial.uniform_type),
):
    p = pos1[index] * 1.0
    dz = material.thickness
    pos2[index * 2 + 0] = vec4(p.x, p.y + dz, p.z, 1.0)
    pos2[index * 2 + 1] = vec4(p.x, p.y - dz, p.z, 1.0)


@python_shader.python2shader
def vertex_shader(
    # io
    in_pos: (python_shader.RES_INPUT, 0, vec4),
    out_pos: (python_shader.RES_OUTPUT, "Position", vec4),
    stdinfo: (python_shader.RES_UNIFORM, (0, 0), stdinfo_uniform_type),
    # resources
):
    world_pos = stdinfo.world_transform * vec4(in_pos.xyz, 1.0)
    ndc_pos = stdinfo.projection_transform * stdinfo.cam_transform * world_pos

    out_pos = ndc_pos  # noqa - shader assign to input arg


@python_shader.python2shader
def fragment_shader(
    out_color: (python_shader.RES_OUTPUT, 0, vec4),
    stdinfo: (python_shader.RES_UNIFORM, (0, 0), stdinfo_uniform_type),
    material: (python_shader.RES_UNIFORM, (0, 1), LineStripMaterial.uniform_type),
):
    out_color = vec4(material.color.rgb, 1.0)  # noqa - shader assign to input arg


@register_wgpu_render_function(Mesh, LineStripMaterial)
def line_renderer(wobject, render_info):
    """ Render function capable of rendering lines.
    """

    material = wobject.material
    geometry = wobject.geometry

    assert isinstance(material, LineStripMaterial)

    if not hasattr(geometry, "_line_renderer_positions2"):
        # Create buffer that only exist on the GPU, we provide stub data so that
        # the buffer knows type and strides (needed when used as vertex buffer)
        stub_array = np.zeros((0, 4), np.float32)
        geometry._line_renderer_positions2 = BufferWrapper(
            stub_array, nbytes=0, mapped=False, usage="vertex|storage"
        )

    positions1 = geometry.positions
    positions2 = geometry._line_renderer_positions2

    n = len(positions1.data)  # number of vertices
    positions2.set_nbytes(2 * positions1.nbytes)

    return [
        {
            "compute_shader": compute_shader,
            "indices": (n, 1, 1),
            "bindings0": [positions1, positions2, material.uniforms],
        },
        {
            "vertex_shader": vertex_shader,
            "fragment_shader": fragment_shader,
            "primitive_topology": wgpu.PrimitiveTopology.triangle_strip,
            "indices": n * 2,
            "vertex_buffers": [positions2],
            "bindings0": [render_info.stdinfo, material.uniforms],
            "target": None,  # default
        },
    ]
