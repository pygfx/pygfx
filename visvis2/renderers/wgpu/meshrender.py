import wgpu  # only for flags/enums
import python_shader
from python_shader import vec4

from . import register_wgpu_render_function, stdinfo_uniform_type
from ...objects import Mesh
from ...materials import Material
from ...datawrappers import BufferWrapper


@python_shader.python2shader
def vertex_shader(
    # input and output
    position: (python_shader.RES_INPUT, 0, vec4),
    out_pos: (python_shader.RES_OUTPUT, "Position", vec4),
    # uniform and storage buffers
    stdinfo: (python_shader.RES_UNIFORM, (0, 0), stdinfo_uniform_type),
):
    world_pos = stdinfo.world_transform * vec4(position.xyz, 1.0)
    ndc_pos = stdinfo.projection_transform * stdinfo.cam_transform * world_pos

    out_pos = ndc_pos  # noqa - shader assign to input arg


@python_shader.python2shader
def fragment_shader(out_color: (python_shader.RES_OUTPUT, 0, vec4),):
    out_color = vec4(1.0, 0.0, 0.0, 1.0)  # noqa - shader assign to input arg


@register_wgpu_render_function(Mesh, Material)
def mesh_renderer(wobject, render_info):
    """ Render function capable of rendering meshes.
    """

    geometry = wobject.geometry
    material = wobject.material  # noqa

    # Get stuff from material

    # ...

    # Get stuff from geometry

    # Use index buffer if present on the geometry
    index_buffer = getattr(geometry, "index", None)
    index_buffer = index_buffer if isinstance(index_buffer, BufferWrapper) else None

    # All buffer objects are considered vertex buffers
    vertex_buffers = [
        val
        for val in geometry.__dict__.values()
        if isinstance(val, BufferWrapper) and val is not index_buffer
    ]
    if not vertex_buffers:
        raise ValueError("Cannot get default wgpu_info: no vertex buffers found.")

    if index_buffer:
        n = len(index_buffer.data)
    else:
        n = len(vertex_buffers[0].data)

    # Put it together!

    return [
        {
            "vertex_shader": vertex_shader,
            "fragment_shader": fragment_shader,
            "primitive_topology": wgpu.PrimitiveTopology.triangle_list,
            "indices": (range(n), range(1)),
            "index_buffer": index_buffer,
            "vertex_buffers": vertex_buffers,
            "bindings0": [render_info.stdinfo],
        }
    ]
