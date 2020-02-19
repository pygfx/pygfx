from ._base import Material, stdinfo_type

import wgpu  # only for flags/enums
import python_shader
from python_shader import vec3, vec4


@python_shader.python2shader
def vertex_shader(
    stdinfo: (python_shader.RES_UNIFORM, 0, stdinfo_type),
    pos: (python_shader.RES_INPUT, 0, vec3),
    out_pos: (python_shader.RES_OUTPUT, "Position", vec4),
):
    out_pos = stdinfo.world_transform * vec4(pos, 1.0) # noqa - shader assign to input arg


@python_shader.python2shader
def fragment_shader(out_color: (python_shader.RES_OUTPUT, 0, vec4),):
    out_color = vec4(1.0, 0.0, 0.0, 1.0)  # noqa - shader assign to input arg


class MeshBasicMaterial(Material):
    def __init__(self):
        super().__init__()
        self.uniforms = None
        self.shaders = {
            "vertex": vertex_shader,
            "fragment": fragment_shader,
        }
        self.primitive_topology = wgpu.PrimitiveTopology.triangle_list
