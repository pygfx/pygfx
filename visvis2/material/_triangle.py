from ._base import Material

import wgpu  # only for flags/enums
from python_shader import python2shader, RES_INPUT, RES_OUTPUT


@python2shader
def vertex_shader(
    index: (RES_INPUT, "VertexId", "i32"),
    pos: (RES_OUTPUT, "Position", "vec4"),
    color: (RES_OUTPUT, 0, "vec3"),
):
    positions = [vec2(+0.0, -0.5), vec2(+0.5, +0.5), vec2(-0.5, +0.7)]

    p = positions[index]
    pos = vec4(p, 0.0, 1.0)  # noqa
    color = vec3(p, 0.5)  # noqa


@python2shader
def fragment_shader(
    in_color: (RES_INPUT, 0, "vec3"),
    out_color: (RES_OUTPUT, 0, "vec4"),
    # u_color: (RES_UNIFORM, 0, "vec3"),
):
    out_color = vec4(in_color, 0.1)  # noqa


class TriangleMaterial(Material):
    def __init__(self):
        super().__init__()
        self.shaders = {
            "vertex": vertex_shader,
            "fragment": fragment_shader,
        }
        self.primitive_topology = wgpu.PrimitiveTopology.triangle_list
