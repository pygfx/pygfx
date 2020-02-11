import wgpu
from python_shader import python2shader, RES_INPUT, RES_OUTPUT


from ._world_object import WorldObject


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


class Triangle(WorldObject):
    """ Our first WorldObject, a triangle!.
    Forget about sources and materials for now ...
    """

    def __init__(self, pos=(0, 0, 0)):
        super().__init__()
        self._pos = [float(x) for x in pos]
        assert len(self._pos) == 3  # x, y, z

    def get_renderer_info_wgpu(self):
        uniforms = [self._pos]
        return {
            "shaders": [vertex_shader, fragment_shader],
            "primitiveTopology": wgpu.PrimitiveTopology.triangle_list,
            "uniforms": uniforms,
        }
