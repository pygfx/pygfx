from ._base import Material

import python_shader
from python_shader import vec3, mat4, Array


# Idea:
# scene objects (like stdinfo, lights) are at bind group 0
# the objects defined by the geometry are at bind group 1
# the objects defined by the material are at bind group 2
# that leaves (by default) one bind group to be used for something else
# what about vertex data, put all vertices in storage buffers?


@python_shader.python2shader
def compute_shader(
    index: (python_shader.RES_INPUT, "GlobalInvocationId", "i32"),
    pos1: (python_shader.RES_BUFFER, (1, 0), Array(vec3)),
    pos2: (python_shader.RES_BUFFER, (0, 2), Array(Array(3, vec3))),
):
    p = pos1[index]
    pos2[index][0] = p
    pos2[index][1] = vec3(p.x + 1, p.y + 1, p.z + 1)
    pos2[index][2] = vec3(p.x + 2, p.y + 2, p.z + 2)


@python_shader.python2shader
def vertex_shader(
    pos: (python_shader.RES_INPUT, 0, vec3),
    transform: (python_shader.RES_UNIFORM, 0, mat4),
    out_pos=(python_shader.RES_OUTPUT, "Position", vec4),
):
    # out_pos = pos * transform
    out_pos = pos  # noqa - shader assign to input arg


@python_shader.python2shader
def fragment_shader(out_color=(python_shader.RES_OUTPUT, 0, vec4),):
    out_color = vec4(1.0, 0.0, 0.0, 1.0)  # noqa - shader assign to input arg


class LineStripMaterial(Material):
    def __init__(self):
        super().__init__()
        self.uniforms = None
        self.remote_buffer = None  # ...
        # self.shaders = [
        #     (compute_shader, {}, {0: }),
        #     (vertex_shader ),
        #     (fragment_shader ),
        # ]
        self.primitive_topology = wgpu.PrimitiveTopology.line_strip

    def get_wgpu_info(self, stdinfo, geometry):

        return {
            "shaders": [
                (compute_shader, {}, {0: geometry.positions, 1: self.remote_buffer}),
                (vertex_shader, {0: self.remote_buffer}, {0: stdinfo}),
                (fragment_shader, {}, {}),
            ]
        }
