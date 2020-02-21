from ._base import Material, stdinfo_type, array_from_shader_type

import wgpu  # only for flags/enums
from python_shader import python2shader, RES_INPUT, RES_OUTPUT, RES_UNIFORM
from python_shader import Struct, vec3


uniform_type = Struct(color=vec3)


@python2shader
def vertex_shader(
    # input/output
    index: (RES_INPUT, "VertexId", "i32"),
    pos: (RES_OUTPUT, "Position", "vec4"),
    color: (RES_OUTPUT, 0, "vec3"),
    # stuff from scene, geometry, material
    stdinfo: (RES_UNIFORM, (0, 0), stdinfo_type),
):
    # Draw in NDC
    # positions = [vec2(+0.0, -0.5), vec2(+0.5, +0.5), vec2(-0.5, +0.7)]
    # p = positions[index]

    # Draw in screen coordinates
    positions = [vec2(10.0, 10.0), vec2(90.0, 10.0), vec2(10.0, 90.0)]
    p = 2.0 * positions[index] / stdinfo.logical_size - 1.0

    pos = vec4(p, 0.0, 1.0)  # noqa
    color = vec3(p, 0.5)  # noqa


@python2shader
def fragment_shader(
    in_color: (RES_INPUT, 0, "vec3"),
    out_color: (RES_OUTPUT, 0, "vec4"),
    uniforms: (RES_UNIFORM, (2, 0), uniform_type),
):
    color = uniforms.color
    out_color = vec4(color, 0.1)  # noqa


class TriangleMaterial(Material):
    def __init__(self):
        super().__init__()
        self.shaders = {
            "vertex": vertex_shader,
            "fragment": fragment_shader,
        }

        self.primitive_topology = wgpu.PrimitiveTopology.triangle_list

        # Instantiate uniforms. This produces a ctypes struct object.
        # The renderer will use it to create a mapped buffer, create a
        # new ctypes object with the same type, mapped onto that buffer,
        # and copy the original data over. So we can just assign fields
        # of our uniforms object and it Just Works!
        # self.uniforms = uniform_type(color=(1, 0, 0))

        uniform_array = array_from_shader_type(uniform_type)
        self.bindings[0] = uniform_array, 2
        self.set_color((1, 0, 0))

    # @property
    # def uniforms(self):
    #     return self.bindings[0][0]

    def set_color(self, color):
        self.bindings[0][0]["color"] = color
        # self.uniforms.color = color  # ctypes handles the setting

    # def _get_wgpu_info(self):
    #
    #     return {"shaders": [vertex_shader fragment_shader],
    #             "primitive_topology": wgpu.PrimitiveTopology.triangle_list,
    #             "bindings": self.bindings,  # note: renderer may replace the arrays
    #     }
