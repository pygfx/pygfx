from python_shader import Struct, vec3, f32

from ._base import Material
from ..utils import array_from_shadertype
from ..datawrappers import Buffer


class LineStripMaterial(Material):

    uniform_type = Struct(color=vec3, thickness=f32)

    def __init__(self, color=(1, 1, 1), thickness=2.0):
        super().__init__()

        array = array_from_shadertype(self.uniform_type)
        self.uniforms = Buffer(array, usage="uniform")
        self.set_color(color)
        self.set_thickness(thickness)

    @property
    def color(self):
        return self.uniforms.data["color"]

    def set_color(self, color):
        self.uniforms.data["color"] = tuple(color)
        self.uniforms.update_range(0, 1)

    @property
    def thickness(self):
        return self.uniforms.data["thickness"]

    def set_thickness(self, thickness):
        self.uniforms.data["thickness"] = thickness
        self.uniforms.update_range(0, 1)
