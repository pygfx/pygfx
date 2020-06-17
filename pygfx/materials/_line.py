from pyshader import Struct, vec3, f32

from ._base import Material
from ..utils import array_from_shadertype
from ..datawrappers import Buffer


class LineStripMaterial(Material):

    uniform_type = Struct(color=vec3, thickness=f32)

    def __init__(self, color=(1, 1, 1), thickness=2.0):
        super().__init__()

        self.uniform_buffer = Buffer(
            array_from_shadertype(self.uniform_type), usage="UNIFORM"
        )
        self.set_color(color)
        self.set_thickness(thickness)

    @property
    def color(self):
        return self.uniform_buffer.data["color"]

    def set_color(self, color):
        self.uniform_buffer.data["color"] = tuple(color)
        self.uniform_buffer.update_range(0, 1)

    @property
    def thickness(self):
        return self.uniform_buffer.data["thickness"]

    def set_thickness(self, thickness):
        self.uniform_buffer.data["thickness"] = thickness
        self.uniform_buffer.update_range(0, 1)
