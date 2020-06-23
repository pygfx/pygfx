from pyshader import Struct, f32, vec4

from ._base import Material
from ..utils import array_from_shadertype
from ..datawrappers import Buffer


class LineMaterial(Material):
    """ The default material to draw lines.
    """

    uniform_type = Struct(color=vec4, thickness=f32)

    def __init__(self, color=(1, 1, 1, 1), thickness=2.0):
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

    # todo: thickness? maybe rename to width?
    @property
    def thickness(self):
        """ The line thickness expressed in logical pixels.
        """
        return self.uniform_buffer.data["thickness"]

    def set_thickness(self, thickness):
        self.uniform_buffer.data["thickness"] = thickness
        self.uniform_buffer.update_range(0, 1)


class LineThinMaterial(LineMaterial):
    """ A simple line, drawn with line_strip primitives that has a width
    of one physical pixel. The thickness is ignored.
    """


class LineSegmentMaterial(LineMaterial):
    """ A material that renders line segments between each two subsequent points.
    """


class LineArrowMaterial(LineSegmentMaterial):
    """ A material that renders line segments that look like little vectors.
    """
