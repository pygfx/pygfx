from ._base import Material
from ..utils import array_from_shadertype
from ..resources import Buffer


class LineMaterial(Material):
    """The default material to draw lines."""

    uniform_type = dict(
        color=("float32", 4),
        thickness=("float32",),
    )

    def __init__(self, color=(1, 1, 1, 1), thickness=2.0, vertex_colors=False):
        super().__init__()

        self.uniform_buffer = Buffer(
            array_from_shadertype(self.uniform_type), usage="UNIFORM"
        )
        self.set_color(color)
        self.set_thickness(thickness)

        self._vertex_colors = vertex_colors

    def _wgpu_get_pick_info(self, pick_value):
        # The instance is zero while renderer doesn't support instancing
        instance = pick_value[1]
        vertex = pick_value[2]
        vertex_sub = pick_value[3] / 1048576
        return {"instance_index": instance, "vertex_index": vertex + vertex_sub}

    @property
    def color(self):
        return self.uniform_buffer.data["color"]

    def set_color(self, color):
        self.uniform_buffer.data["color"] = tuple(color)
        self.uniform_buffer.update_range(0, 1)

    @property
    def vertex_colors(self):
        return self._vertex_colors

    @vertex_colors.setter
    def vertex_colors(self, value):
        if value != self._vertex_colors:
            self._vertex_colors = value
            self._bump_rev()

    # todo: thickness? maybe rename to width?
    @property
    def thickness(self):
        """The line thickness expressed in logical pixels."""
        return self.uniform_buffer.data["thickness"]

    def set_thickness(self, thickness):
        self.uniform_buffer.data["thickness"] = thickness
        self.uniform_buffer.update_range(0, 1)


class LineThinMaterial(LineMaterial):
    """A simple line, drawn with line_strip primitives that has a width
    of one physical pixel. The thickness is ignored.
    """


class LineThinSegmentMaterial(LineMaterial):
    """Simple line segments, drawn with line primitives that has a width
    of one physical pixel. The thickness is ignored.
    """


class LineSegmentMaterial(LineMaterial):
    """A material that renders line segments between each two subsequent points."""


class LineArrowMaterial(LineSegmentMaterial):
    """A material that renders line segments that look like little vectors."""
