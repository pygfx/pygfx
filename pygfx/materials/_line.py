from ._base import Material
from ..utils import unpack_bitfield


class LineMaterial(Material):
    """The default material to draw lines."""

    uniform_type = dict(
        color="4xf4",
        thickness="f4",
    )

    def __init__(
        self, color=(1, 1, 1, 1), thickness=2.0, vertex_colors=False, **kwargs
    ):
        super().__init__(**kwargs)

        self.color = color
        self.thickness = thickness
        self._vertex_colors = vertex_colors

    def _wgpu_get_pick_info(self, pick_value):
        # This should match with the shader
        values = unpack_bitfield(pick_value, 20, 26, 18)
        return {
            "vertex_index": values[1],
            "segment_coord": (values[2] - 100000) / 100000.0,
        }

    @property
    def color(self):
        return self.uniform_buffer.data["color"]

    @color.setter
    def color(self, color):
        self.uniform_buffer.data["color"] = tuple(color)
        self.uniform_buffer.update_range(0, 1)

    @property
    def vertex_colors(self):
        """Whether to use the vertex colors provided in the geometry."""
        return self._vertex_colors

    @vertex_colors.setter
    def vertex_colors(self, value):
        if value != self._vertex_colors:
            self._vertex_colors = value
            self._bump_rev()

    @property
    def thickness(self):
        """The line thickness expressed in logical pixels."""
        return self.uniform_buffer.data["thickness"]

    @thickness.setter
    def thickness(self, thickness):
        self.uniform_buffer.data["thickness"] = thickness
        self.uniform_buffer.update_range(0, 1)


class LineThinMaterial(LineMaterial):
    """A simple line, drawn with line_strip primitives that has a thickness
    of one physical pixel (the thickness property is ignored).
    """


class LineThinSegmentMaterial(LineMaterial):
    """Simple line segments, drawn with line primitives that has a thickness
    of one physical pixel (the thickness property is ignored).
    """


class LineSegmentMaterial(LineMaterial):
    """A material that renders line segments between each two subsequent points."""


class LineArrowMaterial(LineSegmentMaterial):
    """A material that renders line segments that look like little vectors."""
