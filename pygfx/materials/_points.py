from ._base import Material
from ..utils import unpack_bitfield


class PointsMaterial(Material):
    """The default material used by Points. Renders (antialiased) disks
    of the given size and color.
    """

    uniform_type = dict(
        color="4xf4",
        size="f4",
    )

    def __init__(
        self,
        color=(1, 1, 1, 1),
        size=1,
        vertex_colors=False,
        vertex_sizes=False,
        **kwargs
    ):
        super().__init__(**kwargs)

        self._map = None
        self.color = color
        self.size = size
        self._vertex_colors = vertex_colors
        self._vertex_sizes = vertex_sizes

    def _wgpu_get_pick_info(self, pick_value):
        # This should match with the shader
        values = unpack_bitfield(pick_value, wobject_id=20, index=26, x=9, y=9)
        return {
            "vertex_index": values["index"],
            "point_coord": (values["x"] - 256.0, values["y"] - 256.0),
        }

    @property
    def color(self):
        """The color of the points (if map is not set)."""
        return self.uniform_buffer.data["color"]

    @color.setter
    def color(self, color):
        color = tuple(color)
        if (color[3] >= 1) != (self.uniform_buffer.data["color"][3] >= 1):
            self._bump_rev()  # rebuild pipeline if this becomes opaque/transparent
        self.uniform_buffer.data["color"] = color
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
    def size(self):
        """The size (diameter) of the points, in logical pixels."""
        return self.uniform_buffer.data["size"]

    @size.setter
    def size(self, size):
        self.uniform_buffer.data["size"] = size
        self.uniform_buffer.update_range(0, 1)

    @property
    def vertex_sizes(self):
        """Whether to use the vertex sizes provided in the geometry."""
        return self._vertex_sizes

    @vertex_sizes.setter
    def vertex_sizes(self, value):
        if value != self._vertex_sizes:
            self._vertex_sizes = value
            self._bump_rev()

    # @property
    # def map(self):
    #     """ The 1D texture map specifying the color for each point.
    #     """
    #     return self._map
    #
    # @map.setter
    # def map(self, map):
    #     self._map = map
    # todo: sizeAttenuation


class GaussianPointsMaterial(PointsMaterial):
    """A material for points, renders Gaussian blobs with a standard
    deviation of 1/6 of the size.
    """


# idea: a MarkerMaterial with more options for the shape, and an edge around the shape.
# Though perhaps such a material should be part of a higher level plotting lib.
