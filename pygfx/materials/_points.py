from ._base import Material, ColorMode
from ..resources import Texture
from ..utils import unpack_bitfield, Color


class PointsMaterial(Material):
    """Point default material.

    Renders (antialiased) disks of the given size and color.

    Parameters
    ----------
    color : Color
        The uniform color of the points (used depending on the ``color_mode``).
    size : int
        The size (diameter) of the points in screen space (px). Ignored if
        vertex_size is True.
    color_mode : enum or str
        The mode by which the line is coloured. Default 'auto'.
    vertex_sizes : bool
        If True, use the vertex sizes provided in the geometry to set point
        sizes.
    map : Texture
        The texture map specifying the color for each texture coordinate.
    map_interpolation: str
        The method to interpolate the color map. Either 'nearest' or 'linear'. Default 'linear'.
    kwargs : Any
        Additional kwargs will be passed to the :class:`material base class
        <pygfx.Material>`.

    """

    uniform_type = dict(
        Material.uniform_type,
        color="4xf4",
        size="f4",
    )

    def __init__(
        self,
        color=(1, 1, 1, 1),
        size=1,
        color_mode="auto",
        vertex_sizes=False,
        map=None,
        map_interpolation="linear",
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.color = color
        self.color_mode = color_mode
        self.map = map
        self.map_interpolation = map_interpolation
        self.size = size
        self._vertex_sizes = bool(vertex_sizes)

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
        return Color(self.uniform_buffer.data["color"])

    @color.setter
    def color(self, color):
        color = Color(color)
        self.uniform_buffer.data["color"] = color
        self.uniform_buffer.update_range(0, 1)
        self._store.color_is_transparent = color.a < 1

    @property
    def color_is_transparent(self):
        """Whether the color is (semi) transparent (i.e. not fully opaque)."""
        return self._store.color_is_transparent

    @property
    def color_mode(self):
        """The way that color is applied to the mesh.

        * auto: switch between `uniform` and `vertex_map`, depending on whether `map` is set.
        * uniform: use the material's color property for the whole mesh.
        * vertex: use the geometry `colors` buffer, one color per vertex.
        * vertex_map: use the geometry texcoords buffer to sample (per vertex) in the material's ``map`` texture.
        """
        return self._store.color_mode

    @color_mode.setter
    def color_mode(self, value):
        if isinstance(value, ColorMode):
            pass
        elif isinstance(value, str):
            if value.startswith("ColorMode."):
                value = value.split(".")[-1]
            try:
                value = getattr(ColorMode, value.lower())
            except AttributeError:
                raise ValueError(f"Invalid color_mode: '{value}'")
        else:
            raise TypeError(f"Invalid color_mode class: {value.__class__.__name__}")
        if value == ColorMode.face or value == ColorMode.face_map:
            raise ValueError(f"Points cannot have color_mode {value}")
        self._store.color_mode = value

    @property
    def vertex_colors(self):
        return self.color_mode == ColorMode.vertex

    @vertex_colors.setter
    def vertex_colors(self, value):
        raise DeprecationWarning(
            "vertex_colors is deprecated, use ``color_mode='vertex'``"
        )

    @property
    def size(self):
        """The size (diameter) of the points, in logical pixels."""
        return float(self.uniform_buffer.data["size"])

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
        value = bool(value)
        if value != self._vertex_sizes:
            self._vertex_sizes = value

    @property
    def map(self):
        """The texture map specifying the color for each texture coordinate.
        The dimensionality of the map can be 1D, 2D or 3D, but should match the
        number of columns in the geometry's texcoords.
        """
        return self._map

    @map.setter
    def map(self, map):
        assert map is None or isinstance(map, Texture)
        self._map = map

    @property
    def map_interpolation(self):
        """The method to interpolate the colormap. Either 'nearest' or 'linear'."""
        return self._store.map_interpolation

    @map_interpolation.setter
    def map_interpolation(self, value):
        assert value in ("nearest", "linear")
        self._store.map_interpolation = value

    # todo: sizeAttenuation


class GaussianPointsMaterial(PointsMaterial):
    """A material for points, renders Gaussian blobs with a standard
    deviation of 1/6 of the size.
    """


# idea: a MarkerMaterial with more options for the shape, and an edge around the shape.
# Though perhaps such a material should be part of a higher level plotting lib.
