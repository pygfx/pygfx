from ._base import Material, ColorMode, SizeMode
from ..resources import Texture
from ..utils import unpack_bitfield, Color


class PointsMaterial(Material):
    """Point default material.

    Renders disks of the given size and color.

    Parameters
    ----------
    size : float
        The size (diameter) of the points in logical pixels. Default 4.
    size_space : str
        The coordinate space in which the size is expressed ('screen', 'world', 'model'). Default 'screen'.
    size_mode : enum or str
        The mode by which the points are sized. Default 'uniform'.
    color : Color
        The uniform color of the points (used depending on the ``color_mode``).
    color_mode : enum or str
        The mode by which the points are coloured. Default 'auto'.
    map : Texture
        The texture map specifying the color for each texture coordinate.
    map_interpolation: str
        The method to interpolate the color map. Either 'nearest' or 'linear'. Default 'linear'.
    aa : bool
        Whether or not the points are anti-aliased in the shader. Default True.
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
        size=4,
        size_space="screen",
        size_mode="uniform",
        *,
        color=(1, 1, 1, 1),
        color_mode="auto",
        map=None,
        map_interpolation="linear",
        aa=True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.size = size
        self.size_space = size_space
        self.size_mode = size_mode
        self.color = color
        self.color_mode = color_mode
        self.map = map
        self.map_interpolation = map_interpolation
        self.aa = aa

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
    def aa(self):
        """Whether the point's visual edge is anti-aliased.

        Aliasing gives prettier results by producing semi-transparent fragments
        at the edges. Points smaller than one physical pixel are also diminished
        by making them more transparent.

        Note that by default, pygfx uses SSAA to anti-alias the total renderered
        result. Point-based aa results in additional improvement.

        Because semi-transparent fragments are introduced, it may affect how the
        points blends with other (semi-transparent) objects. It can also affect
        performance for very large datasets. In particular, when the points itself
        are opaque, the point is (in most blend modes) drawn twice to account for
        both the opaque and semi-transparent fragments.
        """
        return self._store.aa

    @aa.setter
    def aa(self, aa):
        self._store.aa = bool(aa)

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
        """The size (diameter) of the points, in logical pixels (or world/model space if ``size_space``is set)."""
        return float(self.uniform_buffer.data["size"])

    @size.setter
    def size(self, size):
        self.uniform_buffer.data["size"] = size
        self.uniform_buffer.update_range(0, 1)

    @property
    def size_space(self):
        """The coordinate space in which the size is expressed.

        Possible values are:
        * "screen": logical screen pixels. The Default.
        * "world": the world / scene coordinate frame.
        * "model": the line's local coordinate frame (same as the line's positions).
        """
        return self._store.size_space

    @size_space.setter
    def size_space(self, value):
        if value is None:
            value = "screen"
        if not isinstance(value, str):
            raise TypeError("PointsMaterial.size_space must be str")
        value = value.lower()
        if value not in ["screen", "world", "model"]:
            raise ValueError(f"Invalid value for PointsMaterial.size_space: {value}")
        self._store.size_space = value

    @property
    def size_mode(self):
        """The way that size is applied to the mesh.

        * uniform: use the material's size property for all points.
        * vertex: use the geometry `sizes` buffer, one size per vertex.
        """
        return self._store.size_mode

    @size_mode.setter
    def size_mode(self, value):
        if isinstance(value, SizeMode):
            pass
        elif isinstance(value, str):
            if value.startswith("SizeMode."):
                value = value.split(".")[-1]
            try:
                value = getattr(SizeMode, value.lower())
            except AttributeError:
                raise ValueError(f"Invalid size_mode: '{value}'")
        else:
            raise TypeError(f"Invalid size_mode class: {value.__class__.__name__}")
        self._store.size_mode = value

    @property
    def map(self):
        """The texture map specifying the color for each texture coordinate.
        The dimensionality of the map can be 1D, 2D or 3D, but should match the
        number of columns in the geometry's texcoords.
        """
        return self._store.map

    @map.setter
    def map(self, map):
        assert map is None or isinstance(map, Texture)
        self._store.map = map

    @property
    def map_interpolation(self):
        """The method to interpolate the colormap. Either 'nearest' or 'linear'."""
        return self._store.map_interpolation

    @map_interpolation.setter
    def map_interpolation(self, value):
        assert value in ("nearest", "linear")
        self._store.map_interpolation = value

    # todo: sizeAttenuation


class PointsGaussianBlobMaterial(PointsMaterial):
    """A material to render points as Gaussian blobs.

    Renders Gaussian blobs with a standard deviation that is 1/6th of the
    point-size.
    """


class PointsMarkerMaterial(PointsMaterial):
    """A material to render points as markers.

    Markers come in a variety of shapes, and have an edge with a separate color.
    """

    @property
    def edge_color(self):
        """The color of the edge of the markers."""
        return Color(self.uniform_buffer.data["edge_color"])

    @edge_color.setter
    def edge_color(self, edge_color):
        edge_color = Color(edge_color)
        self.uniform_buffer.data["edge_color"] = edge_color
        self.uniform_buffer.update_range(0, 1)

    @property
    def edge_width(self):
        """The width of the edge of the markers, in logical pixels (or world/model space if ``size_space``is set)."""
        return float(self.uniform_buffer.data["edge_width"])

    @edge_width.setter
    def edge_width(self, edge_width):
        self.uniform_buffer.data["edge_width"] = edge_width
        self.uniform_buffer.update_range(0, 1)


class PointsSpriteMaterial(PointsMaterial):
    """A material to render points as sprite images.

    Renders the provided texture at each point position. The images are square
    and sized just like with a PointMaterial. The texture color is multiplied
    with the point's "normal" color (as calculated depending on ``color_mode``).

    The sprite texture is provided via ``.sprite``.
    """

    def __init__(self, *, sprite=None, **kwargs):
        super().__init__(**kwargs)
        self.sprite = sprite

    @property
    def sprite(self):
        """The texture map specifying the sprite image.

        The dimensionality of the map must be 2D. If None, it just shows a
        uniform color.
        """
        return self._store.sprite

    @sprite.setter
    def sprite(self, sprite):
        assert sprite is None or isinstance(sprite, Texture)
        self._store.sprite = sprite


# Idea: PointsSdfMaterial(PointsMaterial) -> a material where the point shape can be defined via an sdf.
