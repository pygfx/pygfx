from ._base import Material
from ..resources import Texture
from ..utils import unpack_bitfield, Color
from ..utils.enums import EdgeMode, ColorMode, SizeMode, CoordSpace, MarkerShape


class PointsMaterial(Material):
    """Point default material.

    Renders disks of the given size and color.

    Parameters
    ----------
    size : float
        The size (diameter) of the points in logical pixels. Default 4.
    size_space : str | CoordSpace
        The coordinate space in which the size is expressed ('screen', 'world', 'model'). Default 'screen'.
    size_mode : str | SizeMode
        The mode by which the points are sized. Default 'uniform'.
    color : str | tuple | Color
        The uniform color of the points (used depending on the ``color_mode``).
    color_mode : str | ColorMode
        The mode by which the points are coloured. Default 'auto'.
    edge_mode : str | EdgeMode
        The mode by which the points are edged. Default 'centered'.
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
        edge_mode="centered",
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
        self.edge_mode = edge_mode
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
        self.uniform_buffer.update_full()
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

        See :obj:`pygfx.utils.enums.ColorMode`:
        """
        return self._store.color_mode

    @color_mode.setter
    def color_mode(self, value):
        value = value or "auto"
        if value not in ColorMode:
            raise ValueError(
                f"PointsMaterial.color_mode must be a string in {ColorMode}, not {repr(value)}"
            )
        if value in ["face", "face_map"]:
            raise ValueError(
                f"PointsMaterial.color_mode does not support {repr(value)}"
            )
        self._store.color_mode = value

    @property
    def edge_mode(self):
        """The way that edges are applied to the mesh.

        See :obj:`pygfx.utils.enums.EdgeMode`:
        """
        return self._store.edge_mode

    @edge_mode.setter
    def edge_mode(self, value):
        value = value or "centered"
        if value not in EdgeMode:
            raise ValueError(
                f"PointsMaterial.edge_mode must be a string in {EdgeMode}, not {repr(value)}"
            )
        self._store.edge_mode = value

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
        """The size (diameter) of the points, in logical pixels (or world/model space if ``size_space`` is set)."""
        return float(self.uniform_buffer.data["size"])

    @size.setter
    def size(self, size):
        self.uniform_buffer.data["size"] = size
        self.uniform_buffer.update_full()

    @property
    def size_space(self):
        """The coordinate space in which the size is expressed.

        See :obj:`pygfx.utils.enums.CoordSpace`:
        """
        return self._store.size_space

    @size_space.setter
    def size_space(self, value):
        value = value or "screen"
        if value not in CoordSpace:
            raise ValueError(
                f"PointsMaterial.size_space must be a string in {CoordSpace}, not {repr(value)}"
            )
        self._store.size_space = value

    @property
    def size_mode(self):
        """The way that size is applied to the mesh.

        See :obj:`pygfx.utils.enums.SizeMode`:
        """
        return self._store.size_mode

    @size_mode.setter
    def size_mode(self, value):
        value = value or "uniform"
        if value not in SizeMode:
            raise ValueError(
                f"PointsMaterial.size_mode must be a string in {SizeMode}, not {repr(value)}"
            )
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

    Parameters
    ----------
    marker : str | MarkerShape
        The shape of the marker. Default 'circle'.
    edge_color : str | tuple | Color
        The color of line marking the edge of the markers. Default 'black'.
    edge_width : float
        The width of the edge of the markers. Default 1.
    kwargs : Any
        Additional kwargs will be passed to the :class:`PointsMaterial<pygfx.PointsMaterial>`.

    """

    uniform_type = dict(
        PointsMaterial.uniform_type,
        edge_color="4xf4",
        edge_width="f4",
    )

    def __init__(
        self,
        *,
        marker="circle",
        edge_width=1,
        edge_color="black",
        custom_sdf=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.marker = marker
        self.edge_width = edge_width
        self.edge_color = edge_color
        self.custom_sdf = custom_sdf

    @property
    def edge_color(self):
        """The color of the edge of the markers."""
        return Color(self.uniform_buffer.data["edge_color"])

    @edge_color.setter
    def edge_color(self, edge_color):
        edge_color = Color(edge_color)
        self.uniform_buffer.data["edge_color"] = edge_color
        self.uniform_buffer.update_full()
        self._store.edge_color_is_transparent = edge_color.a < 1

    @property
    def edge_color_is_transparent(self):
        """Whether the edge_color is (semi) transparent (i.e. not fully opaque)."""
        return self._store.edge_color_is_transparent

    @property
    def edge_width(self):
        """The width of the edge of the markers, in logical pixels (or world/model space if ``size_space`` is set)."""
        return float(self.uniform_buffer.data["edge_width"])

    @edge_width.setter
    def edge_width(self, edge_width):
        self.uniform_buffer.data["edge_width"] = float(edge_width)
        self.uniform_buffer.update_full()

    @property
    def marker(self):
        """The type/shape of the markers.

        Supported values:

        * A string from :obj:`pygfx.utils.enums.MarkerShape`.
        * Matplotlib compatible characters: "osD+x^v<>".
        * Unicode symbols: "●○■♦♥♠♣✳▲▼◀▶".
        * Emojis: "❤️♠️♣️♦️💎💍✳️📍".
        * A string containing the value "custom". In this case, the WGSL
          code defined by ``custom_sdf`` will be used.
        """
        # TODO: is marker a good name?
        # Note: MPL calls this 'marker', Plotly calls this 'symbol'
        return self._store.marker

    @marker.setter
    def marker(self, name):
        # Define possible values, see:
        # https://matplotlib.org/stable/api/markers_api.html
        # https://plotly.com/python/marker-style/#custom-marker-symbols

        alt_names = {
            # MPL
            "o": "circle",
            "s": "square",
            "D": "diamond",
            "+": "plus",
            "x": "cross",
            "^": "triangle_up",
            "<": "triangle_left",
            ">": "triangle_right",
            "v": "triangle_down",
            # Unicode
            "●": "circle",
            "○": "ring",
            "■": "square",
            "♦": "diamond",
            "♥": "heart",
            "♠": "spade",
            "♣": "club",
            "✳": "asterix",
            "▲": "triangle_up",
            "▼": "triangle_down",
            "◀": "triangle_left",
            "▶": "triangle_right",
            # Emojis (these may look like their plaintext variants in your editor)
            "❤️": "heart",
            "♠️": "spade",
            "♣️": "club",
            "♦️": "diamond",
            "💎": "diamond",
            "💍": "ring",
            "✳️": "asterix",
            "📍": "pin",
        }

        name = name or "circle"
        resolved_name = alt_names.get(name, name).lower()
        if resolved_name not in MarkerShape:
            raise ValueError(
                f"PointsMarkerMaterial.marker must be a string in {SizeMode}, or a supported characted, not {repr(name)}"
            )
        self._store.marker = resolved_name

    @property
    def custom_sdf(self):
        """The SDF code for the marker shape when the marker is set to custom.

        Negative values are inside the shape, positive values are outside the
        shape.

        The SDF's takes in two parameters `coords: vec2<f32>` and `size: f32`.
        The first is a WGSL coordinate and `size` is the overall size of
        the texture. The returned value should be the signed distance from
        any edge of the shape. Distances (positive and negative) that are
        less than half the `edge_width` in absolute terms will be colored
        with the `edge_color`. Other negative distances will be colored by
        `color`.
        """
        return self._store.custom_sdf

    @custom_sdf.setter
    def custom_sdf(self, code):
        self._store.custom_sdf = code


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
