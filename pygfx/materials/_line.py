from ._base import Material, ColorMode
from ..resources import Texture
from ..utils import unpack_bitfield, Color


class LineMaterial(Material):
    """Basic line material.

    Parameters
    ----------
    color : Color
        The uniform color of the line (used depending on the ``color_mode``).
    thickness : float
        The line thickness expressed in logical pixels.
    color_mode : enum or str
        The mode by which the line is coloured. Default 'auto'.
    map : Texture
        The texture map specifying the color for each texture coordinate. Optional.
    map_interpolation: str
        The method to interpolate the color map. Either 'nearest' or 'linear'. Default 'linear'.
    aa : bool
        Whether or not the line is anti-aliased in the shader. This gives smoother
        edges, but may affect performance for very large datasets. Default True.
    kwargs : Any
        Additional kwargs will be passed to the :class:`material base class
        <pygfx.Material>`.

    """

    uniform_type = dict(
        Material.uniform_type,
        color="4xf4",
        thickness="f4",
    )

    def __init__(
        self,
        color=(1, 1, 1, 1),
        thickness=2.0,
        color_mode="auto",
        map=None,
        map_interpolation="linear",
        aa=True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.color = color
        self.aa = aa
        self.map = map
        self.map_interpolation = map_interpolation
        self.thickness = thickness
        self.color_mode = color_mode

    def _wgpu_get_pick_info(self, pick_value):
        # This should match with the shader
        values = unpack_bitfield(pick_value, wobject_id=20, index=26, coord=18)
        return {
            "vertex_index": values["index"],
            "segment_coord": (values["coord"] - 100000) / 100000.0,
        }

    @property
    def color(self):
        """The uniform color of the line."""
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
        """Whether or not the line should be anti-aliased. Aliasing
        gives prettier results, but may affect performance for very large
        datasets. Default True.
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
        * face: use the geometry `colors` buffer, one color per line-piece.
        * vertex_map: use the geometry texcoords buffer to sample (per vertex) in the material's ``map`` texture.
        * faces_map: use the geometry texcoords buffer to sample (per line-piece) in the material's ``map`` texture.
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
    def thickness(self):
        """The line thickness expressed in logical pixels."""
        return float(self.uniform_buffer.data["thickness"])

    @thickness.setter
    def thickness(self, thickness):
        self.uniform_buffer.data["thickness"] = thickness
        self.uniform_buffer.update_range(0, 1)

    @property
    def map(self):
        """The texture map specifying the color for each texture coordinate.
        Can be None. The dimensionality of the map can be 1D, 2D or 3D,
        but should match the number of columns in the geometry's
        texcoords.
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


class LineDashedMaterial(LineMaterial):
    """Line dashed material.

    A meterial that renders dashed lines.

    Parameters
    ----------
    dash_pattern : tuple
        The pattern of the dash, expressed as a series of (2, 4, 6, or 8) floats.
    dash_offset : float
        The offset into the dash cycle to start drawing at. Default 0.0.
    dash_is_screen_space : bool
        Whether the dash measures are expressed in screen space. Default True.
        If False, the dash is expressed in model/world coordinates.
    """

    uniform_type = dict(
        LineMaterial.uniform_type,
        dash_pattern="4x2xf4",
    )

    def __init__(
        self,
        *args,
        dash_pattern=(1, 1),
        dash_offset=0,
        dash_is_screen_space=True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.dash_pattern = dash_pattern
        self.dash_offset = dash_offset
        self.dash_is_screen_space = dash_is_screen_space

    @property
    def dash_pattern(self):
        """The dash pattern.

        A sequence of floats describing the length of strokes and gaps. For example, (5, 2, 1, 2)
        describes a a stroke of 5 unnits, a gap of 2, then short stroke of 1, and another gap of 2.
        Units are relative to the line thickness.
        """
        return tuple(
            float(v) for v in self.uniform_buffer.data["dash_pattern"].flat if v >= 0
        )

    @dash_pattern.setter
    def dash_pattern(self, value):
        if not isinstance(value, (tuple, list)):
            raise TypeError(
                "Line dash_pattern must be a sequence of floats, not '{value}'"
            )
        if len(value) not in (2, 4, 6, 8):
            raise ValueError("Line dash_pattern must have 2, 4, 6, or 8 elements.")
        pattern = [max(0.0, float(v)) for v in value]
        while len(pattern) < 8:
            pattern.append(-1)

        self.uniform_buffer.data["dash_pattern"].flat = pattern
        self.uniform_buffer.update_range(0, 1)

    @property
    def dash_offset(self):
        """The offset into the dash cycle to start drawing at, i.e. the phase."""
        return self._store.dash_offset

    @dash_offset.setter
    def dash_offset(self, value):
        self._store.dash_offset = float(value)

    @property
    def dash_is_screen_space(self):
        """Whether the dash measures are expressed in screen space."""
        return self._store.dash_is_screen_space

    @dash_is_screen_space.setter
    def dash_is_screen_space(self, value):
        self._store.dash_is_screen_space = bool(value)


class LineSegmentMaterial(LineMaterial):
    """Line segment material.

    A material that renders line segments between each two subsequent points.
    """


class LineArrowMaterial(LineSegmentMaterial):
    """Arrow (vector) line material.

    A material that renders line segments that look like little vectors.
    """


class LineThinMaterial(LineMaterial):
    """Thin line material.

    A simple line, drawn with line_strip primitives that has a thickness
    of one physical pixel (the thickness property is ignored).

    While you probably don't want to use this property in your application (its
    width is inconsistent and looks *very* thin on HiDPI monitors), it can be
    useful for debugging as it is more performant than other line materials.

    """


class LineThinSegmentMaterial(LineMaterial):
    """Thin line segment material.

    Simple line segments, drawn with line primitives that has a thickness
    of one physical pixel (the thickness property is ignored).

    While you probably don't want to use this property in your application (its
    width is inconsistent and looks *very* thin on HiDPI monitors), it can be
    useful for debugging as it is more performant than other line materials.

    """
