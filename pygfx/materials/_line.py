from ._base import Material
from ..resources import Texture, TextureMap
from ..utils import unpack_bitfield, Color, assert_type
from ..utils.enums import ColorMode, CoordSpace, DashFit, DashScaling


class LineMaterial(Material):
    """Basic line material.

    Parameters
    ----------
    thickness : float
        The line thickness expressed in logical pixels. Default 2.0.
    thickness_space : str | CoordSpace
        The coordinate space in which the thickness is expressed ('screen', 'world', 'model'). Default 'screen'.
    color : Color
        The uniform color of the line (used depending on the ``color_mode``).
    color_mode : str | ColorMode
        The mode by which the line is coloured. Default 'auto'.
    map : TextureMap | Texture
        The texture map specifying the color for each texture coordinate. Optional.
    maprange : tuple
        The range of the ``geometry.texcoords`` that is projected onto the (color) map. Default (0, 1).
    dash_pattern : tuple
        The pattern of the dash, e.g. `[2, 3]`. See `dash_pattern` docs for details. Defaults to an empty tuple, i.e. no dashing.
    dash_offset : float
        The offset into the dash phase. Default 0.0.
    dash_scaling : str | DashScaling
        How the dash pattern behaves when the view scale changes ('continuous', 'quantized'). Default 'continuous'.
    dash_scale_step : float
        For 'quantized' scaling, the factor between successive dash sizes. 1 reproduces continuous scaling, 2 makes each dash split in two. Default 2.
    dash_max_scale : float | None
        For 'quantized' scaling, the largest the dashes may grow, relative to the requested size, before they split. Default None, i.e. centred on the requested size.
    dash_scale_hysteresis : float
        For 'quantized' scaling, how far past a step boundary the view must go before the dash size follows, as a fraction of one step. Default 0.1.
    dash_fit : str | DashFit
        Whether and how the pattern is fitted so that a whole number of periods spans each line piece ('exact', 'stretch', 'spread'). Default 'exact'.
    loop : bool
        Whether the line's end should be connected. Default False.
    aa : bool
        Whether the line is anti-aliased in the shader. Default False.
    kwargs : Any
        Additional kwargs will be passed to the :class:`material base class <pygfx.Material>`.
    """

    uniform_type = dict(
        Material.uniform_type,
        color="4xf4",
        maprange="2xf4",
        thickness="f4",
        dash_offset="f4",
    )

    def __init__(
        self,
        thickness=2.0,
        thickness_space="screen",
        *,
        color=(1, 1, 1, 1),
        color_mode="auto",
        map=None,
        maprange=None,
        dash_pattern=(),
        dash_offset=0,
        dash_scaling="continuous",
        dash_scale_step=2.0,
        dash_max_scale=None,
        dash_scale_hysteresis=0.1,
        dash_fit="exact",
        loop=False,
        aa=False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.thickness = thickness
        self.thickness_space = thickness_space
        self.color = color
        self.color_mode = color_mode
        self.map = map
        self.maprange = maprange
        self.dash_pattern = dash_pattern
        self.dash_offset = dash_offset
        self.dash_scaling = dash_scaling
        self.dash_scale_step = dash_scale_step
        self.dash_max_scale = dash_max_scale
        self.dash_scale_hysteresis = dash_scale_hysteresis
        self.dash_fit = dash_fit
        self.loop = loop
        self.aa = aa

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
        self.uniform_buffer.update_full()

    @property
    def aa(self):
        """Whether the line's edges are anti-aliased.

        Aliasing gives prettier results by producing semi-transparent fragments
        at the edges. Lines thinner than one physical pixel are also diminished
        by making them more transparent.

        However, because semi-transparent fragments are introduced, artifacts
        may occur if certain cases. For the same reason, aa only works for the
        "blended" and "weighted" alpha methods.

        Note that by default, pygfx already uses SSAA and/or PPAA to anti-alias
        the total renderered result. Line-based aa is an *additional* visual
        improvement.
        """
        return self._store.aa

    @aa.setter
    def aa(self, aa):
        self._store.aa = bool(aa)

    @property
    def _gfx_effective_aa(self):
        aa_able_methods = ("blended", "weighted")
        return self._store.aa and self.alpha_method in aa_able_methods

    @property
    def color_mode(self):
        """The way that color is applied to the line.

        See :obj:`pygfx.utils.enums.ColorMode`:
        """
        return self._store.color_mode

    @color_mode.setter
    def color_mode(self, value):
        value = value or "auto"
        if value not in ColorMode:
            raise ValueError(
                f"LineMaterial.color_mode must be a string in {ColorMode}, not {value!r}"
            )
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
        """The line thickness.

        The interpretation depends on `thickness_space`. By default it is in logical
        pixels, but it can also be in world or model coordinates.
        """
        return float(self.uniform_buffer.data["thickness"])

    @thickness.setter
    def thickness(self, thickness):
        self.uniform_buffer.data["thickness"] = max(0.0, float(thickness))
        self.uniform_buffer.update_full()

    @property
    def thickness_space(self):
        """The coordinate space in which the thickness (and dash_pattern) are expressed.

        See :obj:`pygfx.utils.enums.CoordSpace`:
        """
        return self._store.thickness_space

    @thickness_space.setter
    def thickness_space(self, value):
        value = value or "screen"
        if value not in CoordSpace:
            raise ValueError(
                f"LineMaterial.thickness_space must be a string in {CoordSpace}, not {value!r}"
            )
        self._store.thickness_space = value

    @property
    def map(self):
        """The texture map specifying the color for each texture coordinate.

        Can be None. The dimensionality of the map can be 1D, 2D or 3D, but
        should match the number of columns in the geometry's texcoords.
        """
        return self._store.map

    @map.setter
    def map(self, map):
        assert_type("map", map, None, Texture, TextureMap)
        if isinstance(map, Texture):
            map = TextureMap(map)
        self._store.map = map

    @property
    def maprange(self):
        """The range of the ``geometry.texcoords`` that is projected onto the (color) map.

        By default this value is (0.0, 1.0), but if the ``texcoords`` represents some
        domain-specific value, e.g. temperature, then ``maprange`` can be set to e.g. (0, 100).
        """
        v1, v2 = self.uniform_buffer.data["maprange"]
        return float(v1), float(v2)

    @maprange.setter
    def maprange(self, maprange):
        # Check and store given value
        if maprange is None:
            maprange = 0, 1
        maprange = float(maprange[0]), float(maprange[1])
        # Update uniform data
        self.uniform_buffer.data["maprange"] = maprange
        self.uniform_buffer.update_full()

    @property
    def dash_pattern(self):
        """The dash pattern.

        A sequence of floats describing the length of strokes and gaps. The
        length of the sequence must be an even number. Setting to None or the
        empty tuple means no dashing.

        For example, (5, 2, 1, 2) describes a a stroke of 5 units, a gap of 2,
        then a short stroke of 1, and another gap of 2. Units are relative to
        the line thickness (and therefore `thickness_space` also applies to  the
        `dash_pattern`).
        """
        return self._store.dash_pattern

    @dash_pattern.setter
    def dash_pattern(self, value):
        if value is None:
            value = ()
        if not isinstance(value, (tuple, list)):
            raise TypeError(
                "Line dash_pattern must be a sequence of floats, not '{value}'"
            )
        if len(value) % 2:
            raise ValueError("Line dash_pattern must have an even number of elements.")
        self._store.dash_pattern = tuple(max(0.0, float(v)) for v in value)

    @property
    def dash_offset(self):
        """The offset into the dash cycle to start drawing at, i.e. the phase."""
        return float(self.uniform_buffer.data["dash_offset"])

    @dash_offset.setter
    def dash_offset(self, value):
        self.uniform_buffer.data["dash_offset"] = float(value)
        self.uniform_buffer.update_full()

    @property
    def dash_scaling(self):
        """How the dash pattern responds to a change in view scale.

        See :obj:`pygfx.utils.enums.DashScaling`.

        With the default 'continuous', the dash pattern has exactly the size
        that `dash_pattern` and `thickness_space` ask for. The catch, when
        `thickness_space` is 'screen', is that the number of dashes along the
        line then changes continuously as you zoom, so the dashes slide along
        the line. A dash drifts by an amount proportional to its distance from
        the start of its line piece, which makes the far end of a long line
        appear to race while the near end sits still.

        With 'quantized', the pattern is instead anchored to the object, and
        only its period is allowed to follow the view scale, snapped to a power
        of two. Dashes therefore never move: each time the view scale doubles,
        every dash splits into two, and each time it halves, pairs of dashes
        merge. The price is that the on-screen dash size is only approximately
        the requested one, staying within about a factor of 1.5 of it, and
        that it changes in steps rather than smoothly. The step is snapped with
        hysteresis, so a view parked near a boundary does not flicker between
        two levels.

        Note that with 'quantized' the `dash_offset` is expressed in periods
        rather than in units of the pattern, because only a whole number of
        periods keeps the dashes from moving when the period changes.
        """
        return self._store.dash_scaling

    @dash_scaling.setter
    def dash_scaling(self, value):
        value = "continuous" if value is None else str(value)
        if value not in DashScaling:
            raise ValueError(
                f"LineMaterial.dash_scaling must be a string in {DashScaling}, not {value!r}"
            )
        self._store.dash_scaling = value

    @property
    def dash_scale_step(self):
        """The factor between successive dash sizes, when `dash_scaling` is
        'quantized'.

        As the view scale changes, the dash size does not follow it smoothly;
        it is held, and then jumps by this factor. The parameter is a
        continuous dial between the two behaviours:

        * 1.0 -- the size follows the view exactly, i.e. this reproduces
          `dash_scaling='continuous'`. The dashes keep the size asked for, and
          slide along the line as you zoom.
        * 2.0 (the default) -- the size snaps to powers of two. Nothing slides:
          each step splits every dash into two.
        * 3.0 -- as above, but each step splits every dash into three, over a
          three-fold range of sizes.

        Only a whole number gives dashes that truly never move, because the
        dash starts of one level are a subset of the next level's only when the
        levels differ by a whole factor. In between, the dashes still jump
        rather than slide, but they land somewhere new when they do; the
        smaller the step, the smaller and more frequent those jumps, until at
        1.0 they merge into the continuous slide.

        The size varies over a range of exactly this factor, so a larger step
        buys stiller dashes at the cost of a less accurate size.
        """
        return self._store.dash_scale_step

    @dash_scale_step.setter
    def dash_scale_step(self, value):
        value = float(value)
        if value < 1.0:
            raise ValueError(
                f"LineMaterial.dash_scale_step must be at least 1, not {value!r}"
            )
        self._store.dash_scale_step = value

    @property
    def dash_max_scale(self):
        """How large the dashes may grow before they split, when `dash_scaling`
        is 'quantized'.

        Expressed as a ratio to the size that `dash_pattern` asks for. The
        dashes range over exactly `dash_scale_step`, from
        `dash_max_scale / dash_scale_step` up to `dash_max_scale` times the
        requested size; this property only chooses where that range sits.
        It is clamped to lie between 1 and `dash_scale_step`.

        The default, None, centres the range on the requested size, i.e.
        ``sqrt(dash_scale_step)``, so that the dashes are never off by more
        than that factor in either direction. Setting it to
        `dash_scale_step` means the dashes are never finer than requested, and
        setting it to 1 means they are never coarser.

        Note that the hysteresis on the snap (see the `dash_scaling` docs) lets
        the dashes overshoot the range slightly, so that a view parked on a
        boundary does not flicker between two levels.
        """
        return self._store.dash_max_scale

    @dash_max_scale.setter
    def dash_max_scale(self, value):
        if value is not None:
            value = float(value)
            if value < 1.0:
                raise ValueError(
                    f"LineMaterial.dash_max_scale must be at least 1, not {value!r}"
                )
        self._store.dash_max_scale = value

    @property
    def dash_scale_hysteresis(self):
        """How far past a step boundary the view must go before the dash size
        follows, when `dash_scaling` is 'quantized'.

        In fractions of one `dash_scale_step`. The snap is a Schmitt trigger:
        a segment keeps the size it has until the size it would ideally take is
        this much beyond the boundary, which is what stops it flickering between
        two sizes when the view sits near one. Measured at a boundary over 200
        frames, with the default 0.1 against 0 (a plain threshold):

        =============  =================  ===================
        camera jitter  size changes at 0  size changes at 0.1
        =============  =================  ===================
        0.5%           108                0
        5.0%           108                0
        =============  =================  ===================

        A monotonic zoom still changes size exactly once per step, just slightly
        later than the nominal point. The price is that the dashes may stray
        that much further from the size asked for: the range widens by
        ``dash_scale_step ** dash_scale_hysteresis`` at each end, so at the
        default step of 2 the bound goes from a factor of 1.41 to 1.52.

        Set to 0 for a plain threshold, which is worth doing only to see the
        flicker it exists to prevent.
        """
        return self._store.dash_scale_hysteresis

    @dash_scale_hysteresis.setter
    def dash_scale_hysteresis(self, value):
        value = float(value)
        if value < 0:
            raise ValueError(
                f"LineMaterial.dash_scale_hysteresis cannot be negative, got {value!r}"
            )
        self._store.dash_scale_hysteresis = value

    @property
    def dash_fit(self):
        """Whether the dash pattern is stretched to fit the line pieces.

        See :obj:`pygfx.utils.enums.DashFit`.

        With the default 'exact', the dashes have exactly the size that
        `dash_pattern` asks for, and the pattern simply stops wherever the line
        piece happens to end. A closed piece therefore has a seam: the pattern
        arrives back at its starting point mid-period, so the last stroke
        before the join is the wrong length, or lands on top of the first one.

        With 'stretch', each line piece is scaled by the small factor that
        makes a whole number of periods span it, which removes the seam. A
        closed piece joins up as though the pattern had been drawn for it; an
        open one begins and ends with a full stroke.

        Each nan-separated piece is fitted on its own, so pieces of different
        lengths end up with slightly different dash sizes. That is the price:
        the dash size is no longer the one `dash_pattern` asked for, but the
        nearest one that divides the piece. The error is at most half a period
        spread over the whole piece, so it is largest on short pieces -- a
        piece shorter than one period is stretched to exactly one.

        With 'spread', only the gaps give. The strokes keep exactly the size
        `dash_pattern` asks for and the space between them is widened until the
        piece is a whole number of periods long. The pattern is then a *floor*:
        nothing is ever drawn smaller, or closer together, than asked for. Where
        'stretch' would have compressed a piece to reach the nearest whole
        number of periods, 'spread' drops to the next one down and widens the
        gaps instead; when a piece has grown enough to take another stroke it
        gets one, and the gaps snap back to their minimum.

        A piece shorter than a single period cannot be fitted without drawing
        something smaller than asked, so it is left alone and the pattern simply
        runs off the end of it.

        Note that the fit is to the pattern's *period*. With a `dash_offset`
        that is not a whole number of periods, a closed piece still joins up,
        but an open one no longer starts and ends on a stroke boundary.
        """
        return self._store.dash_fit

    @dash_fit.setter
    def dash_fit(self, value):
        value = "exact" if value is None else str(value)
        if value not in DashFit:
            raise ValueError(
                f"LineMaterial.dash_fit must be a string in {DashFit}, not {value!r}"
            )
        self._store.dash_fit = value

    @property
    def loop(self) -> bool:
        """Whether the line's ends should be connected.

        If set to True, the end of the line is connected to its beginning, in
        such a way there is no overlap (which would otherwise be visible for
        semi-transparent lines). When the line consists of multiple pieces
        separated by nan-positions, each line-piece is considered a loop.
        """
        return self._store.loop

    @loop.setter
    def loop(self, loop: bool):
        self._store.loop = bool(loop)


class LineDebugMaterial(LineMaterial):
    """Line debug material.

    A material that renders the triangles that the line is made up off.
    """

    pass


class LineSegmentMaterial(LineMaterial):
    """Line segment material.

    A material that renders line segments between each two subsequent points.
    """


class LineInfiniteSegmentMaterial(LineSegmentMaterial):
    """Infinite line segment material.

    A material that renders infenitely long line segments between each two
    subsequent points. The end-points of each segment are displaced (along the
    vector defined by the two points) such that the points are at the edge of
    the viewport. Other than that, dashing, vertex colors, etc. should work as
    expected (interpolating between the points that are now on the viewport edge).

    Parameters
    ----------
    start_is_infinite : bool
        Whether start of each segment is made infinitely long. Default True.
    end_is_infinite : bool
        Whether end of each segment is made infinitely long. Default True.
    """

    def __init__(self, start_is_infinite=True, end_is_infinite=True, **kwargs):
        super().__init__(**kwargs)
        self.start_is_infinite = start_is_infinite
        self.end_is_infinite = end_is_infinite

    @property
    def start_is_infinite(self):
        """Whether start of each segment is made infinitely long."""
        return self._store.start_is_infinite

    @start_is_infinite.setter
    def start_is_infinite(self, value):
        self._store.start_is_infinite = bool(value)

    @property
    def end_is_infinite(self):
        """Whether end of each segment is made infinitely long."""
        return self._store.end_is_infinite

    @end_is_infinite.setter
    def end_is_infinite(self, value):
        self._store.end_is_infinite = bool(value)


class LineArrowMaterial(LineSegmentMaterial):
    """Arrow (vector) line material.

    A material that renders line segments that look like little arrows.
    """


class LineThinMaterial(LineMaterial):
    """Thin line material.

    A simple line, drawn with line_strip primitives that has a thickness
    of one physical pixel. Thickness, dashing, and aa are ignored.

    While you typically don't want to use this in your application (its
    width is inconsistent and looks *very* thin on HiDPI monitors), it can be
    useful for debugging as it is more performant than other line materials.

    """


class LineThinSegmentMaterial(LineMaterial):
    """Thin line segment material.

    Simple line segments, drawn with line primitives that has a thickness
    of one physical pixel. Thickness, dashing, and aa are ignored.

    While you typically don't want to use this in your application (its
    width is inconsistent and looks *very* thin on HiDPI monitors), it can be
    useful for debugging as it is more performant than other line materials.

    """
