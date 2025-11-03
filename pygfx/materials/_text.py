from ._base import Material
from ..utils import unpack_bitfield, Color


class TextMaterial(Material):
    """Basic text material.

    Parameters
    ----------
    color : Color
        The color of the text.
    outline_color : Color
        The color of the outline of the text.
    outline_thickness : int
        A value indicating the relative width of the outline. Valid values are
        between 0.0 and 0.5.
    weight_offset : int
        A value representing an offset to the font weight. Font weights are in
        the range 100-900, so this value should be in the same order of
        magnitude. Can be negative to make text thinner. Default zero.
    aa : bool
        Whether the glyphs is anti-aliased in the shader. Default False.
    kwargs : Any
        Additional kwargs will be passed to the :class:`material base class
        <pygfx.Material>`.

    Notes
    -----
    One use-case for weight_offset is to make dark text on a light background 50
    units thicker, to counter the psychological effect that such text *looks*
    thinner than a light text on a dark background.

    """

    uniform_type = dict(
        Material.uniform_type,
        color="4xf4",
        weight_offset="f4",
        outline_thickness="f4",
        outline_color="4xf4",
    )

    def __init__(
        self,
        color="#fff",
        *,
        outline_color="#000",
        outline_thickness=0,
        weight_offset=0,
        aa=False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.color = color
        self.outline_color = outline_color
        self.outline_thickness = outline_thickness
        self.weight_offset = weight_offset
        self.aa = aa

    def _wgpu_get_pick_info(self, pick_value):
        # Note that the glyph index is not necessarily the same as the
        # char index. It would not be worth the effort to let the shader produce
        # the char index, but I think we could make it possible to look
        # up the char index from the glyph index via the geometry.
        values = unpack_bitfield(pick_value, wobject_id=20, index=26, x=9, y=9)
        return {
            "glyph_index": values["index"],
            "point_coord": (values["x"] / 511.0, values["y"] / 511.0),
        }

    @property
    def aa(self):
        """Whether the glyphs edges are anti-aliased.

        Aliasing gives prettier results by producing semi-transparent fragments
        at the edges. Lines thinner than one physical pixel are also diminished
        by making them more transparent.

        However, because semi-transparent fragments are introduced, artifacts
        may occur if certain cases. For the same reason, aa only works for the
        "blended" and "weighted" alpha methods.

        Note that by default, pygfx already uses SSAA and/or PPAA to anti-alias
        the total renderered result. Text-based aa is an *additional* visual
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
    def color(self):
        """The color of the text."""
        return Color(self.uniform_buffer.data["color"])

    @color.setter
    def color(self, color):
        color = Color(color)
        self.uniform_buffer.data["color"] = color
        self.uniform_buffer.update_full()

    @property
    def outline_thickness(self):
        """A value indicating the relative width of the outline. Valid
        values are between 0.0 and 0.5. Default 0 (no outline).
        """
        return float(self.uniform_buffer.data["outline_thickness"])

    @outline_thickness.setter
    def outline_thickness(self, value):
        self.uniform_buffer.data["outline_thickness"] = max(0.0, min(0.5, float(value)))
        self.uniform_buffer.update_full()

    @property
    def outline_color(self):
        """The color of the outline of the text."""
        return Color(self.uniform_buffer.data["outline_color"])

    @outline_color.setter
    def outline_color(self, outline_color):
        outline_color = Color(outline_color)
        self.uniform_buffer.data["outline_color"] = outline_color
        self.uniform_buffer.update_full()

    @property
    def weight_offset(self):
        """A value representing an offset to the font weight. Font weights
        are in the range 100-900, so this value should be in the same order of
        magnitude. Can be negative to make text thinner. Default zero.

        One use-case is to make dark text on a light background 50 units
        thicker, to counter the psychological effect that such text
        *looks* thinner than a light text on a dark background.
        """
        return float(self.uniform_buffer.data["weight_offset"])

    @weight_offset.setter
    def weight_offset(self, value):
        self.uniform_buffer.data["weight_offset"] = float(value)
        self.uniform_buffer.update_full()
