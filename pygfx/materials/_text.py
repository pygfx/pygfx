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
        If True, use anti-aliasing while rendering glyphs. Aliasing gives
        prettier results, but may affect performance for very large texts.
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
        aa=True,
        **kwargs
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
        """Whether or not the glyphs should be anti-aliased. Aliasing
        gives prettier results, but may affect performance for very large
        texts. Default True.
        """
        return self._store.aa

    @aa.setter
    def aa(self, aa):
        self._store.aa = bool(aa)

    @property
    def color(self):
        """The color of the text."""
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
        self._store.outline_color_is_transparent = outline_color.a < 1

    @property
    def outline_color_is_transparent(self):
        """Whether the outline_color is (semi) transparent (i.e. not fully opaque)."""
        return self._store.outline_color_is_transparent

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
