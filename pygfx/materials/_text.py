from ._base import Material
from ..utils import unpack_bitfield, Color


class TextMaterial(Material):
    """The default material used by Text."""

    uniform_type = dict(
        color="4xf4",
        extra_thickness="f4",
        outline_thickness="f4",
        outline_color="4xf4",
    )

    def __init__(
        self,
        color="#fff",
        *,
        outline_color="#000",
        outline_thickness=0,
        extra_thickness=0.0,
        screen_space=True,
        aa=True,
        **kwargs
    ):
        super().__init__(**kwargs)

        self._screen_space = None
        self.screen_space = screen_space
        self.color = color
        self.outline_color = outline_color
        self.outline_thickness = outline_thickness
        self.extra_thickness = extra_thickness
        self.aa = aa

    def _wgpu_get_pick_info(self, pick_value):
        # This should match with the shader
        # todo: map glyph index to characters (can use wobject._wgpu_get_pick_info)
        _ = unpack_bitfield(pick_value, wobject_id=20, index=26, x=9, y=9)
        return {}

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
    def screen_space(self):
        """Whether the text is applied in screen space (in contrast to model space)."""
        # todo: maybe make different materials for these cases instead
        return self._store.screen_space

    @screen_space.setter
    def screen_space(self, value):
        self._store.screen_space = bool(value)

    @property
    def color(self):
        """The color of the text."""
        return Color(self.uniform_buffer.data["color"])

    @color.setter
    def color(self, color):
        color = Color(color)
        self.uniform_buffer.data["color"] = color
        self.uniform_buffer.update_range(0, 1)
        self._check_color_is_transparent()

    def _check_color_is_transparent(self):
        max_a = max(self.color.a, self.outline_color.a)
        self._store.color_is_transparent = max_a < 0

    @property
    def color_is_transparent(self):
        """Whether the color is (semi) transparent (i.e. not fully opaque)."""
        return self._store.color_is_transparent

    @property
    def outline_thickness(self):
        """A value indicating the relative width of the outline. Valid
        values are between 0 and 0.5, as a fraction of the font size.
        Default 0 (no outline).
        """
        return float(self.uniform_buffer.data["outline_thickness"])

    @outline_thickness.setter
    def outline_thickness(self, value):
        self.uniform_buffer.data["outline_thickness"] = max(0, min(0.5, float(value)))
        self.uniform_buffer.update_range(0, 1)

    @property
    def outline_color(self):
        """The color of the outline of the text."""
        return Color(self.uniform_buffer.data["outline_color"])

    @outline_color.setter
    def outline_color(self, color):
        color = Color(color)
        self.uniform_buffer.data["outline_color"] = color
        self.uniform_buffer.update_range(0, 1)
        self._check_color_is_transparent()

    @property
    def extra_thickness(self):
        """A value indicating additional thickness for the glyphs.
        Could be seen as a font-weight / boldness correction. Valid
        values are between -0.25 and 0.5, as a fraction of the font
        size. Default 0.
        """
        return float(self.uniform_buffer.data["extra_thickness"])

    @extra_thickness.setter
    def extra_thickness(self, value):
        self.uniform_buffer.data["extra_thickness"] = max(-0.25, min(0.5, float(value)))
        self.uniform_buffer.update_range(0, 1)
