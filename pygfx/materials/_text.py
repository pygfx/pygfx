from ._base import Material
from ..utils import unpack_bitfield, Color


class TextMaterial(Material):
    """The default material used by Text."""

    uniform_type = dict(
        color="4xf4",
        thickness="f4",
    )

    def __init__(self, color=(1, 1, 1, 1), thickness=1.0, screen_space=True, **kwargs):
        super().__init__(**kwargs)

        self._screen_space = None
        self.screen_space = screen_space
        self.color = color
        self.thickness = thickness

    def _wgpu_get_pick_info(self, pick_value):
        # This should match with the shader
        # todo: map glyph index to characters (can use wobject._wgpu_get_pick_info)
        _ = unpack_bitfield(pick_value, wobject_id=20, index=26, x=9, y=9)
        return {}

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
        self._store.color_is_transparent = color.a < 1

    @property
    def color_is_transparent(self):
        """Whether the color is (semi) transparent (i.e. not fully opaque)."""
        return self._store.color_is_transparent

    @property
    def thickness(self):
        """A value indicating the relative thickness of the glyphs. Could
        be seen as a boldness scale factor. Default 1.
        """
        return self.uniform_buffer.data["thickness"]

    @thickness.setter
    def thickness(self, value):
        self.uniform_buffer.data["thickness"] = float(value)
        self.uniform_buffer.update_range(0, 1)

    # todo: with SDF the weight may be dynamic? @property def weight(self)
