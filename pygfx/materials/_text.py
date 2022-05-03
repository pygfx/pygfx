from ._base import Material
from ..utils import unpack_bitfield, Color


class TextMaterial(Material):
    """The default material used by Text."""

    uniform_type = dict(
        color="4xf4",
    )

    def __init__(self, color=(1, 1, 1, 1), screen_space=True, **kwargs):
        super().__init__(**kwargs)

        self._screen_space = None
        self.screen_space = screen_space
        self.color = color

    def _wgpu_get_pick_info(self, pick_value):
        # This should match with the shader
        _ = unpack_bitfield(pick_value, wobject_id=20, index=26, x=9, y=9)
        return {}

    @property
    def screen_space(self):
        """Whether the text is applied in screen space (in contrast to model space)."""
        return self._screen_space

    @screen_space.setter
    def screen_space(self, value):
        value = bool(value)
        if value != self._screen_space:
            self._bump_rev()
        self._screen_space = value

    @property
    def color(self):
        """The color of the text."""
        return Color(self.uniform_buffer.data["color"])

    @color.setter
    def color(self, color):
        color = Color(color)
        # todo: need this?
        if (color[3] >= 1) != (self.uniform_buffer.data["color"][3] >= 1):
            self._bump_rev()  # rebuild pipeline if this becomes opaque/transparent
        self.uniform_buffer.data["color"] = color
        self.uniform_buffer.update_range(0, 1)

    # todo: with SDF the weight may be dynamic? @property def weight(self)
