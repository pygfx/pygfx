from .enums import ColorSpace


def _srgb_to_linear(c):
    # The simplified version has a maximum error less than 1%, but that's still
    # two steps in the range 0..255.
    # return c ** 2.2
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def _linear_to_srgb(c):
    # return c ** (1 / 2.2)
    return c * 12.92 if c <= 0.0031308 else c ** (1 / 2.4) * 1.055 - 0.055


class _ColorManagement:
    def __init__(self):
        self._working_color_space = ColorSpace.linear_srgb

    @property
    def working_color_space(self):
        """The working color space used for rendering."""
        return self._working_color_space

    @working_color_space.setter
    def working_color_space(self, value):
        """Set the working color space."""
        if value not in (ColorSpace.linear_srgb, ColorSpace.srgb):
            raise ValueError("working_color_space must be either linear_srgb or srgb")
        self._working_color_space = value

    def convert_to_working_space(self, color, colorspace):
        """Convert a color from a given colorspace to the working color space."""
        if colorspace == ColorSpace.no_colorspace:
            return color

        if colorspace == self._working_color_space:
            return color

        if colorspace == ColorSpace.srgb:
            r = _srgb_to_linear(color.r)
            g = _srgb_to_linear(color.g)
            b = _srgb_to_linear(color.b)
            color._set_from_rgba(r, g, b, color.a)
        elif colorspace == ColorSpace.linear_srgb:
            r = _linear_to_srgb(color.r)
            g = _linear_to_srgb(color.g)
            b = _linear_to_srgb(color.b)
            color._set_from_rgba(r, g, b, color.a)

        return color


ColorManagement = _ColorManagement()
