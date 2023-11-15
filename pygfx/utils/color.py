"""Provides utilities to deal with color."""

# Possible improvements:
#
# Colorspaces
# * Support for HSLuv, a colorspace with uniform brightness https://www.hsluv.org/
#
# CSS:
# * Support "rgb(100%, 50%, 0%)"
# * Support "hsl(...)" and "hsla(...)"
#
# Other:
# * Color.lerp(color, t) linear interpolate towards other color.
# * Color.lerpHSL(color, t) same as lerp but interpolate in HSL / HSLuv space.
# * Color.lighter(factor) and Color.darker(factor)


import ctypes
import colorsys


F4 = ctypes.c_float * 4


def _float_from_css_value(v, i):
    v = v.strip()
    if v.endswith("%"):
        return float(v[:-1]) / 100
    elif i < 3:
        return float(v) / 255
    else:
        return float(v)


class Color:
    """A representation of color (in the sRGB colorspace).

    Internally the color is stored using 4 32-bit floats (rgba). It can be
    instantiated in a variety of ways. E.g. by providing the color components as
    values between 0 and 1:

        * `Color(r, g, b, a)` providing rgba values.
        * `Color(r, g, b)` providing rgb, alpha is 1.
        * `Color(gray, a)` grayscale intensity and alpha.
        * `Color(gray)` grayscale intensity.

    The above variations can also be supplied as a single tuple/list, or
    anything that:

        * `Color((r, g, b))`.

    Named colors:

        * `Color("red")` base color names.
        * `Color("cornflowerblue")` CSS color names.
        * `Color("m")` Matlab color chars.

    Hex colors:

        * `Color("#ff0000")` the common hex format.
        * `Color("#ff0000ff")` the hex format that includes alpha.
        * `Color("#ff0")` the short form hex format.
        * `Color("#ff0f")` the short form hex format that includes alpha.

    CSS color functions:

        * `Color("rgb(255, 0, 0)")`.
        * `Color("rgba(255, 0, 0, 1.0)")`.

    Parameters
    ----------
    args : tuple, int, str
        The color specification. Check the docstring of this function for
        details on available format options.

    """

    # Internally, the color is a ctypes float array
    __slots__ = ["_val"]

    def __init__(self, *args):
        if len(args) == 1:
            color = args[0]
            if isinstance(color, (int, float)):
                self._set_from_tuple(args)
            elif isinstance(color, str):
                self._set_from_str(color)
            else:
                # Assume it's an iterable,
                # may raise TypeError 'object is not iterable'
                self._set_from_tuple(color)
        else:
            self._set_from_tuple(args)

    def __repr__(self):
        # A precision of 4 decimals, i.e. 10001 possible values for each color.
        # We truncate zeros, but make sure the value does not end with a dot.
        f = lambda v: f"{v:0.4f}".rstrip("0").ljust(3, "0")  # noqa: stfu
        return f"Color({f(self.r)}, {f(self.g)}, {f(self.b)}, {f(self.a)})"

    @property
    def __array_interface__(self):
        # Numpy can wrap our memory in an array without copying
        readonly = True
        ptr = ctypes.addressof(self._val)
        x = dict(version=3, shape=(4,), typestr="<f4", data=(ptr, readonly))
        return x

    def __len__(self):
        return 4

    def __getitem__(self, index):
        return self._val[index]

    def __iter__(self):
        return self.rgba.__iter__()

    def __eq__(self, other):
        if not isinstance(other, Color):
            other = Color(other)
        return all(self._val[i] == other._val[i] for i in range(4))

    def __add__(self, other):
        return Color(
            self.r + other.r,
            self.g + other.g,
            self.b + other.b,
            self.a,
        )

    def __mul__(self, factor):
        if not isinstance(factor, (float, int)):
            raise TypeError("Can only multiple a color with a scalar.")
        return Color(
            self.r * factor,
            self.g * factor,
            self.b * factor,
            self.a,
        )

    def __truediv__(self, factor):
        if not isinstance(factor, (float, int)):
            raise TypeError("Can only multiple a color with a scalar.")
        return self.__mul__(1 / factor)

    def _set_from_rgba(self, r, g, b, a):
        a = max(0.0, min(1.0, float(a)))
        self._val = F4(float(r), float(g), float(b), a)

    def _set_from_tuple(self, color):
        color = tuple(float(c) for c in color)
        if len(color) == 4:
            self._set_from_rgba(*color)
        elif len(color) == 3:
            self._set_from_rgba(*color, 1)
        elif len(color) == 2:
            self._set_from_rgba(color[0], color[0], color[0], color[1])
        elif len(color) == 1:
            self._set_from_rgba(color[0], color[0], color[0], 1)
        else:
            raise ValueError(f"Cannot parse color tuple with {len(color)} values")

    def _set_from_str(self, color):
        color = color.lower()
        if color.startswith("#"):
            # A hex number
            if len(color) == 7:  # #rrggbb
                self._set_from_rgba(
                    int(color[1:3], 16) / 255,
                    int(color[3:5], 16) / 255,
                    int(color[5:7], 16) / 255,
                    1,
                )
            elif len(color) == 4:  # #rgb
                self._set_from_rgba(
                    int(color[1], 16) / 15,
                    int(color[2], 16) / 15,
                    int(color[3], 16) / 15,
                    1,
                )
            elif len(color) == 9:  # #rrggbbaa
                self._set_from_rgba(
                    int(color[1:3], 16) / 255,
                    int(color[3:5], 16) / 255,
                    int(color[5:7], 16) / 255,
                    int(color[7:9], 16) / 255,
                )
            elif len(color) == 5:  # #rgba
                self._set_from_rgba(
                    int(color[1], 16) / 15,
                    int(color[2], 16) / 15,
                    int(color[3], 16) / 15,
                    int(color[4], 16) / 15,
                )
            else:
                raise ValueError(
                    f"Expecting 4, 5, 7, or 9 chars in a hex number, got {len(color)}."
                )
        elif color.startswith(("rgb(", "rgba(")):
            # A CSS color 'function'
            parts = color.split("(")[1].split(")")[0].split(",")
            parts = [_float_from_css_value(p, i) for i, p in enumerate(parts)]
            if len(parts) == 3:
                self._set_from_rgba(parts[0], parts[1], parts[2], 1)
            elif len(parts) == 4:
                self._set_from_rgba(parts[0], parts[1], parts[2], parts[3])
            else:
                raise ValueError(
                    f"CSS color {color.split('(')[0]}(..) must have 3 or 4 elements, not {len(parts)} "
                )
        else:
            # Maybe a named color
            try:
                color_int = NAMED_COLORS[color.lower()]
            except KeyError:
                raise ValueError(f"Unknown color: '{color}'") from None
            else:
                self._set_from_str(color_int)

    @property
    def rgba(self):
        """The RGBA tuple (values between 0 and 1)."""
        return self._val[0], self._val[1], self._val[2], self._val[3]

    @property
    def rgb(self):
        """The RGB tuple (values between 0 and 1)."""
        return self._val[0], self._val[1], self._val[2]

    @property
    def r(self):
        """The red value."""
        return self._val[0]

    @property
    def g(self):
        """The green value."""
        return self._val[1]

    @property
    def b(self):
        """The blue value."""
        return self._val[2]

    @property
    def a(self):
        """The alpha (transparency) value, between 0 and 1."""
        return self._val[3]

    @property
    def gray(self):
        """Return the grayscale intensity."""
        # Calculate with the weighted (a.k.a. luminosity) method, same as Matlab
        r, g, b = self.rgb
        return 0.2989 * r + 0.5870 * g + 0.1140 * b

    @property
    def hex(self):
        """The CSS hex string, e.g. "#00ff00". The alpha channel is ignored.
        Values are clipped to 00 an ff.
        """
        c = self.clip()
        r = int(c.r * 255 + 0.5)
        b = int(c.b * 255 + 0.5)
        g = int(c.g * 255 + 0.5)
        i = (r << 16) + (g << 8) + b
        return "#" + hex(i)[2:].rjust(6, "0")

    @property
    def hexa(self):
        """The hex string including alpha, e.g. "#00ff00ff".
        Values are clipped to 00 an ff.
        """
        c = self.clip()
        r = int(c.r * 255 + 0.5)
        b = int(c.b * 255 + 0.5)
        g = int(c.g * 255 + 0.5)
        a = int(c.a * 255 + 0.5)
        i = (r << 24) + (g << 16) + (b << 8) + a
        return "#" + hex(i)[2:].rjust(8, "0")

    @property
    def css(self):
        """The CSS color string, e.g. "rgba(0,255,0,0.5)"."""
        r, g, b, a = self.rgba
        if a == 1:
            return f"rgb({int(255*r+0.5)},{int(255*g+0.5)},{int(255*b+0.5)})"
        else:
            return f"rgba({int(255*r+0.5)},{int(255*g+0.5)},{int(255*b+0.5)},{a:0.3f})"

    def clip(self):
        """Return a new Color with the values clipped between 0 and 1."""
        return Color(max(0.0, min(1.0, x)) for x in self.rgba)

    @classmethod
    def from_physical(cls, r, g, b, a=1):
        """Create a Color object from a color in the physical colorspace.

        With the physical colorspace we mean what is sometimes called
        Linear-sRGB. It has the same gamut as sRGB, but where sRGB is
        linear w.r.t. human perception, Linear-sRGB is linear w.r.t.
        lumen and photon counts. Calculations on colors in pygfx's shaders
        are done in the physical colorspace.
        """
        return Color(_physical2srgb(r), _physical2srgb(g), _physical2srgb(b), a)

    def to_physical(self):
        """Get the color represented in the physical colorspace, as 3 floats."""
        return _srgb2physical(self.r), _srgb2physical(self.g), _srgb2physical(self.b)

    @classmethod
    def from_hsv(cls, hue, saturation, value):
        """Create a Color object from an a color in the HSV (a.k.a. HSB) colorspace.

        HSV stands for hue, saturation, value (aka brightness). The hue
        component indicates the color tone. Values go from red (0) to
        green (0.333) to blue (0.666) and back to red (1). The
        satutation indicates vividness, with 0 meaning gray and 1
        meaning the primary color. The value/brightness indicates goes
        from 0 (black) to 1 (white).
        """
        return Color(colorsys.hsv_to_rgb(hue, saturation, value))

    def to_hsv(self):
        """Get the color represented in the HSV colorspace, as 3 floats."""
        return colorsys.rgb_to_hsv(*self.rgb)

    @classmethod
    def from_hsl(cls, hue, saturation, lightness):
        """Create a Color object from an a color in the HSL colorspace.

        The HSL colorspace is similar to the HSV colorspace, except the
        "value" is replaced with "lightness". This lightness scales the
        color differently, e.g. a lightness of 1 always represents full
        white.
        """
        return Color(colorsys.hls_to_rgb(hue, lightness, saturation))

    def to_hsl(self):
        """Get the color represented in the HSL colorspace, as 3 floats."""
        hue, lightness, saturation = colorsys.rgb_to_hls(*self.rgb)
        return hue, saturation, lightness


def _srgb2physical(c):
    # The simplified version has a maximum error less than 1%, but that's still
    # two steps in the range 0..255.
    # return c ** 2.2
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def _physical2srgb(c):
    # return c ** (1 / 2.2)
    return c * 12.92 if c <= 0.0031308 else c ** (1 / 2.4) * 1.055 - 0.055


NAMED_COLORS = {
    # CSS Level 1
    "black": "#000000",
    "silver": "#C0C0C0",
    "gray": "#808080",
    "white": "#FFFFFF",
    "maroon": "#800000",
    "red": "#FF0000",
    "purple": "#800080",
    "fuchsia": "#FF00FF",
    "green": "#008000",
    "lime": "#00FF00",
    "olive": "#808000",
    "yellow": "#FFFF00",
    "navy": "#000080",
    "blue": "#0000FF",
    "teal": "#008080",
    "aqua": "#00FFFF",
    # CSS Level 2
    "orange": "#FFA500",
    # CSS Color Module Level 3
    "aliceblue": "#F0F8FF",
    "antiquewhite": "#FAEBD7",
    "aquamarine": "#7FFFD4",
    "azure": "#F0FFFF",
    "beige": "#F5F5DC",
    "bisque": "#FFE4C4",
    "blanchedalmond": "#FFEBCD",
    "blueviolet": "#8A2BE2",
    "brown": "#A52A2A",
    "burlywood": "#DEB887",
    "cadetblue": "#5F9EA0",
    "chartreuse": "#7FFF00",
    "chocolate": "#D2691E",
    "coral": "#FF7F50",
    "cornflowerblue": "#6495ED",
    "cornsilk": "#FFF8DC",
    "crimson": "#DC143C",
    "cyan": "#00FFFF",
    "aqua": "#00FFFF",
    "darkblue": "#00008B",
    "darkcyan": "#008B8B",
    "darkgoldenrod": "#B8860B",
    "darkgray": "#A9A9A9",
    "darkgreen": "#006400",
    "darkgrey": "#A9A9A9",
    "darkkhaki": "#BDB76B",
    "darkmagenta": "#8B008B",
    "darkolivegreen": "#556B2F",
    "darkorange": "#FF8C00",
    "darkorchid": "#9932CC",
    "darkred": "#8B0000",
    "darksalmon": "#E9967A",
    "darkseagreen": "#8FBC8F",
    "darkslateblue": "#483D8B",
    "darkslategray": "#2F4F4F",
    "darkslategrey": "#2F4F4F",
    "darkturquoise": "#00CED1",
    "darkviolet": "#9400D3",
    "deeppink": "#FF1493",
    "deepskyblue": "#00BFFF",
    "dimgray": "#696969",
    "dimgrey": "#696969",
    "dodgerblue": "#1E90FF",
    "firebrick": "#B22222",
    "floralwhite": "#FFFAF0",
    "forestgreen": "#228B22",
    "gainsboro": "#DCDCDC",
    "ghostwhite": "#F8F8FF",
    "gold": "#FFD700",
    "goldenrod": "#DAA520",
    "greenyellow": "#ADFF2F",
    "grey": "#808080",
    "honeydew": "#F0FFF0",
    "hotpink": "#FF69B4",
    "indianred": "#CD5C5C",
    "indigo": "#4B0082",
    "ivory": "#FFFFF0",
    "khaki": "#F0E68C",
    "lavender": "#E6E6FA",
    "lavenderblush": "#FFF0F5",
    "lawngreen": "#7CFC00",
    "lemonchiffon": "#FFFACD",
    "lightblue": "#ADD8E6",
    "lightcoral": "#F08080",
    "lightcyan": "#E0FFFF",
    "lightgoldenrodyellow": "#FAFAD2",
    "lightgray": "#D3D3D3",
    "lightgreen": "#90EE90",
    "lightgrey": "#D3D3D3",
    "lightpink": "#FFB6C1",
    "lightsalmon": "#FFA07A",
    "lightseagreen": "#20B2AA",
    "lightskyblue": "#87CEFA",
    "lightslategray": "#778899",
    "lightslategrey": "#778899",
    "lightsteelblue": "#B0C4DE",
    "lightyellow": "#FFFFE0",
    "limegreen": "#32CD32",
    "linen": "#FAF0E6",
    "magenta": "#FF00FF",
    "fuchsia": "#FF00FF",
    "mediumaquamarine": "#66CDAA",
    "mediumblue": "#0000CD",
    "mediumorchid": "#BA55D3",
    "mediumpurple": "#9370DB",
    "mediumseagreen": "#3CB371",
    "mediumslateblue": "#7B68EE",
    "mediumspringgreen": "#00FA9A",
    "mediumturquoise": "#48D1CC",
    "mediumvioletred": "#C71585",
    "midnightblue": "#191970",
    "mintcream": "#F5FFFA",
    "mistyrose": "#FFE4E1",
    "moccasin": "#FFE4B5",
    "navajowhite": "#FFDEAD",
    "oldlace": "#FDF5E6",
    "olivedrab": "#6B8E23",
    "orangered": "#FF4500",
    "orchid": "#DA70D6",
    "palegoldenrod": "#EEE8AA",
    "palegreen": "#98FB98",
    "paleturquoise": "#AFEEEE",
    "palevioletred": "#DB7093",
    "papayawhip": "#FFEFD5",
    "peachpuff": "#FFDAB9",
    "peru": "#CD853F",
    "pink": "#FFC0CB",
    "plum": "#DDA0DD",
    "powderblue": "#B0E0E6",
    "rosybrown": "#BC8F8F",
    "royalblue": "#4169E1",
    "saddlebrown": "#8B4513",
    "salmon": "#FA8072",
    "sandybrown": "#F4A460",
    "seagreen": "#2E8B57",
    "seashell": "#FFF5EE",
    "sienna": "#A0522D",
    "skyblue": "#87CEEB",
    "slateblue": "#6A5ACD",
    "slategray": "#708090",
    "slategrey": "#708090",
    "snow": "#FFFAFA",
    "springgreen": "#00FF7F",
    "steelblue": "#4682B4",
    "tan": "#D2B48C",
    "thistle": "#D8BFD8",
    "tomato": "#FF6347",
    "turquoise": "#40E0D0",
    "violet": "#EE82EE",
    "wheat": "#F5DEB3",
    "whitesmoke": "#F5F5F5",
    "yellowgreen": "#9ACD32",
    # CSS Color Module Level 4
    "rebeccapurple": "#663399",
    # Matlab / Matplotlib
    "b": "#0000FF",
    "g": "#00FF00",
    "r": "#FF0000",
    "c": "#00FFFF",
    "m": "#FF00FF",
    "y": "#FFFF00",
    "k": "#000000",
    "w": "#FFFFFF",
}
