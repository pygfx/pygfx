"""Provides utilities to deal with color."""

import ctypes
import colorsys
import hsluv

F4 = ctypes.c_float * 4


def _float_from_css_value(v, i, is_hue=False):
    v = v.strip()
    if not is_hue:
        if v.endswith("%"):
            return float(v[:-1]) / 100
        elif i < 3:
            return float(v) / 255
        else:
            return float(v)
    else:
        # Hue is a special case, it can be in degrees or raw number between 0 and 1
        if i == 0:
            return float(v[:-3]) / 360 if v.endswith("deg") else float(v)
        else:
            return float(v[:-1]) / 100 if v.endswith("%") else float(v)


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

        * `Color("rgb(255, 0, 0)")` or `Color("rgb(100%, 0%, 0%)")`.
        * `Color("rgba(255, 0, 0, 1.0)")` or `Color("rgba(100%, 0%, 0%, 100%)")`.
        * `Color("hsv(0.333, 1, 0.5)")` or `Color("hsv(120deg, 100%, 50%)")`.
        * `Color("hsva(0.333, 1, 0.5, 1.0)")` or `Color("hsva(120deg, 100%, 50%, 100%)")`.
        * `Color("hsl(0.333, 1, 0.5)")` or `Color("hsl(120deg, 100%, 50%)")`.
        * `Color("hsla(0.333, 1, 0.5, 1.0)")` or `Color("hsla(120deg, 100%, 50%, 100%)")`.
        * `Color("hsluv(0.333, 0.5, 0.5)")` or `Color("hsluv(120deg, 50%, 50%)")`.
        * `Color("hsluva(0.333, 0.5, 0.5, 1.0)")` or `Color("hsluva(120deg, 50%, 50%, 100%)")`.

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
        f = lambda v: f"{v:0.4f}".rstrip("0").ljust(3, "0")
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
        elif color.startswith(("hsl(", "hsla(", "hsv(", "hsva(", "hsluv(", "hsluva(")):
            parts = color.split("(")[1].split(")")[0].split(",")
            parts = [
                _float_from_css_value(p, i, is_hue=True) for i, p in enumerate(parts)
            ]
            if len(parts) == 3 or len(parts) == 4:
                if color.startswith(("hsl(", "hsla(")):
                    color = Color.from_hsl(*parts)
                elif color.startswith(("hsv(", "hsva(")):
                    color = Color.from_hsv(*parts)
                elif color.startswith(("hsluv(", "hsluva(")):
                    color = Color.from_hsluv(*parts)
                self._set_from_rgba(
                    color._val[0], color._val[1], color._val[2], color._val[3]
                )
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
            return (
                f"rgb({int(255 * r + 0.5)},{int(255 * g + 0.5)},{int(255 * b + 0.5)})"
            )
        else:
            return f"rgba({int(255 * r + 0.5)},{int(255 * g + 0.5)},{int(255 * b + 0.5)},{a:0.3f})"

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
    def from_hsv(cls, hue, saturation, value, alpha=1):
        """Create a Color object from a color in the HSV (a.k.a. HSB) colorspace.

        HSV stands for hue, saturation, value (aka brightness). The hue
        component indicates the color tone. Values go from red (0) to
        green (0.333) to blue (0.666) and back to red (1). The
        satutation indicates vividness, with 0 meaning gray and 1
        meaning the primary color. The value/brightness indicates goes
        from 0 (black) to 1 (white).

        The alpha channel is optional and defaults to 1.
        """
        color = Color(colorsys.hsv_to_rgb(hue, saturation, value))
        color._val[3] = alpha
        return color

    def to_hsv(self):
        """Get the color represented in the HSV colorspace, as 3 floats."""
        return colorsys.rgb_to_hsv(*self.rgb)

    def to_hsva(self):
        """Get the color represented in the HSV colorspace, as 4 floats."""
        h, s, v = colorsys.rgb_to_hsv(*self.rgb)
        return h, s, v, self.a

    @classmethod
    def from_hsl(cls, hue, saturation, lightness, alpha=1):
        """Create a Color object from a color in the HSL colorspace.

        The HSL colorspace is similar to the HSV colorspace, except the
        "value" is replaced with "lightness". This lightness scales the
        color differently, e.g. a lightness of 1 always represents full
        white.

        The alpha channel is optional and defaults to 1.
        """
        color = Color(colorsys.hls_to_rgb(hue, lightness, saturation))
        color._val[3] = alpha
        return color

    def to_hsl(self):
        """Get the color represented in the HSL colorspace, as 3 floats."""
        hue, lightness, saturation = colorsys.rgb_to_hls(*self.rgb)
        return hue, saturation, lightness

    def to_hsla(self):
        """Get the color represented in the HSL colorspace, as 4 floats."""
        hue, lightness, saturation = colorsys.rgb_to_hls(*self.rgb)
        return hue, saturation, lightness, self.a

    @classmethod
    def from_hsluv(cls, hue, saturation, lightness, alpha=1):
        """Create a Color object from a color in the HSLuv colorspace.

        HSLuv is a human-friendly alternative to HSL. The hue component works
        the same as HSL/HSV, going from red (0) through green (0.333) and blue (0.666)
        back to red (1). The saturation ranges from 0 (grayscale) to 1 (pure color).
        The lightness ranges from 0 (black) to 1 (white). Unlike HSL/HSV, HSLuv
        provides perceptually uniform brightness and saturation.

        The alpha channel is optional and defaults to 1.
        """
        h, s, light = 360.0 * hue, 100.0 * saturation, 100.0 * lightness
        color = Color(hsluv.hsluv_to_rgb((h, s, light)))
        color._val[3] = alpha
        return color

    def to_hsluv(self):
        """Get the color represented in the HSLuv colorspace, as 3 floats."""
        h, s, light = hsluv.rgb_to_hsluv(self.rgb)
        return h / 360.0, s / 100.0, light / 100.0

    def to_hsluva(self):
        """Get the color represented in the HSLuv colorspace, as 4 floats."""
        h, s, light = hsluv.rgb_to_hsluv(self.rgb)
        return h / 360.0, s / 100.0, light / 100.0, self.a

    def lerp(self, target, t):
        """Linear interpolate from source color towards target color with factor t in RGBA space.

        Parameters
        ----------
        target : Color
            The target color to interpolate towards
        t : float
            Interpolation factor between 0 and 1

        Returns
        -------
        Color
            The interpolated color
        """
        r = self.r + (target.r - self.r) * t
        g = self.g + (target.g - self.g) * t
        b = self.b + (target.b - self.b) * t
        a = self.a + (target.a - self.a) * t
        return Color(r, g, b, a)

    def lerp_in_hue(self, target, t, colorspace="hsluv"):
        """Linear interpolate from source color towards target color with factor t in specified colorspace.

        Parameters
        ----------
        target : Color
            The target color to interpolate towards
        t : float
            Interpolation factor between 0 and 1
        colorspace : str
            The colorspace to interpolate in. One of "hsl", "hsluv", "hsv". Default is "hsluv"

        Returns
        -------
        Color
            The interpolated color
        """
        if colorspace == "hsl":
            h1, s1, l1 = self.to_hsl()
            h2, s2, l2 = target.to_hsl()
            to_rgba = lambda h, s, light, a: Color.from_hsl(h, s, light, a)
        elif colorspace == "hsluv":
            h1, s1, l1 = self.to_hsluv()
            h2, s2, l2 = target.to_hsluv()
            to_rgba = lambda h, s, light, a: Color.from_hsluv(h, s, light, a)
        elif colorspace == "hsv":
            h1, s1, l1 = self.to_hsv()
            h2, s2, l2 = target.to_hsv()
            to_rgba = lambda h, s, light, a: Color.from_hsv(h, s, light, a)
        else:
            raise ValueError(f"Unknown colorspace {colorspace}")

        # Special case for hue - interpolate along shortest path
        if abs(h2 - h1) > 0.5:
            if h1 > h2:
                h2 += 1.0
            else:
                h1 += 1.0
        h = (h1 + (h2 - h1) * t) % 1.0
        s = s1 + (s2 - s1) * t
        light = l1 + (l2 - l1) * t
        a = self.a + (target.a - self.a) * t

        return to_rgba(h, s, light, a)

    def lighter(self, factor=0.5, colorspace="hsluv"):
        """Make the color lighter by the given factor.

        Parameters
        ----------
        factor : float
            Factor to lighten by, between 0 and 1. Default is 0.5
        colorspace : str
            The colorspace to use, one of "hsl", "hsluv", or "hsv". Default is "hsluv"

        Returns
        -------
        Color
            A lighter version of the color
        """
        if colorspace not in ["hsl", "hsluv", "hsv"]:
            raise ValueError("colorspace must be one of 'hsl', 'hsluv', or 'hsv'")
        if not 0 <= factor <= 1:
            raise ValueError("factor must be between 0 and 1")

        if colorspace == "hsl":
            h, s, light = self.to_hsl()
            light = light + (1 - light) * factor
            return Color.from_hsl(h, s, light, self.a)
        elif colorspace == "hsluv":
            h, s, light = self.to_hsluv()
            light = light + (1 - light) * factor
            return Color.from_hsluv(h, s, light, self.a)
        else:  # hsv
            h, s, v = self.to_hsv()
            v = v + (1 - v) * factor
            return Color.from_hsv(h, s, v, self.a)

    def darker(self, factor=0.5, colorspace="hsluv"):
        """Make the color darker by the given factor.

        Parameters
        ----------
        factor : float
            Factor to darken by, between 0 and 1. Default is 0.5
        colorspace : str
            The colorspace to use, one of "hsl", "hsluv", or "hsv". Default is "hsluv"

        Returns
        -------
        Color
            A darker version of the color
        """
        if colorspace not in ["hsl", "hsluv", "hsv"]:
            raise ValueError("colorspace must be one of 'hsl', 'hsluv', or 'hsv'")
        if not 0 <= factor <= 1:
            raise ValueError("factor must be between 0 and 1")

        if colorspace == "hsl":
            h, s, light = self.to_hsl()
            light = light * (1 - factor)
            return Color.from_hsl(h, s, light, self.a)
        elif colorspace == "hsluv":
            h, s, light = self.to_hsluv()
            light = light * (1 - factor)
            return Color.from_hsluv(h, s, light, self.a)
        else:  # hsv
            h, s, v = self.to_hsv()
            v = v * (1 - factor)
            return Color.from_hsv(h, s, v, self.a)


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
