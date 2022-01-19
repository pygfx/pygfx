"""Provides utilities to deal with color."""

import ctypes


F4 = ctypes.c_float * 4


class Color:
    """An object representing a color.

    Internally the color is stored using 4 32-bit floats (rgba).
    It can be instantiated in a variety of ways. E.g. by providing
    the color components as values between 0 and 1:

    * `Color(r, g, b, a)` providing rgba values.
    * `Color(r, g, b)` providing rgb, alpha is 1.
    * `Color(v, a)` value (gray) and alpha.

    The above variations can also be supplied as a single tuple/list:

    * `Color((r, g, b))`.

    Named colors:

    * `Color("red")` base color names.
    * `Color("cornflowerblue")` CSS color names.
    * `Color("m")` Matlab color chars.

    Hex colors:

    * `Color("#ff0000")` the commmon hex format.
    * `Color("#ff0000ff")` the hex format that includes alpha.
    * `Color("#ff0)` the short form hex format.
    * `Color("#ff0f)` the short form hex format that includes alpha.
    * `Color(0xff0000)` the int form hex color (rgb only).

    CSS color functions:

    * `Color("rgb(255, 0, 0")`.
    * `Color("rgba(255, 0, 0, 1.0")`.

    """

    # Internally, the color is a ctypes float array
    __slots__ = ["_val"]

    def __init__(self, *args):

        if len(args) == 1:
            color = args[0]
            if isinstance(color, Color):
                self._val = color._val
            elif isinstance(color, int):
                self._save_from_int(color)
            elif isinstance(color, str):
                self._save_from_str(color)
            elif isinstance(color, (tuple, list)):
                self._save_from_tuple(color)
            else:
                raise ValueError("Cannot make color from a {type(color).__name__}")
        else:
            self._save_from_tuple(args)

    def __repr__(self):
        return "<Color {:0.2f} {:0.2f} {:0.2f} {:0.2f}>".format(*self.rgba)

    def __len__(self):
        return 4

    def __getitem__(self, index):
        return self._val[index]

    def __iter__(self):
        return self.rgba.__iter__()

    @property
    def __array_interface__(self):
        # Numpy can wrap our memory in an array without copying
        readonly = True
        ptr = ctypes.addressof(self._val)
        x = dict(version=3, shape=(4,), typestr="<f4", data=(ptr, readonly))
        return x

    def _save_from_rgba(self, r, g, b, a):
        self._val = F4(
            max(0.0, min(1.0, float(r))),
            max(0.0, min(1.0, float(g))),
            max(0.0, min(1.0, float(b))),
            max(0.0, min(1.0, float(a))),
        )

    def _save_from_int(self, color):
        v = color
        b = v % 256
        v = v >> 8
        g = v % 256
        v = v >> 8
        r = v % 256
        self._save_from_rgba(r, g, b, 1)

    def _save_from_tuple(self, color):
        color = tuple(float(c) for c in color)
        if len(color) == 4:
            self._save_from_rgba(*color)
        elif len(color) == 3:
            self._save_from_rgba(*color, 1)
        elif len(color) == 2:
            self._save_from_rgba(color[0], color[0], color[0], color[1])
        elif len(color) == 1:
            self._save_from_rgba(color[0], color[0], color[0], 1)
        else:
            raise ValueError(f"Cannot parse color tuple with {len(color)} values")

    def _save_from_str(self, color):
        color = color.lower()
        if color.startswith("0x"):
            # In case someone accidentally puts quotes around the int
            self._save_from_int(int(color, 0))
        elif color.startswith("#"):
            # A hex number
            if len(color) == 7:  # #rrggbb
                self._save_from_rgba(
                    int(color[1:3], 16) / 255,
                    int(color[3:5], 16) / 255,
                    int(color[5:7], 16) / 255,
                    1,
                )
            elif len(color) == 4:  # #rgb
                self._save_from_rgba(
                    int(color[1], 16) / 15,
                    int(color[2], 16) / 15,
                    int(color[3], 16) / 15,
                    1,
                )
            elif len(color) == 9:  # #rrggbbaa
                self._save_from_rgba(
                    int(color[1:3], 16) / 255,
                    int(color[3:5], 16) / 255,
                    int(color[5:7], 16) / 255,
                    int(color[7:9], 16) / 255,
                )
            elif len(color) == 5:  # #rgba
                self._save_from_rgba(
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
            parts = [float(p) for p in parts]
            if len(parts) == 3:
                self._save_from_rgba(parts[0] / 255, parts[1] / 255, parts[2] / 255, 1)
            elif len(parts) == 4:
                self._save_from_rgba(
                    parts[0] / 255, parts[1] / 255, parts[2] / 255, parts[3]
                )
            else:
                raise ValueError(
                    f"CSS color {color.split('(')[0]}(..) must have 3 or 4 elements, not {len(parts)} "
                )
        else:
            # Maybe a CSS named color
            try:
                color_int = NAMED_COLORS[color.lower()]
            except KeyError:
                raise ValueError(f"Unknown color: '{color}'") from None
            else:
                self._save_from_int(color_int)

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
        """The alpha (transparency) value."""
        return self._val[3]

    @property
    def ihex(self):
        """Return as int in which the rgb values are packed."""
        r, g, b = self.rgb
        return (
            (int(r * 255 + 0.5) << 16)
            + (int(g * 255 + 0.5) << 8)
            + (int(b * 255 + 0.5) << 0)
        )

    @property
    def hex(self):
        """The CSS hex string, e.g. "#00ff00". The alpha channel is ignored."""
        r = int(self.r * 255 + 0.5)
        b = int(self.b * 255 + 0.5)
        g = int(self.g * 255 + 0.5)
        i = (r << 16) + (g << 8) + b
        return "#" + hex(i)[2:].rjust(6, "0")

    @property
    def hexa(self):
        """The hex string including alpha, e.g. "#00ff00ff"."""
        r = int(self.r * 255 + 0.5)
        b = int(self.b * 255 + 0.5)
        g = int(self.g * 255 + 0.5)
        a = int(self.a * 255 + 0.5)
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

    # todo: __add__, lighter(), darker(), etc.


NAMED_COLORS = {
    # CSS Level 1
    "black": 0x000000,
    "silver": 0xC0C0C0,
    "gray": 0x808080,
    "white": 0xFFFFFF,
    "maroon": 0x800000,
    "red": 0xFF0000,
    "purple": 0x800080,
    "fuchsia": 0xFF00FF,
    "green": 0x008000,
    "lime": 0x00FF00,
    "olive": 0x808000,
    "yellow": 0xFFFF00,
    "navy": 0x000080,
    "blue": 0x0000FF,
    "teal": 0x008080,
    "aqua": 0x00FFFF,
    # CSS Level 2
    "orange": 0xFFA500,
    # CSS Color Module Level 3
    "aliceblue": 0xF0F8FF,
    "antiquewhite": 0xFAEBD7,
    "aquamarine": 0x7FFFD4,
    "azure": 0xF0FFFF,
    "beige": 0xF5F5DC,
    "bisque": 0xFFE4C4,
    "blanchedalmond": 0xFFEBCD,
    "blueviolet": 0x8A2BE2,
    "brown": 0xA52A2A,
    "burlywood": 0xDEB887,
    "cadetblue": 0x5F9EA0,
    "chartreuse": 0x7FFF00,
    "chocolate": 0xD2691E,
    "coral": 0xFF7F50,
    "cornflowerblue": 0x6495ED,
    "cornsilk": 0xFFF8DC,
    "crimson": 0xDC143C,
    "cyan": 0x00FFFF,
    "aqua": 0x00FFFF,
    "darkblue": 0x00008B,
    "darkcyan": 0x008B8B,
    "darkgoldenrod": 0xB8860B,
    "darkgray": 0xA9A9A9,
    "darkgreen": 0x006400,
    "darkgrey": 0xA9A9A9,
    "darkkhaki": 0xBDB76B,
    "darkmagenta": 0x8B008B,
    "darkolivegreen": 0x556B2F,
    "darkorange": 0xFF8C00,
    "darkorchid": 0x9932CC,
    "darkred": 0x8B0000,
    "darksalmon": 0xE9967A,
    "darkseagreen": 0x8FBC8F,
    "darkslateblue": 0x483D8B,
    "darkslategray": 0x2F4F4F,
    "darkslategrey": 0x2F4F4F,
    "darkturquoise": 0x00CED1,
    "darkviolet": 0x9400D3,
    "deeppink": 0xFF1493,
    "deepskyblue": 0x00BFFF,
    "dimgray": 0x696969,
    "dimgrey": 0x696969,
    "dodgerblue": 0x1E90FF,
    "firebrick": 0xB22222,
    "floralwhite": 0xFFFAF0,
    "forestgreen": 0x228B22,
    "gainsboro": 0xDCDCDC,
    "ghostwhite": 0xF8F8FF,
    "gold": 0xFFD700,
    "goldenrod": 0xDAA520,
    "greenyellow": 0xADFF2F,
    "grey": 0x808080,
    "honeydew": 0xF0FFF0,
    "hotpink": 0xFF69B4,
    "indianred": 0xCD5C5C,
    "indigo": 0x4B0082,
    "ivory": 0xFFFFF0,
    "khaki": 0xF0E68C,
    "lavender": 0xE6E6FA,
    "lavenderblush": 0xFFF0F5,
    "lawngreen": 0x7CFC00,
    "lemonchiffon": 0xFFFACD,
    "lightblue": 0xADD8E6,
    "lightcoral": 0xF08080,
    "lightcyan": 0xE0FFFF,
    "lightgoldenrodyellow": 0xFAFAD2,
    "lightgray": 0xD3D3D3,
    "lightgreen": 0x90EE90,
    "lightgrey": 0xD3D3D3,
    "lightpink": 0xFFB6C1,
    "lightsalmon": 0xFFA07A,
    "lightseagreen": 0x20B2AA,
    "lightskyblue": 0x87CEFA,
    "lightslategray": 0x778899,
    "lightslategrey": 0x778899,
    "lightsteelblue": 0xB0C4DE,
    "lightyellow": 0xFFFFE0,
    "limegreen": 0x32CD32,
    "linen": 0xFAF0E6,
    "magenta": 0xFF00FF,
    "fuchsia": 0xFF00FF,
    "mediumaquamarine": 0x66CDAA,
    "mediumblue": 0x0000CD,
    "mediumorchid": 0xBA55D3,
    "mediumpurple": 0x9370DB,
    "mediumseagreen": 0x3CB371,
    "mediumslateblue": 0x7B68EE,
    "mediumspringgreen": 0x00FA9A,
    "mediumturquoise": 0x48D1CC,
    "mediumvioletred": 0xC71585,
    "midnightblue": 0x191970,
    "mintcream": 0xF5FFFA,
    "mistyrose": 0xFFE4E1,
    "moccasin": 0xFFE4B5,
    "navajowhite": 0xFFDEAD,
    "oldlace": 0xFDF5E6,
    "olivedrab": 0x6B8E23,
    "orangered": 0xFF4500,
    "orchid": 0xDA70D6,
    "palegoldenrod": 0xEEE8AA,
    "palegreen": 0x98FB98,
    "paleturquoise": 0xAFEEEE,
    "palevioletred": 0xDB7093,
    "papayawhip": 0xFFEFD5,
    "peachpuff": 0xFFDAB9,
    "peru": 0xCD853F,
    "pink": 0xFFC0CB,
    "plum": 0xDDA0DD,
    "powderblue": 0xB0E0E6,
    "rosybrown": 0xBC8F8F,
    "royalblue": 0x4169E1,
    "saddlebrown": 0x8B4513,
    "salmon": 0xFA8072,
    "sandybrown": 0xF4A460,
    "seagreen": 0x2E8B57,
    "seashell": 0xFFF5EE,
    "sienna": 0xA0522D,
    "skyblue": 0x87CEEB,
    "slateblue": 0x6A5ACD,
    "slategray": 0x708090,
    "slategrey": 0x708090,
    "snow": 0xFFFAFA,
    "springgreen": 0x00FF7F,
    "steelblue": 0x4682B4,
    "tan": 0xD2B48C,
    "thistle": 0xD8BFD8,
    "tomato": 0xFF6347,
    "turquoise": 0x40E0D0,
    "violet": 0xEE82EE,
    "wheat": 0xF5DEB3,
    "whitesmoke": 0xF5F5F5,
    "yellowgreen": 0x9ACD32,
    # CSS Color Module Level 4
    "rebeccapurple": 0x663399,
    # Matlab / Matplotlib
    "b": 0x0000FF,
    "g": 0x00FF00,
    "r": 0xFF0000,
    "c": 0x00FFFF,
    "m": 0xFF00FF,
    "y": 0xFFFF00,
    "k": 0x000000,
    "w": 0xFFFFFF,
}
