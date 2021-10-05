import ctypes


F4 = ctypes.c_float * 4


class Color:
    """An object to hold a color value."""

    __slots__ = ["_val"]

    def __init__(self, *args):

        if len(args) == 0:
            self._save_from_rgba(0, 0, 0, 0)
        elif len(args) == 1:
            color = args[0]
            if isinstance(color, Color):
                self._val = color._val
            elif isinstance(color, float):
                self._save_from_tuple((color,))
            elif isinstance(color, int):
                self._save_from_int(color)
            elif isinstance(color, str):
                self._save_from_str(color)
            elif isinstance(color, (tuple, list)):
                self._save_from_tuple(color)
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
        readonly = True
        ptr = ctypes.addressof(self._val)
        x = dict(version=3, shape=(4,), typestr="<f4", data=(ptr, readonly))
        return x

    def _save_from_rgba(self, r, g, b, a):
        r = max(0.0, min(1.0, float(r)))
        g = max(0.0, min(1.0, float(g)))
        b = max(0.0, min(1.0, float(b)))
        a = max(0.0, min(1.0, float(a)))
        self._val = F4(r, g, b, a)

    def _save_from_int(self, color):
        v = color
        a = v % 256
        v = v >> 8
        b = v % 256
        v = v >> 8
        g = v % 256
        v = v >> 8
        r = v % 256
        self._save_from_rgba(r, g, b, a)

    def _save_from_tuple(self, color):
        color = tuple(float(c) for c in color)
        if len(color) == 1:
            self._save_from_rgba(color[0], color[0], color[0], 1)
        elif len(color) == 2:
            self._save_from_rgba(color[0], color[0], color[0], color[1])
        elif len(color) == 3:
            self._save_from_rgba(color[0], color[1], color[2], 1)
        elif len(color) == 4:
            self._save_from_rgba(*color)
        else:
            raise ValueError(f"Color tuple must have 1-4 value, got: {color}")

    def _save_from_str(self, color):
        color = color.lower()
        if color.startswith("#"):
            # todo: Hex RGB or RGBA
            raise NotImplementedError()
        else:
            try:
                self._save_from_int(CSS_NAMES[color])
            except KeyError:
                raise ValueError(f"Unknown color: '{color}'")

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
    def i(self):
        """Return as int in which the rgba values are packed."""
        r, g, b, a = self.rgba
        return (
            (int(r * 255) << 24)
            + (int(g * 255) << 16)
            + (int(b * 255) << 8)
            + int(a * 255)
        )

    @property
    def hex(self):
        """The CSS hex string, e.g. "#00ff00". The alpha channel is ignored."""
        return "#" + hex(self.i)[2:]

    @property
    def css(self):
        """The CSS color string. If the alpha is 1, returns as hex, otherwise
        returns e.g. "rgba(0,255,0,0.5)".
        """
        if self.a == 1:
            return self.hex
        else:
            r, g, b, a = self.rgba
            return "rgba({int(255*r)},int(255*g),int(255*b),{a:03f})"

    # todo: __add__, lighter(), darker(), etc.


CSS_NAMES = {
    "red": 0xFF0000FF,
    "green": 0x00FF00FF,
    "blue": 0x0000FFFF,
    # todo: etc.
}
