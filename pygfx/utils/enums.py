"""
Enums used in pygfx.
"""

# We implement a custom enum class that's much simpler than Python's enum.Enum,
# and simply maps to strings. This is (nearly) the same implementation of the
# enums in wgpu-py.

_use_sphinx_repr = False


class Enum:
    def __init__(self, name, **kwargs):
        self._name = name
        for key, val in kwargs.items():
            val = key if val is None else val
            setattr(self, key, val)  # == self.__dict__[key] = val

    def __dir__(self):
        # Support dir(enum).
        # Note that the returned order matches the definition, but dir() returns in alphabetic order.
        return [key for key in self.__dict__.keys() if not key.startswith("_")]

    def __iter__(self):
        # Support list(enum) and iterating over the enum.
        return iter([getattr(self, key) for key in self.__dir__()])

    def __getitem__(self, key):
        # Support enum[key]
        return self.__dict__[key]

    def __repr__(self):
        if _use_sphinx_repr:  # no-cover
            return ""
        values = ", ".join(f"{repr(x)}" for x in self)
        return f"<pygfx.{self._name} enum with values: {values}>"


RenderMask = Enum(
    "RenderMask",
    auto=0,
    opaque=1,
    transparent=2,
    all=3,
)


ColorMode = Enum(
    "ColorMode",
    auto=None,
    uniform=None,
    vertex=None,
    face=None,
    vertex_map=None,
    face_map=None,
)


SizeMode = Enum(
    "SizeMode",
    uniform=None,
    vertex=None,
)
