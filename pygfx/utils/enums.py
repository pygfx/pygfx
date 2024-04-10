"""
The enums used in pygfx. The enums are all available from the root ``pygfx`` namespace.

.. currentmodule:: pygfx.utils.enums

.. autosummary::
    :toctree: utils/enums
    :template: ../_templates/custom_layout.rst

    RenderMask
    ColorMode
    SizeMode

"""

__all__ = ["RenderMask", "ColorMode", "SizeMode"]

# We implement a custom enum class that's much simpler than Python's enum.Enum,
# and simply maps to strings or ints. The enums are classes, so IDE's provide
# autocompletion, and documenting with Sphinx is easy. That does mean we need a
# metaclass though.

import types


class EnumType(type):
    """Enum metaclas."""

    def __new__(cls, name, bases, dct):

        # Collect and check fields
        member_map = {}
        for key, val in dct.items():
            if key.startswith("_"):
                continue
            if val is None:
                val = key
            elif not isinstance(val, (int, str)):
                raise TypeError("Enum fields must be str or int.")
            member_map[key] = val

        # Some field values may have been updated
        dct.update(member_map)

        # Create class
        klass = super().__new__(cls, name, bases, dct)

        # Attach some fields
        klass.__fields__ = tuple(member_map)
        klass.__members__ = types.MappingProxyType(member_map)  # enums.Enum compat

        # Create bound methods
        klass.__dir__ = types.MethodType(cls.__dir__, klass)
        klass.__iter__ = types.MethodType(cls.__iter__, klass)
        klass.__getitem__ = types.MethodType(cls.__getitem__, klass)
        klass.__repr__ = types.MethodType(cls.__repr__, klass)
        klass.__setattr__ = types.MethodType(cls.__setattr__, klass)

        return klass

    def __dir__(cls):
        # Support dir(enum).
        # Note that the returned order matches the definition, but dir() returns in alphabetic order.
        return cls.__fields__

    def __iter__(cls):
        # Support list(enum) and iterating over the enum.
        return iter([getattr(cls, key) for key in cls.__fields__])

    def __getitem__(cls, key):
        # Support enum[key]
        return cls.__dict__[key]

    def __repr__(cls):
        name = cls.__name__
        options = []
        for key in cls.__fields__:
            val = cls[key]
            if isinstance(val, int):
                options.append(f"'{key}' ({val})")
            else:
                options.append(f"'{val}'")
        return f"<pygfx.{name} enum with options: {', '.join(options)}>"

    def __setattr__(cls, name, value):
        if name.startswith("_"):
            super().__setattr__(name, value)
        else:
            raise RuntimeError("Cannot set values on an enum.")


class Enum(metaclass=EnumType):
    """Enum base class."""

    def __init__(self):
        raise RuntimeError("Connot instantiate an enum.")


# --- The enums


class RenderMask(Enum):
    """The RenderMask enum specifies the render passes in which an object participates."""

    auto = 0  #: Select the appropriate render passes automatically.
    opaque = 1  #: Only render in the opaque pass.
    transparent = 2  #: Only render in the transparancy pass.
    all = 3  #: Render in both passes.


class ColorMode(Enum):
    """The ColorMode enum specifies how an object's color is established."""

    auto = None  #: Use either ``uniform`` and ``vertex_map``, depending on whether ``map`` is set.
    uniform = None  #: Use the uniform color (usually ``material.color``).
    vertex = None  #: Use the per-vertex color specified in the geometry  (usually  ``geometry.colors``).
    face = None  #: Use the per-face color specified in the geometry  (usually  ``geometry.colors``).
    vertex_map = None  #: Use per-vertex texture coords (``geometry.texcoords``), and sample these in ``material.map``.
    face_map = None  #: Use per-face texture coords (``geometry.texcoords``), and sample these in ``material.map``.


class SizeMode(Enum):
    """The SizeMode enum specifies how an object's size/width/thickness is established."""

    uniform = None  #: Use a uniform size.
    vertex = None  #: Use a per-vertex size specified on the geometry.


# NOTE: Don't forget to add new enums to the toctree and __all__
