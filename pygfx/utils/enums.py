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
# and simply maps to strings. This is (nearly) the same implementation of the
# enums in wgpu-py.
#
# Making it render correctly in the Sphinx docs is a bit of a pain though.

import types


def _get_values_for_docs(cls):
    reprs = []
    for key in cls.__dir__():
        val = cls[key]
        if isinstance(val, int):
            reprs.append(f"'{key}' ({val})")
        else:
            reprs.append(f"'{val}'")
    return reprs


class EnumType(type):
    """Enum metaclas."""

    def __new__(metacls, name, bases, dct):

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
        cls = super().__new__(metacls, name, bases, dct)

        # Attach some fields
        cls.__fields__ = tuple(member_map)
        cls.__members__ = types.MappingProxyType(member_map)  # enums.Enum compat

        # Create bound methods
        cls.__dir__ = types.MethodType(metacls.__dir__, cls)
        cls.__iter__ = types.MethodType(metacls.__iter__, cls)
        cls.__getitem__ = types.MethodType(metacls.__getitem__, cls)
        cls.__repr__ = types.MethodType(metacls.__repr__, cls)
        cls.__setattr__ = types.MethodType(metacls.__setattr__, cls)

        return cls

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
        options = ", ".join(_get_values_for_docs(cls))
        return f"<pygfx.{name} enum with options: {options}>"

    def __setattr__(cls, name, value):
        if name.startswith("_"):
            super().__setattr__(name, value)
        else:
            raise RuntimeError("Cannot set values on an enum.")


class Enum(metaclass=EnumType):
    def __init__(self):
        raise RuntimeError("Connot instantiate an enum.")


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


# NOTE: Don't forget to add new enums to __all__
