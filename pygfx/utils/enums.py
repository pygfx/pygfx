"""
The enums used in pygfx. The enums are all available from the root ``pygfx`` namespace.

.. currentmodule:: pygfx.utils.enums

.. autosummary::
    :toctree: utils/enums
    :template: ../_templates/custom_layout.rst

    ColorMode
    CoordSpace
    MarkerShape
    RenderMask
    SizeMode
    ElementFormat
    VisibleSide
    BindMode

"""

__all__ = [
    "RenderMask",
    "ColorMode",
    "SizeMode",
    "CoordSpace",
    "MarkerShape",
    "ElementFormat",
    "VisibleSide",
    "BindMode",
]

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
            if not key.startswith("_"):
                val = key if val is None else val
                if not isinstance(val, (int, str)):
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
        for name in ["__dir__", "__iter__", "__getitem__", "__setattr__", "__repr__"]:
            setattr(klass, name, types.MethodType(getattr(cls, name), klass))
        return klass

    def __dir__(cls):
        # Support dir(enum). Note that this order matches the definition, but dir() makes it alphabetic.
        return cls.__fields__

    def __iter__(cls):
        # Support list(enum), iterating over the enum, and doing ``x in enum``.
        return iter([getattr(cls, key) for key in cls.__fields__])

    def __getitem__(cls, key):
        # Support enum[key]
        return cls.__dict__[key]

    def __repr__(cls):
        name = cls.__name__
        options = []
        for key in cls.__fields__:
            val = cls[key]
            options.append(f"'{key}' ({val})" if isinstance(val, int) else f"'{val}'")
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


class CoordSpace(Enum):
    """The CoordSpace enum specifies a coordinate space."""

    model = None  #: the WorldObject's own coordinate space.
    world = None  #: the coordinate space of the scene.
    screen = None  #: the coordiate space in logical pixels.


class MarkerShape(Enum):
    """The MarkerShape enum specifies the shape of a markers in the PointsMarkerMaterial."""

    circle = None  #: ● A circular shape (i.e. a disk).
    ring = None  #: ○ A circular shape with a hole in the middle.
    square = None  #: ■ A big square shape (sized to encompass the circle shape).
    diamond = None  #: ♦ A rotated square (sized to fit inside the circle).
    plus = None  #: + A plus symbol.
    cross = None  #: x A rotated plus symbol.
    asterix = None  #: ✳️ A plus and a cross combined.
    triangle_up = None  #: ▲
    triangle_down = None  #: ▼
    triangle_left = None  #: ◀
    triangle_right = None  #: ▶
    heart = None  #: ♥
    spade = None  #: ♠
    club = None  #: ♣
    pin = None  #: 📍


class ElementFormat(Enum):
    """The base elements to specify formats.

    These values can be used to compose formats of various layouts, e.g. 2D
    positions with "2xf4", rgb colors with "3xu8" or matrices with "4x4xf32".
    The purpose is to provide a common representation for simple formats, that
    can be used for buffers, textures and uniform buffers.
    """

    i1 = None  #: A signed 8bit integer.
    u1 = None  #: An unsigned 8-bit integer (byte).
    i2 = None  #: A signed 16-bit integer.
    u2 = None  #: An unsigned 16-bit integer.
    i4 = None  #: A signed 32-bit integer.
    u4 = None  #: An unsigned 32-bit integer.
    f2 = None  #: A 16-bit float.
    f4 = None  #: A 32-bit float.


class VisibleSide(Enum):
    """The VisibleSide enum specifies what side of a mesh is visible.

    Note that this is the inverse of the "CullMode", as it specifies what
    side is visible rather than what side is culled.
    """

    front = None  #: The front is visible.
    back = None  #: The back is visible.
    both = None  #: Both the front and back are visible.


class BindMode(Enum):
    """The BindMode enum specifies how a skinned mesh is bound to its skeleton."""

    attached = (
        "attached"  #: The skinned mesh shares the same world space as the skeleton.
    )
    detached = "detached"  #: The skinned mesh has its own world space.


# NOTE: Don't forget to add new enums to the toctree and __all__
