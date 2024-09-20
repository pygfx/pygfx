"""
The enums used in pygfx. The enums are all available from the root ``pygfx`` namespace.

.. currentmodule:: pygfx.utils.enums

.. autosummary::
    :toctree: utils/enums
    :template: ../_templates/custom_layout.rst

    ColorMode
    CoordSpace
    EdgeMode
    MarkerShape
    RenderMask
    SizeMode
    ElementFormat
    VisibleSide
    BindMode

"""

from wgpu.utils import BaseEnum


__all__ = [
    "RenderMask",
    "ColorMode",
    "EdgeMode",
    "SizeMode",
    "CoordSpace",
    "MarkerShape",
    "ElementFormat",
    "VisibleSide",
    "BindMode",
]


class Enum(BaseEnum):
    """Enum base class for pygfx."""


class RenderMask(Enum):
    """The RenderMask enum specifies the render passes in which an object participates."""

    auto = 0  #: Select the appropriate render passes automatically.
    opaque = 1  #: Only render in the opaque pass.
    transparent = 2  #: Only render in the transparancy pass.
    all = 3  #: Render in both passes.


class EdgeMode(Enum):
    centered = None  #: Centered edges (half the width on each side).
    inner = None  #: Inner edges (the width is added to the inside).
    outer = None  #: Outer edges (the width is added to the outside).


class ColorMode(Enum):
    """The ColorMode enum specifies how an object's color is established."""

    auto = None  #: Use either ``uniform`` and ``vertex_map``, depending on whether ``map`` is set.
    uniform = None  #: Use the uniform color (usually ``material.color``).
    vertex = None  #: Use the per-vertex color specified in the geometry  (usually  ``geometry.colors``).
    face = None  #: Use the per-face color specified in the geometry  (usually  ``geometry.colors``).
    vertex_map = None  #: Use per-vertex texture coords (``geometry.texcoords``), and sample these in ``material.map``.
    face_map = None  #: Use per-face texture coords (``geometry.texcoords``), and sample these in ``material.map``.
    debug = (
        None  #: Use colors most suitable for debugging. Defined on a per shader basis.
    )


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
    custom = None  # Custom shape allowing users to provide their own SDF function


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
