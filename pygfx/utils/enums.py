"""
The enums used in pygfx. The enums are all available from the root ``pygfx`` namespace.

.. currentmodule:: pygfx.utils.enums

.. autosummary::
    :toctree: utils/enums
    :template: ../_templates/custom_layout.rst

    BindMode
    ColorMode
    CoordSpace
    EdgeMode
    ElementFormat
    MarkerShape
    SizeMode
    TextAlign
    TextAnchor
    VisibleSide
    PixelFilter

"""

from wgpu.utils import BaseEnum
from typing import TypeAlias, Literal


__all__ = [
    "BindMode",
    "ColorMode",
    "CoordSpace",
    "EdgeMode",
    "ElementFormat",
    "MarkerShape",
    "PixelFilter",
    "SizeMode",
    "TextAlign",
    "TextAnchor",
    "VisibleSide",
]


class Enum(BaseEnum):
    """Enum base class for pygfx."""


class EdgeMode(Enum):
    centered = None  #: Centered edges (half the width on each side).
    inner = None  #: Inner edges (the width is added to the inside).
    outer = None  #: Outer edges (the width is added to the outside).


class ColorMode(Enum):
    """The ColorMode enum specifies how an object's color is established."""

    auto = None  #: Use (multiply) all the color sources in (material.color, geometry.colors, material.map) if available.
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
    vertex = None  #: Use a per-vertex size specified with ``geometry.sizes``.


class RotationMode(Enum):
    """The RotationMode enum specifies how an object's rotation is established.

    Currently only used for PointsMaterial.
    """

    uniform = None  #: Use a uniform rotation.
    vertex = None  #: Use a per-vertex rotation specified with ``geometry.rotations``.


class CoordSpace(Enum):
    """The CoordSpace enum specifies a coordinate space."""

    model = None  #: The space relative to the object. When the object (or a parent) is e.g. scaled with ``wobject.local.scale = 2`` the thing becomes bigger.
    world = None  #: The space of the scene (the root object). Scaling or rotating of objects does not affect the thing's size or orientation.
    screen = None  #: The screen space (in logical pixels). The thing's size is not affected by zooming or scaling.


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


class TextAlign(Enum):
    """How text is aligned."""

    start = None  #: The same as left if direction is left-to-right and right if direction is right-to-left.
    end = None  #: The same as right if direction is left-to-right and left if direction is right-to-left.
    left = None  #: The text is aligned to the left edge.
    right = None  #: The text is aligned to the right edge.
    center = None  #: The text is centered between the left and right edges.
    justify = None  #: The words are spread to fill the full space set by max_width.
    justify_all = None  #: Like justifym but also justify the last line.
    auto = None  #: Unspecified alignment (use default).


class TextAnchor(Enum):
    """How text is aligned."""

    top_left = "top-left"
    top_center = "top-center"
    top_right = "top-right"

    baseline_left = "baseline-left"
    baseline_center = "baseline-center"
    baseline_right = "baseline-right"

    middle_left = "middle-left"
    middle_center = "middle-center"
    middle_right = "middle-right"

    bottom_left = "bottom-left"
    bottom_center = "bottom-center"
    bottom_right = "bottom-right"


# TODO: I experimented with using a Literal[] here, an idea discussed in https://github.com/pygfx/wgpu-py/issues/720.
# We should eventually use the same approach to all enums (either an Enum class, or Literal type aliases).

PixelFilter: TypeAlias = Literal[
    "nearest", "linear", "tent", "disk", "bspline", "mitchell", "catmull"
]  #:
""" The type of interpolation for flushing the result of a renderer to a target.

The filter is used both when upsampling and downsampling. The recommended (and default) is "mitchell".
Note that when the source and target image are of the same size, the filter is always nearest.

* "nearest": nearest-neighbour interpolation. Note that this introduces aliasing when downsampling.
* "linear": linear interpolation. Note that this introduces aliasing when downsampling.
* "tent": linearly combines samples based on their distance using a smaller kernel than the cubic filters. When upsampling, it does the same a 'linear'.
* "disk": a circular filter shape to show individual pixels in upsampling cases.
* "bspline": cubic spline of the type b-spline, which is rather smooth but has no overshoot.
* "mitchell": cubic spline of the type Mitchel-Netravali, which is optimized for image interpolation.
* "catmull": cubic spline of the type Catmull-Rom, which is sharper, but has more ringing effects.
"""


# NOTE: Don't forget to add new enums to the toctree and __all__
