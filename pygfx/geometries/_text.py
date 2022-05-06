"""
This module implements text geometry.

The TextGeometry class accepts a list of TextItem objects, and turns these into
positions and atlas_indices. Each TextItem object represents a piece of text
with specific font properties.

The TextGeometry object has a few text properties that affect the positioning
of the text.

The text_geometry() function is the user-frienfly API to generate text geometry.

For details about the text rendering process, see pygfx/utils/text/README.md
"""

import numpy as np

from ._base import Geometry
from ..resources import Buffer
from ..utils.text import FontProps, find_font, shape_text


def text_geometry(
    *,
    text=None,
    markdown=None,
    family=None,
    max_width=None,
    line_height=None,
    text_align=None,
):
    """Generate text geometry."""

    # === Itemization - generate a list of TextItem objects
    items = []
    if text:
        if not isinstance(text, str):
            raise TypeError("Text must be a Unicode string.")
        items.append(TextItem(text, family))
    if markdown:
        raise NotImplementedError()

    return TextGeometry(
        items, max_width=max_width, line_height=line_height, text_align=text_align
    )


class TextItem:
    """A text item represents a unit piece of text that is formatted
    in a specific way. The TextGeometry positions a list of text items so that
    they together display the intended total text.
    """

    def __init__(self, text, font_props=None):

        # Set font props
        if font_props is None:
            font_props = FontProps()  # default

        # === Font selection
        font_filename = find_font(font_props)

        # === Shaping - generate indices and positions
        indices, positions, coverages = shape_text(text, font_filename)

        # or ..
        # glyph_indices, positions = shape_text(text, font_props, font_filename)
        # atlas_indices, position_updates = generate_glyphs(glyph_indices, font_filename)

        # indices = np.zeros((len(text),), np.uint32)
        # positions = np.zeros((len(text), 3), np.float32)
        #
        # for i in range(len(text)):
        #     positions[i] = i * font_props.size, 0, 0

        # Store stuff for the geometry to use
        self.props = font_props
        self.indices = indices
        self.positions = positions
        self.coverages = coverages


class TextGeometry(Geometry):
    """Produce renderable geometry from a list of TextItem objects."""

    def __init__(self, text_items, max_width=0, line_height=1.2, text_align="left"):
        super().__init__()

        # Check incoming items
        self._text_items = tuple(text_items)
        for item in self._text_items:
            if not isinstance(item, TextItem):
                raise TypeError("TextGeometry only accepts TextItem objects.")

        # Compose the items in a single geometry
        indices_arrays = []
        coverages_arrays = []
        positions_arrays = []
        glyph_count = 0
        for item in self._text_items:
            assert item.indices.dtype == np.uint32
            assert item.positions.dtype == np.float32
            assert item.positions.shape == (item.indices.size, 2)
            item.offset = glyph_count
            glyph_count += item.indices.size
            indices_arrays.append(item.indices)
            coverages_arrays.append(item.coverages)
            positions_arrays.append(item.positions)

        # Store
        self.indices = Buffer(np.concatenate(indices_arrays, 0))
        self.coverages = Buffer(np.concatenate(coverages_arrays, 0))
        self.positions = Buffer(np.concatenate(positions_arrays, 0))

        # Set props
        # todo: each of the below line invokes the positioning algotrithm :/
        self.max_width = max_width
        self.line_height = line_height
        self.text_align = text_align

    @property
    def max_width(self):
        """The maximum width of the text. Text will wrap if beyond this
        limit. The coordinate system that this applies to depends on
        the material, but it's coordinate system that the text_size
        applies to. Set to 0 for no wrap. Default 0.
        """
        # todo: double-check these docstrings on coord systems when we're done
        return self._max_width

    @max_width.setter
    def max_width(self, width):
        self._max_width = float(width or 0)
        self._position()

    @property
    def line_height(self):
        """The height of a line of text, used to set the distance between
        linces. Represented as a factor of the font size. Default 1.2.
        """
        return self._line_height

    @line_height.setter
    def line_height(self, heigh):
        self._line_height = float(heigh or 1.2)
        self._position()

    @property
    def text_align(self):
        """Set the alignment of the text. Can be left, right, center, or justify.
        Default "left".
        """
        return self._text_align

    @text_align.setter
    def text_align(self, align):
        alignments = "left", "right", "center", "justify"
        if align is None:
            align = "left"  # todo: or does center make more sense in a viz lib?
        elif isinstance(align, int):
            align = {-1: "left", 0: "center", 1: "right"}[align]
        elif not isinstance(align, str):
            raise TypeError("Align must be a string.")
        align = align.lower()
        if align not in alignments:
            raise ValueError(f"Align must be one of {alignments}")
        self._text_align = align
        self._position()

    def _position(self):

        # === Positioning
        # Handle alignment, wrapping and all that.
        # Note, could render the text in a curve or something.
        # todo: perhaps use a hook for custom positioning effects?

        for item in self._text_items:
            positions = item.positions.copy()
            positions *= 12
            # todo: tweak positions
            if not (positions == item.positions).all():
                i1, i2 = item.offset, item.offset + positions.shape[0]
                self.positions.data[i1:i2] = positions
                self.positions.update_range(i1, i2)
