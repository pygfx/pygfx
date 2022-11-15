"""
This module implements text geometry.

The TextGeometry class accepts a list of TextItem objects, and turns these into
positions and atlas_indices. Each TextItem object represents a piece of text
with specific font properties.

The TextGeometry object has a few text properties that affect the positioning
of the text.

The text_geometry() function is the user-friendly API to generate text geometry.

For details about the text rendering process, see pygfx/utils/text/README.md
"""

import numpy as np

from ._base import Geometry
from ..resources import Buffer
from ..utils.text import FontProps, font_manager, shape_text, generate_glyph


def text_geometry(
    *,
    text=None,
    markdown=None,
    font_size=12,
    max_width=None,
    line_height=None,
    text_align=None,
    **font_props,
):
    """Generate text geometry."""

    font_props = FontProps(**font_props)

    # === Itemization - generate a list of TextItem objects
    items = []
    if text:
        if not isinstance(text, str):
            raise TypeError("Text must be a Unicode string.")
        # todo: replace with regex search on last-space-in-series-of-spaces, and newlines
        for piece in text.split():
            items.append(TextItem(piece, font_props))
    if markdown:
        raise NotImplementedError()

    return TextGeometry(
        items,
        font_size=font_size,
        max_width=max_width,
        line_height=line_height,
        text_align=text_align,
    )


class TextItem:
    """A text item represents a unit piece of text that is formatted,
    in a specific way. The TextGeometry positions a list of text items so that
    they together display the intended total text.
    """

    def __init__(self, text, font_props=None, *, allow_break=True):

        if font_props is None:
            font_props = FontProps()
        elif not isinstance(font_props, FontProps):
            raise TypeError("font_props is not a FontProps object.")
        self._font_props = font_props

        self._text = text
        self._allow_break = bool(allow_break)

    @property
    def text(self):
        """The text for this item."""
        return self._text

    @property
    def font_props(self):
        return self._font_props

    def convert_to_glyphs(self):
        """Convert this text item in one or more GlyphItem objects.
        This process includes font selection, shaping, and glyph generation.
        """
        text = self._text

        # === Font selection
        text_pieces = font_manager.select_font_for_text(
            self._text, self._font_props.family
        )

        glyph_items = []
        for text, font in text_pieces:

            # === Shaping - generate indices and positions
            glyph_indices, positions, full_width, meta = shape_text(text, font.filename)

            # === Glyph generation (populate atlas)
            atlas_indices = generate_glyph(glyph_indices, font.filename)

            self._encode_font_props_in_atlas_indices(atlas_indices)
            glyph_items.append(GlyphItem(positions, atlas_indices, full_width, meta))

        # Set props so these items will be grouped correctly
        glyph_items[0].allow_break = self._allow_break
        glyph_items[0].margin_before = glyph_items[0].meta["space_width"] / 2
        glyph_items[-1].margin_after = glyph_items[-1].meta["space_width"] / 2
        return glyph_items

    def _encode_font_props_in_atlas_indices(self, atlas_indices):
        is_slanted = self._font_props.style in ("italic", "oblique", "slanted")
        if is_slanted:
            atlas_indices += 0x08000000
        weight_0_15 = int((max(150, self._font_props.weight) - 150) / 50 + 0.4999)
        atlas_indices += max(0, min(15, weight_0_15)) << 28


class GlyphItem:
    """A small collection of glyphs that represents a piece of text. Intended for internal use only."""

    def __init__(self, positions, indices, width, meta):
        # Arrays with glyph data
        self.positions = positions
        self.indices = indices
        # Layout data
        self.width = width
        self.meta = meta
        self.allow_break = False
        self.margin_before = 0
        self.margin_after = 0


class TextGeometry(Geometry):
    """Produce renderable geometry from a list of TextItem objects."""

    def __init__(
        self, text_items, font_size=12, max_width=0, line_height=1.2, text_align="left"
    ):
        super().__init__()

        # Check incoming items
        glyph_items = []
        for item in text_items:
            if not isinstance(item, TextItem):
                raise TypeError("TextGeometry only accepts TextItem objects.")
            glyph_items.extend(item.convert_to_glyphs())

        # Re-order the items if needed
        i = 0
        while i < len(glyph_items) - 1:
            item = glyph_items[i]
            if item.meta["direction"] == "rtl":
                i1 = i2 = i
                for j in range(i + 1, len(glyph_items)):
                    if item.meta["direction"] != "rtl":
                        break
                    i2 = j
                if i1 != i2:
                    glyph_items[i1 : i2 + 1] = reversed(glyph_items[i1 : i2 + 1])
                i = i2 + 1
            else:
                i += 1

        self._glyph_items = tuple(glyph_items)

        # Compose the items in a single geometry
        indices_arrays = []
        positions_arrays = []
        glyph_count = 0
        for item in self._glyph_items:
            assert item.indices.dtype == np.uint32
            assert item.positions.dtype == np.float32
            assert item.positions.shape == (item.indices.size, 2)
            item.offset = glyph_count
            glyph_count += item.indices.size
            indices_arrays.append(item.indices)
            positions_arrays.append(item.positions)

        # Store
        self.indices = Buffer(np.concatenate(indices_arrays, 0))
        self.positions = Buffer(np.concatenate(positions_arrays, 0))
        self.sizes = Buffer(np.zeros((self.positions.nitems,), np.float32))

        # Set props
        # todo: each of the below line invokes the positioning algorithm :/
        self.font_size = font_size
        self.max_width = max_width
        self.line_height = line_height
        self.text_align = text_align

    @property
    def font_size(self):
        """The size of the text. For text rendered in screen space, the
        size is in logical pixels. For text rendered in world space,
        the size represents world units. Note that the font_size is an
        indicative size - most glyphs are smaller, and some may be
        larger. Also note that some pieces of the text may have a
        different size due to formatting.
        """
        return self._font_size

    @font_size.setter
    def font_size(self, value):
        self._font_size = float(value)
        self._position()

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
        lines. Represented as a factor of the font size. Default 1.2.
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
            try:
                align = {-1: "left", 0: "center", 1: "right"}[align]
            except KeyError:
                raise ValueError("Align as an int must be -1, 0 or 1.")
        elif not isinstance(align, str):
            raise TypeError("Align must be a None, str or int.")
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

        font_size = self._font_size

        x_offset = 0
        for item in self._glyph_items:
            if x_offset > 0:
                x_offset += item.margin_before * font_size
            positions = item.positions * font_size + np.array([x_offset, 0])
            i1, i2 = item.offset, item.offset + positions.shape[0]
            self.positions.data[i1:i2] = positions
            self.positions.update_range(i1, i2)
            self.sizes.data[i1:i2] = font_size
            self.sizes.update_range(i1, i2)
            x_offset += item.width * font_size + item.margin_after * font_size
