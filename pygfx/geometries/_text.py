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
        # We could put font properties in their own buffer(s), but to
        # safe memory, we encode them in the top bits of the atlas
        # indices. This seems like a good place, because these top bits
        # won't be used (2**24 is more than enough glyphs), and the
        # glyph index is a rather "opaque" value to the user anyway.
        # You can think of the new glyph index as the index to the glyph
        # in the atlas, plus props to tweak its appearance.
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
        # Int offset. Note that this means that a glyph item is bound to a TextGeometry
        self.offset = 0


class TextGeometry(Geometry):
    """Produce renderable geometry from a list of TextItem objects.

    Parameters:
        text (str): the text to render (optional). Either text or markdown must be given.
        markdown (str): the text to render, formatted as markdown (optional). TODO
        font_size (float): the size of the font, in scene coordinates or pixel screen
            coordinates, depending on ``material.screen_space``. Default 12.
        max_width (float): the maximum width of the text. Words are wrapped if necessary.
            A value of zero means no wrapping. Default zero.
        line_height (float): a factor to scale the distance between lines. A value
            of 1 means the "native" font's line distance. Default 1.2.
        text_align (str): How to align the text. Not implemented.
        family (str, tuple): the name(s) of the font to prefer. If multiple names
            are given, they are preferred in the given order. Characters that are
            not supported by any of the given fonts are rendered with the default
            font (from the Noto Sans collection).
    """

    def __init__(
        self,
        text=None,
        *,
        markdown=None,
        font_size=12,
        max_width=0,
        line_height=1.2,
        text_align="left",
        family=None,
    ):
        super().__init__()

        # Check inputs
        inputs = text, markdown
        if all(i is not None for i in inputs):
            raise TypeError("Either text or markdown must be given, not both.")

        # Init stub buffers
        self.indices = None
        self.positions = None
        self.sizes = None

        # Disable positioning, so we can initialize first
        self._do_positioning = False

        # Process input
        if text is not None:
            self.set_text(text, family=family)
        elif markdown is not None:
            self.set_markdown(markdown, family=family)
        else:
            raise TypeError("Either text or markdown must be given")

        # Set props
        self.font_size = font_size
        self.max_width = max_width
        self.line_height = line_height
        self.text_align = text_align

        # Positioning
        self._do_positioning = True
        self.apply_layout()

    def set_text_items(self, text_items):
        """Provide new text in the form of a list of TextItem objects.

        A note on performance: if the new text consists of more glyphs
        than the current, new (larger) buffers are created. If the
        number of glyphs is smaller, the buffers are not replaced, but
        simply not fully used.
        """

        # Check incoming items
        glyph_items = []
        for item in text_items:
            if not isinstance(item, TextItem):
                raise TypeError("TextGeometry only accepts TextItem objects.")
            glyph_items.extend(item.convert_to_glyphs())

        # We cannot have nonzero buffers, so we create a single space
        if not glyph_items:
            glyph_items = TextItem(" ").convert_to_glyphs()

        # Re-order the items if needed, based on text direction
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

        # Iterate over the glyph_items to set offsets
        glyph_count = 0
        for item in self._glyph_items:
            assert item.indices.dtype == np.uint32
            assert item.positions.dtype == np.float32
            assert item.positions.shape == (item.indices.size, 2)
            item.offset = glyph_count
            glyph_count += item.indices.size

        # Do we need new buffers?
        if self.indices is None or self.indices.nitems < glyph_count:
            self.indices = Buffer(np.zeros((glyph_count,), np.uint32))
            self.positions = Buffer(np.zeros((glyph_count, 2), np.float32))
            self.sizes = Buffer(np.zeros((glyph_count,), np.float32))

        # Copy the glyph arrays into the buffers
        for item in self._glyph_items:
            i1, i2 = item.offset, item.offset + item.indices.shape[0]
            self.indices.data[i1:i2] = item.indices
            self.positions.data[i1:i2] = item.positions

        # Disable the unused space
        self.indices.data[glyph_count:] = 0
        self.positions.data[glyph_count:] = 0

        # Schedule the array to be uploaded
        self.indices.update_range(0, self.indices.nitems)
        # self.positions.update_range(0, self.positions.nitems)
        self.apply_layout()

    def set_text(self, text, family=None, style=None, weight=None):
        """Update the geometry's text.

        A note on performance: if the new text consists of more glyphs
        than the current, new (larger) buffers are created. If the
        number of glyphs is smaller, the buffers are not replaced, but
        simply not fully used.

        Parameters:
            text (str): the text to render.
            family (str, tuple): the name(s) of the font to prefer. If multiple names
                are given, they are preferred in the given order. Characters that are
                not supported by any of the given fonts are rendered with the default
                font (from the Noto Sans collection).
            style (str): The style of the font (normal, italic, oblique).
            weight (str, int): The weight of the font. E.g. "normal" or "bold" or a
                number between 100 and 900.
        """

        if not isinstance(text, str):
            raise TypeError("Text must be a Unicode string.")
        font_props = FontProps(family=family, style=style, weight=weight)

        # === Itemization - generate a list of TextItem objects
        # todo: when we impove the layout, replace below with regex search on last-space-in-series-of-spaces, and newlines
        items = []
        for piece in text.split():
            items.append(TextItem(piece, font_props))

        self.set_text_items(items)

    def set_markdown(self, text, family=None):
        """Update the geometry's text using markdown formatting.

        The supported subset of markdown is limited to surrounding words with
        single and double stars for oblique and bold text respectively.
        """

        if not isinstance(text, str):
            raise TypeError("Markdown text must be a Unicode string.")
        font_props = FontProps(family=family)

        items = []
        for piece in text.split():
            props = font_props
            if piece.startswith("*") and piece.endswith("*"):
                if piece.startswith("**") and piece.endswith("**"):
                    piece = piece[2:-2]
                    props = props.copy(weight="bold")
                else:
                    piece = piece[1:-1]
                    props = props.copy(style="slanted")
            items.append(TextItem(piece, props))

        self.set_text_items(items)

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
        self.apply_layout()

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
        self.apply_layout()

    @property
    def line_height(self):
        """The height of a line of text, used to set the distance between
        lines. Represented as a factor of the font size. Default 1.2.
        """
        return self._line_height

    @line_height.setter
    def line_height(self, heigh):
        self._line_height = float(heigh or 1.2)
        self.apply_layout()

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
        self.apply_layout()

    def apply_layout(self):
        """Apply the layout algorithm to position the (internal) glyph items.

        To overload this with a custom layout, overload ``_apply_layout()``.
        """

        if self._do_positioning:
            self._apply_layout()

    def _apply_layout(self):

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
