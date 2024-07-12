"""
This module implements text geometry. This is where the text rendering comes
together. Most steps in the text rendering process come from pygfx.utils.text,
though most of the alignment is implemented here.

For details about the text rendering process, see pygfx/utils/text/README.md
"""

import numpy as np

from ..resources import Buffer
from ..utils import text as textmodule
from ._base import Geometry

_TEXT_ALIGNMENTS = [
    "start",
    "end",
    "left",
    "right",
    "center",
    "justify",
    "justify-all",
]
_TEXT_ALIGNMENTS_LAST = [
    "start",
    "end",
    "left",
    "right",
    "center",
    "justify",
    "auto",
]
ANCHOR_X_ALTS = {
    "left": "left",
    "center": "center",
    "middle": "center",
    "right": "right",
}
ANCHOR_Y_ALTS = {
    "top": "top",
    "middle": "middle",
    "center": "middle",
    "baseline": "baseline",
    "bottom": "bottom",
}


# We cache the extents of small whitespace strings to improve performance
WHITESPACE_EXTENTS = {}


class TextItem:
    """A formatted piece of text.

    A text item represents a unit piece of text that is formatted in a specific
    way. The TextGeometry converts these into GlyphItem's, and positions these
    so that they together display the intended total text.

    Parameters
    ----------
    text : str
        The text to display.
    font_props : textmodule.FontProps
        Format information for this text item.
    ws_before : str
        Whitespace before the text.
    ws_after : str
        Whitespace after the text.
    nl_before: str
        Newline before the text.
    nl_after: str
        Newline after the text.
    allow_break : bool
        If True, allow a linebreak to be placed after this piece of text.

    """

    def __init__(
        self,
        text,
        font_props=None,
        *,
        ws_before="",
        ws_after="",
        nl_before="",
        nl_after="",
        allow_break=True,
    ):
        if not isinstance(text, str):
            raise TypeError("TextItem text must be str.")
        if font_props is None:
            font_props = textmodule.FontProps()
        elif not isinstance(font_props, textmodule.FontProps):
            raise TypeError("TextItem font_props must be a FontProps object.")

        self._text = text
        self._font_props = font_props
        self._ws_before = ws_before
        self._ws_after = ws_after
        self._nl_before = nl_before
        self._nl_after = nl_after
        self._allow_break = bool(allow_break)

    @property
    def text(self):
        """The text for this item."""
        return self._text

    @property
    def font_props(self):
        """The FontProps object to format this text item."""
        return self._font_props

    @property
    def ws_before(self):
        """The whitespace text in front of this item."""
        return self._ws_before

    @property
    def ws_after(self):
        """The whitespace text after this item."""
        return self._ws_after

    @property
    def nl_before(self):
        """The newline text in front of this item."""
        return self._nl_before

    @property
    def nl_after(self):
        """The newline text after this item."""
        return self._nl_after

    @property
    def allow_break(self):
        """Whether or not a line-break is allowed after this item."""
        return self._allow_break


class GlyphItem:
    """A collection of glyphs that represents a unit piece of text.
    Intended for internal use only. In most cases one TextItem results
    in one GlyphItem, but it can be more if multiple fonts are required
    to render the TextItem.
    """

    def __init__(self, positions, indices, meta):
        # Arrays with glyph data
        self.positions = positions
        self.indices = indices
        # Layout data
        self.meta = meta
        self.extent = meta["extent"]
        self.direction = meta["direction"]
        self.ascender = meta["ascender"]
        self.descender = meta["descender"]
        self.allow_break = False
        self.margin_before = 0
        self.margin_after = 0
        self.newline_before = 0
        self.newline_after = 0
        # Int offset. Note that this means that a glyph item is bound to a TextGeometry
        self.offset = 0


class TextGeometry(Geometry):
    """Geometry specific for representing text.

    The TextGeometry creates and stores the geometry to render a piece of text.
    It can be provided as plain text or in markdown to support basic formatting.

    Parameters
    ----------
    text : str
        The plain text to render (optional).
    markdown : str
        The text to render, formatted as markdown (optional). See
        ``set_markdown()`` for details on the supported formatting.
    screen_space : bool
        Whether the text is rendered in screen space, in contrast to world
        space.
    font_size : float
        The size of the font, in object coordinates or pixel screen coordinates,
        depending on the value of the ``screen_space`` property. Default 12.
    anchor : str
        The position of the origin of the text. Default "middle-center".
    anchor_offset : float
        The offset (extra margin) for the 'top', 'bottom', 'left', and 'right' anchors.
    max_width : float
        The maximum width of the text. Words are wrapped if necessary. A value
        of zero means no wrapping. Default zero.
    line_height : float
        A factor to scale the distance between lines. A value of 1 means the
        "native" font's line distance. Default 1.2.
    text_align : str
        The horizontal alignment of the inline-level content. Can be "start",
        "end", "left", "right", "center", "justify" or "justify-all". Default
        "left". Text alignment is ignored for top to bottom ('ttb') and
        bottom to top ('btt') directions.
    text_align_last: str
        The horizontal alignment of the last line of the content
        element. Can be "start", "end", "left", "right", "center", "justify" or
        "auto". Default "auto". Text alignment is ignored for top to
        bottom ('ttb') and bottom to top ('btt') directions.
    family : str, tuple
        The name(s) of the font to prefer. If multiple names are given, they are
        preferred in the given order. Characters that are not supported by any
        of the given fonts are rendered with the default font (from the Noto
        Sans collection).
    direction : str
        The text direction. By default the text direction is determined
        automatically, but is always horizontal. Can be set to 'lrt', 'rtl',
        'ttb' or 'btt'.

    """

    def __init__(
        self,
        text=None,
        *,
        markdown=None,
        screen_space=False,
        font_size=12,
        anchor="middle-center",
        anchor_offset=0,
        max_width=0,
        line_height=1.2,
        text_align="left",
        text_align_last="auto",
        family=None,
        direction=None,
    ):
        super().__init__()

        # Init stub buffers
        self.indices = None
        self.positions = None
        self.sizes = None

        # Init props unrelated to layout
        self.screen_space = screen_space
        self._direction = direction

        # Disable layout, so we can initialize first
        self._do_layout = False

        # Check inputs
        inputs = text, markdown
        inputs = [i for i in inputs if i is not None]
        if len(inputs) > 1:
            raise TypeError("Either text or markdown must be given, not both.")

        # Process input
        if text is not None:
            self.set_text(text, family=family)
        elif markdown is not None:
            self.set_markdown(markdown, family=family)
        else:
            self.set_text_items([])

        # Set layout props
        self.font_size = font_size
        self.anchor = anchor
        self.anchor_offset = anchor_offset
        self.max_width = max_width
        self.line_height = line_height
        self.text_align = text_align
        self.text_align_last = text_align_last

        # Finish layout
        self._do_layout = True
        self.apply_layout()

    @property
    def screen_space(self):
        """Text size unit (screen vs local).

        Returns
        -------
        screen_space : bool
            If False, text size uses the unit of the local frame (e.g. cm).
            Otherwise it is uses the logical screen's units (e.g. px). The
            latter mode is typically used for annotations.

        Notes
        -----
        Regardless of choice, the local object's rotation and scale will still
        transform the text.

        """
        return self._store.screen_space

    @screen_space.setter
    def screen_space(self, value):
        self._store.screen_space = bool(value)

    def set_text_items(self, text_items):
        """Update the text using one or more TextItems.

        .. note::
            This is considered a low level function to provide more control. Use
            ``set_text`` or ``set_markdown`` for more convenience.

        Parameters
        ----------
        text_items : list
            A list of :class:`pygfx.TextItem` objects to update the text with.

        Notes
        -----
        If the new text has more glyphs than the current one a new (larger)
        buffer is created. Otherwise, the previous buffers are reused.
        """

        # This function can be considered the core of the text rendering.
        # Everything comes together here.

        # We cannot have nonzero buffers, so if we have nothing create a single space
        if not text_items:
            text_items = [TextItem(" ")]

        # Convert incoming text items to glyph items
        glyph_items = []
        for item in text_items:
            if not isinstance(item, TextItem):
                raise TypeError("TextGeometry only accepts TextItem objects.")
            first_index = len(glyph_items)

            # Text rendering steps: font selection, shaping, glyph generation
            text_pieces = self._select_font(item.text, item.font_props)
            for text, font in text_pieces:
                glyph_indices, positions, meta = self._shape_text(text, font.filename)
                atlas_indices = self._generate_glyph(glyph_indices, font.filename)
                self._encode_font_props_in_atlas_indices(
                    atlas_indices, item.font_props, font
                )
                glyph_items.append(GlyphItem(positions, atlas_indices, meta))

            # Get whitespace after and before the text
            margin_before = self._get_ws_extent(item.ws_before, text_pieces[0][1])
            margin_after = self._get_ws_extent(item.ws_after, text_pieces[-1][1])

            # Set props so these items will be grouped correctly
            first_item, last_item = glyph_items[first_index], glyph_items[-1]
            first_item.allow_break = item.allow_break
            first_item.margin_before = margin_before
            first_item.newline_before = item.nl_before
            last_item.margin_after = margin_after
            last_item.newline_after = item.nl_after

        # Layout pre-processing: re-order the items if needed, based on text direction
        i = 0
        while i < len(glyph_items) - 1:
            item = glyph_items[i]
            if item.direction in ("rtl", "btt"):
                i1 = i2 = i
                for j in range(i + 1, len(glyph_items)):
                    if glyph_items[j].direction not in ("rtl", "btt"):
                        break
                    i2 = j
                if i1 != i2:
                    glyph_items[i1 : i2 + 1] = reversed(glyph_items[i1 : i2 + 1])
                i = i2 + 1
            else:
                i += 1

        # We can now store the glyph items
        self._glyph_items = tuple(glyph_items)

        # We set the glyph offsets so we know their place in the total buffer
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

        # Disable the unused space by setting the sizes to zero, leading
        # to degenerate triangles. Leave the indices intact, so that
        # any errors will be detected by the old text shining through.
        self.sizes.data[glyph_count:] = 0

        # Trigger new indices and sizes to be uploaded to the GPU.
        self.sizes.update_range(0, self.indices.nitems)
        self.indices.update_range(0, self.indices.nitems)

        # Finalize the buffers by applying the layout algorithm.
        self.apply_layout()

    # %%%%% Entrypoint and itemization

    def set_text(self, text, family=None, style=None, weight=None):
        """Update the text.

        Parameters
        ----------
        text : str
            The new text.
        family : str, tuple
            The name(s) of the preferred font(s) to prefer. If multiple names are
            given, they are preferred in the given order. Characters that are
            not supported by any of the given fonts are rendered with the
            default font.
        style : str
            The style of the font (normal, italic, oblique). Default "normal".
        weight : str, int
            The weight of the font. E.g. "normal" or "bold" or a number between
            100 and 900. Default "normal".

        See Also
        --------
        TextGeometry.set_text_items

        """

        if not isinstance(text, str):
            raise TypeError("Text must be a Unicode string.")

        font_props = textmodule.FontProps(family=family, style=style, weight=weight)

        # Split the text in pieces using a tokenizer. We put the
        # whitespace as margin on the text items (whitespace is not rendered)
        items = []
        pending_whitespace = ""
        pending_newline = ""
        previous_was_newline = True
        for kind, piece in textmodule.tokenize_text(text):
            if kind == "ws":
                if previous_was_newline:
                    pending_whitespace += piece
                else:
                    items[-1]._ws_after += piece
            elif kind == "nl":
                if not items:
                    pending_newline += piece
                else:
                    items[-1]._nl_after += piece
                previous_was_newline = True
            else:
                items.append(TextItem(piece, font_props))
                items[-1]._ws_before += pending_whitespace
                pending_whitespace = ""

                items[-1]._nl_before += pending_newline
                pending_newline = ""
                previous_was_newline = False

        self.set_text_items(items)
        return self

    def set_markdown(self, markdown, family=None):
        """Update the text using markdown formatting.

        The supported subset of markdown is limited to surrounding pieces of
        text with single and double stars for slanted and bold text
        respectively.

        Parameters
        ----------
        markdown : str
            The new text (including markdown).
        family : str, tuple
            The name(s) of the font(s) to prefer. If multiple names are given,
            they are preferred in the given order. Characters that are not
            supported by any of the given fonts are rendered with the default
            font.

        See Also
        --------
        TextGeometry.set_text_items

        """

        if not isinstance(markdown, str):
            raise TypeError("Markdown text must be a Unicode string.")

        # Split text in pieces using a tokenizer
        pieces = list(textmodule.tokenize_markdown(markdown))

        # Put a virtual zero-char space in front and at the end, to make the alg simpler
        pieces.insert(0, ("ws", ""))
        pieces.append(("ws", ""))

        # Prepare font props
        font_props = textmodule.FontProps(family=family)
        pieces_props = [font_props for x in pieces]

        # Now resolve starts to detect bold and italic pieces
        bold_start = slant_start = None
        for i in range(len(pieces)):
            kind, piece = pieces[i]
            if kind == "stars":
                prev_is_wordlike = pieces[i - 1][0] not in ("ws", "punctuation")
                next_is_wordlike = pieces[i + 1][0] not in ("ws", "punctuation")
                if not prev_is_wordlike and next_is_wordlike:
                    # Might be a beginning
                    if piece == "**" and not bold_start:
                        bold_start = i
                    elif piece == "*" and not slant_start:
                        slant_start = i
                elif prev_is_wordlike and not next_is_wordlike:
                    # Might be an end
                    if piece == "**" and bold_start:
                        pieces[bold_start] = "", ""
                        pieces[i] = "", ""
                        for j in range(bold_start + 1, i):
                            pieces_props[j] = pieces_props[j].copy(weight="bold")
                        bold_start = None
                    elif piece == "*" and slant_start:
                        pieces[slant_start] = "", ""
                        pieces[i] = "", ""
                        for j in range(slant_start + 1, i):
                            pieces_props[j] = pieces_props[j].copy(style="slanted")
                        slant_start = None

        # Convert to TextItem objects
        items = []
        pending_whitespace = None
        for i in range(len(pieces)):
            kind, piece = pieces[i]
            if not kind:
                pass
            elif kind == "ws":
                if not items:
                    pending_whitespace = piece
                else:
                    items[-1]._ws_after += piece
            else:
                items.append(TextItem(piece, pieces_props[i]))
        if items and pending_whitespace:
            items[0]._ws_before += pending_whitespace

        self.set_text_items(items)
        return self

    # %%%%% Font selection

    def _select_font(self, text, font_props):
        """The font selection step. Returns (text, font_filename) tuples.
        Can be overloaded for custom behavior.
        """
        return textmodule.select_font(text, font_props)

    # %%%%% Shaping

    def _shape_text(self, text, font_filename):
        """The shaping step. Returns (glyph_indices, positions, meta).
        Can be overloaded for custom behavior.
        """
        return textmodule.shape_text(text, font_filename, self._direction)

    def _get_ws_extent(self, s, font):
        """Get the extent of a piece of whitespace text. Results of small strings are cached."""
        if not s:
            return 0
        elif len(s) <= 8:
            key = (s, self._direction, font.filename)
            try:
                return WHITESPACE_EXTENTS[key]
            except KeyError:
                meta = self._shape_text(s, font.filename)[2]
                extent = meta["extent"]
                WHITESPACE_EXTENTS[key] = extent
                return extent
        else:
            meta = self._shape_text(s, font.filename)[2]
            return meta["extent"]

    # %%%%% Glyph generation

    def _generate_glyph(self, glyph_indices, font_filename):
        """The glyph generation step. Returns an array with atlas indices.
        Can be overloaded for custom behavior.
        """
        return textmodule.generate_glyph(glyph_indices, font_filename)

    def _encode_font_props_in_atlas_indices(self, atlas_indices, font_props, font):
        # We could put font properties in their own buffer(s), but to
        # safe memory, we encode them in the top bits of the atlas
        # indices. This seems like a good place, because these top bits
        # won't be used (2**24 is more than enough glyphs), and the
        # glyph index is a rather "opaque" value to the user anyway.
        # You can think of the new glyph index as the index to the glyph
        # in the atlas, plus props to tweak its appearance.

        # We compare the font_props (i.e. the requested font variant)
        # with the actual font to see what correcion we need to apply.
        slanted_like = "italic", "oblique", "slanted"
        if font_props.style in slanted_like and font.style not in slanted_like:
            atlas_indices += 0x08000000
        weight_offset = font_props.weight - font.weight
        weight_0_15 = int((max(-250, weight_offset) + 250) / 50 + 0.4999)
        atlas_indices += max(0, min(15, weight_0_15)) << 28

    # %%%%% Layout

    def get_bounding_box(self):
        if self.screen_space:
            # There is no sensible bounding box for text in screen
            # space, except for the anchor point. Although the point
            # has no volume, it does contribute to e.g. the scene's
            # bounding box.
            return np.array([[0, 0, 0], [0, 0, 0]], np.float32)
        else:
            if self._aabb_rev == self.positions.rev:
                return self._aabb
            pos = self.positions.data
            aabb_2d = np.array(
                [np.nanmin(pos, axis=0), np.nanmax(pos, axis=0)], np.float32
            )
            self._aabb[1, 0] += self.font_size  # positions do not include char width
            self._aabb = np.column_stack([aabb_2d, np.zeros((2, 1), np.float32)])
            self._aabb_rev = self.positions.rev
            return self._aabb

    def _apply_layout(self):
        """The layout step. Updates positions and sizes to finalize the geometry.
        Can be overloaded for custom behavior.
        """

        # Prepare

        font_size = self._font_size
        # We try to follow CSS  which multiplies the line_height by the font_size as well
        line_height = self.line_height * font_size

        anchor = self._anchor
        text_align = self._text_align
        text_align_last = self._text_align_last
        if text_align_last == "auto":
            if text_align == "justify":
                text_align_last = "start"
            elif text_align == "justify-all":
                text_align_last = "justify"
            else:
                text_align_last = text_align

        if text_align == "justify-all":
            text_align = "justify"

        # TODO: handle Right to Left (RTL) text
        if self._direction == "ltr":
            if text_align == "start":
                text_align = "left"
            elif text_align == "end":
                text_align = "right"

            if text_align_last == "start":
                text_align_last = "left"
            elif text_align_last == "end":
                text_align_last = "right"
        elif self._direction == "rtl":
            if text_align == "end":
                text_align = "left"
            elif text_align == "start":
                text_align = "right"

            if text_align_last == "end":
                text_align_last = "left"
            elif text_align_last == "start":
                text_align_last = "right"

        positions_array = self.positions.data
        sizes_array = self.sizes.data
        is_horizontal = self._direction is None or self._direction in ("ltr", "rtl")

        # The algorightm doesn't support text alignment for ttb and btt yet
        if not is_horizontal:
            text_align = "left"
            text_align_last = "left"

        left = right = 0
        top = bottom = 0

        line_left = float("inf")
        line_right = -float("inf")
        line_top = -float("inf")
        line_bottom = float("inf")

        vertical_offset = 0

        # Resolve position and sizes

        extent_offset = 0
        lines = []
        current_line = []
        lines_aabb = []
        for item in self._glyph_items:
            if item.newline_before:
                if current_line:
                    lines.append(current_line)
                    lines_aabb.append(
                        np.array(
                            [(line_left, line_bottom, 0), (line_right, line_top, 0)],
                            np.float32,
                        )
                    )
                current_line = []
                line_left = float("inf")
                line_right = -float("inf")
                line_top = -float("inf")
                line_bottom = float("inf")
                vertical_offset -= len(item.newline_before) * line_height
                extent_offset = 0

            extent_offset += item.margin_before * font_size
            if is_horizontal:
                positions = item.positions * font_size + (
                    extent_offset,
                    vertical_offset,
                )
            else:
                positions = item.positions * font_size + (
                    vertical_offset,
                    extent_offset,
                )
            i1, i2 = item.offset, item.offset + positions.shape[0]
            positions_array[i1:i2] = positions
            # Keep a pointer to the array so we can align the text
            current_line.append(positions_array[i1:i2])

            sizes_array[i1:i2] = font_size

            # Prepare for next
            extent_offset += item.extent * font_size
            ws_margin = item.margin_after * font_size
            extent_offset += ws_margin

            # Update line extent
            if is_horizontal:
                line_left = 0
                line_right = extent_offset
                line_top = max(line_top, item.ascender * font_size + vertical_offset)
                line_bottom = min(
                    line_bottom, item.descender * font_size + vertical_offset
                )
            else:
                line_top = 0
                line_bottom = extent_offset
                line_right = max(
                    line_right, item.ascender * font_size + vertical_offset
                )
                line_left = min(line_left, item.descender * font_size + vertical_offset)

            # Update total extent
            top = max(top, line_top)
            bottom = min(bottom, line_bottom)
            right = max(right, line_right)
            left = min(left, line_left)

            if item.newline_after:
                if current_line:
                    lines.append(current_line)
                    lines_aabb.append(
                        np.array(
                            [(line_left, line_bottom, 0), (line_right, line_top, 0)],
                            np.float32,
                        )
                    )
                current_line = []
                line_left = float("inf")
                line_right = -float("inf")
                line_top = -float("inf")
                line_bottom = float("inf")
                vertical_offset -= len(item.newline_after) * line_height
                extent_offset = 0

        if current_line:
            lines.append(current_line)
            lines_aabb.append(
                np.array(
                    [(line_left, line_bottom, 0), (line_right, line_top, 0)], np.float32
                )
            )
        current_line = []
        line_left = float("inf")
        line_right = -float("inf")
        line_top = -float("inf")
        line_bottom = float("inf")

        # take care of new lines at the end of the text
        bottom = min(bottom, vertical_offset)

        self._aabb = np.array([(left, bottom, 0), (right, top, 0)], np.float32)

        # Anchoring

        anchor_offset = self.anchor_offset
        if anchor.endswith("left"):
            pos_offset_x = -left + anchor_offset
        elif anchor.endswith("right"):
            pos_offset_x = -right - anchor_offset
        else:
            pos_offset_x = -0.5 * (left + right)

        if anchor.startswith("top"):
            pos_offset_y = -top - anchor_offset
        elif anchor.startswith("middle"):
            pos_offset_y = -0.5 * (top + bottom)
        elif anchor.startswith("baseline"):
            pos_offset_y = -vertical_offset
        elif anchor.startswith("bottom"):
            pos_offset_y = -bottom + anchor_offset
        else:
            pos_offset_y = 0

        positions_array += pos_offset_x, pos_offset_y
        self._aabb += pos_offset_x, pos_offset_y, 0

        # Align the text accordingly
        total_length = right - left
        num_lines = len(lines)

        align = text_align
        for i, (line, line_aabb) in enumerate(zip(lines, lines_aabb)):
            if i == num_lines - 1:
                align = text_align_last

            line_right = line_aabb[1, 0]
            line_left = line_aabb[0, 0]
            line_length = line_right - line_left

            extra_space_per_word = 0
            length_to_add = 0
            if align == "justify":
                length_to_add = total_length - line_length
                words = len(line)
                if words > 1:
                    extra_space_per_word = length_to_add / (words - 1)
                else:
                    length_to_add = 0

            if align == "center":
                line_pos_offset_x = 0.5 * (right - left) - 0.5 * (
                    line_right - line_left + length_to_add
                )
            elif align == "right":
                line_pos_offset_x = (right - left) - (
                    line_right - line_left + length_to_add
                )
            else:  # elif align == "left":
                line_pos_offset_x = 0
            for j, positions in enumerate(line):
                positions += (line_pos_offset_x + j * extra_space_per_word), 0

        # Trigger uploads to GPU
        self.sizes.update_range(0, i2)
        self.positions.update_range(0, i2)

    def apply_layout(self):
        """Update the internal contained glyphs.

        To overload this with a custom layout, overload ``_apply_layout()``.
        """

        if self._do_layout:
            self._apply_layout()

    @property
    def font_size(self):
        """The text size.

        For text rendered in screen space (``screen_space`` property is set),
        the size is in logical pixels, and the object's local transform affects
        the final text size.

        For text rendered in world space (``screen_space`` property is *not*
        set), the size is in object coordinates, and the the object's
        world-transform affects the final text size.

        Notes
        -----
        Font size is indicative only. Final glyph size further depends on the
        font family, as glyphs may be smaller (or larger) than the indicative
        size. Final glyph size may further vary based on additional formatting
        applied a particular subsection.

        """
        return self._font_size

    @font_size.setter
    def font_size(self, value):
        self._font_size = float(value)
        self.apply_layout()

    @property
    def max_width(self):
        """The maximum width of the text. Text will wrap if beyond this
        limit. The coordinate system that this applies to is the same
        as for ``font_size``. Set to 0 for no wrap. Default 0.

        TEXT WRAPPING IS NOT YET IMPLEMENTED
        """
        return self._max_width

    @max_width.setter
    def max_width(self, width):
        self._max_width = float(width or 0)
        self.apply_layout()

    @property
    def line_height(self):
        """The relative height of a line of text, used to set the
        distance between lines. Default 1.2.
        """
        return self._line_height

    @line_height.setter
    def line_height(self, height):
        self._line_height = float(height or 1.2)
        self.apply_layout()

    @property
    def text_align(self):
        """Set the alignment of wrapped text. Can be start, end, or center.
        Default "start".

        Text alignment is ignored for top to bottom ('ttb') and
        bottom to top ('btt') directions.
        """
        return self._text_align

    @text_align.setter
    def text_align(self, align):
        if align is None:
            align = "start"
        if not isinstance(align, str):
            raise TypeError("text-align must be a None or str.")
        align = align.lower()
        if align not in _TEXT_ALIGNMENTS:
            raise ValueError(f"Align must be one of {_TEXT_ALIGNMENTS}. Got {align}.")
        self._text_align = align
        self.apply_layout()

    @property
    def text_align_last(self):
        """Set the alignment of the last line of text.
        Default "auto".

        Text alignment is ignored for top to bottom ('ttb') and
        bottom to top ('btt') directions.
        """
        return self._text_align_last

    @text_align_last.setter
    def text_align_last(self, align):
        if align is None:
            align = "start"
        if not isinstance(align, str):
            raise TypeError("text-align must be a None or str.")
        align = align.lower()
        if align not in _TEXT_ALIGNMENTS_LAST:
            raise ValueError(
                f"Align must be one of {_TEXT_ALIGNMENTS_LAST}. Got {align}"
            )
        self._text_align_last = align
        self.apply_layout()

    @property
    def anchor(self):
        """The position of the origin of the text. This is a string
        representing the vertical and horizontal anchors, separated by
        a dash, e.g. "top-left" or "bottom-center".

        * Vertical values: "top", "middle", "baseline", "bottom".
        * Horizontal values: "left", "center", "right".
        """
        return self._anchor

    @anchor.setter
    def anchor(self, anchor):
        # Init
        if anchor is None:
            anchor = "middle-center"
        elif not isinstance(anchor, str):
            raise TypeError("Text anchor must be str.")
        anchor = anchor.lower().strip()
        # Split
        if anchor.count("-") == 1:
            anchory, _, anchorx = anchor.partition("-")
        else:
            anchory = anchorx = ""
            for key, val in ANCHOR_Y_ALTS.items():
                if anchor.startswith(key):
                    anchory = val
                    break
            for key, val in ANCHOR_X_ALTS.items():
                if anchor.endswith(key):
                    anchorx = val
                    break
        # Resolve
        try:
            anchory = ANCHOR_Y_ALTS[anchory]
            anchorx = ANCHOR_X_ALTS[anchorx]
        except KeyError:
            raise ValueError(f"Invalid anchor value '{anchor}'")
        # Apply
        self._anchor = f"{anchory}-{anchorx}"
        self.apply_layout()

    @property
    def anchor_offset(self):
        """The offset (extra margin) for the 'top', 'bottom', 'left', and 'right' anchors."""
        return self._anchor_offset

    @anchor_offset.setter
    def anchor_offset(self, value):
        self._anchor_offset = float(value)
        self.apply_layout()
