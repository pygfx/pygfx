"""
This module implements text geometry. This is where the text rendering comes
together. Most steps in the text rendering process come from pygfx.utils.text,
though most of the alignment is implemented here.

For details about the text rendering process, see pygfx/utils/text/README.md
"""

# TODO: check below explanation at the end. I wrote this when picking this pr up again
# Text is divided into multiple blocks, which represents lines/paragraphs, and which can
# be individually positioned, e.g. to be used as labels.
# Each TextBlock contains multiple TextItem's, which represent groups of characters that stick
# together,during laytout i.e. words in most cases.
# The geomety maintains per-block arrays/buffers for position (and size?). It also has arrays
# to store the per-glyph data in the text items.

import numpy as np

from ..resources import Buffer
from ..utils import text as textmodule
from ..utils.enums import TextAlign, TextAnchor
from ..utils import logger
from ._base import Geometry


# We cache the extents of small whitespace strings to improve performance
WHITESPACE_EXTENTS = {}


class TextEngine:
    def select_font(self, text, font_props):
        """The font selection step. Returns (text, font_filename) tuples.
        Can be overloaded for custom behavior.
        """
        return textmodule.select_font(text, font_props)

    def shape_text(self, text, font_filename, direction):
        """The shaping step. Returns (glyph_indices, positions, meta).
        Can be overloaded for custom behavior.
        """
        return textmodule.shape_text(text, font_filename, direction)

    def generate_glyph(self, glyph_indices, font_filename):
        """The glyph generation step. Returns an array with atlas indices.
        Can be overloaded for custom behavior.
        """
        return textmodule.generate_glyph(glyph_indices, font_filename)

    def get_ws_extent(self, ws, font):
        """Get the extent of a piece of whitespace text. Results of small strings are cached."""
        direction = "ltr"
        map = WHITESPACE_EXTENTS.setdefault(font.filename, {})
        extent = 0
        for c in ws:
            try:
                extent += map[c]
            except KeyError:
                meta = self.shape_text(c, font.filename, direction)[2]
                map[c] = meta["extent"]
                extent += map[c]
        return extent


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
    font_size : float
        The size of the font, in object coordinates or pixel screen coordinates,
        depending on the value of the ``screen_space`` property. Default 12.
    family : str, tuple
        The name(s) of the font to prefer. If multiple names are given, they are
        preferred in the given order. Characters that are not supported by any
        of the given fonts are rendered with the default font (from the Noto
        Sans collection).
    direction : str
        The text direction. By default the text direction is determined
        automatically, but is always horizontal. Can be set to 'lrt', 'rtl',
        'ttb' or 'btt'.
    space_mode : enums.CoordSpace
        The coordinate space in which the text is rendered. Can be "screen" or "model", default "screen".
    position_mode : str
        TODO: maybe one enum space-mode: screen, model, label
        How ``TextBlock`` objects are positioned. With "auto" the layout is performed automatically (in the same space as ``space``).
        If "model" the positioning is done in model-space, a bit as if its a point set with labels (i.e. TextBlock's) as markers.
    anchor : str | TextAnchor
        The position of the origin of the text. Default "middle-center".
    anchor_offset : float
        The offset (extra margin) for the 'top', 'bottom', 'left', and 'right' anchors.
    max_width : float
        The maximum width of the text. Words are wrapped if necessary. A value
        of zero means no wrapping. Default zero.
    line_height : float
        A factor to scale the distance between lines. A value of 1 means the
        "native" font's line distance. Default 1.2.
    paragraph_spacing : float
        An extra space between paragraphs. Default 0.
    text_align : str | TextAlign
        The horizontal alignment of the inline-level content. Can be "start",
        "end", "left", "right", "center", "justify" or "justify_all". Default
        "left". Text alignment is ignored for top to bottom ('ttb') and
        bottom to top ('btt') directions.
    text_align_last: str | TextAlign
        The horizontal alignment of the last line of the content
        element. Can be "start", "end", "left", "right", "center", "justify" or
        "auto". Default "auto". Text alignment is ignored for top to
        bottom ('ttb') and bottom to top ('btt') directions.
    """

    _text_engine = TextEngine()

    def __init__(
        self,
        text=None,
        *,
        markdown=None,
        font_size=12,
        family=None,
        direction=None,
        screen_space=None,
        space_mode="model",
        anchor="middle-center",
        anchor_offset=0,
        max_width=0,
        line_height=1.2,
        paragraph_spacing=0,
        text_align="start",
        text_align_last="auto",
    ):
        super().__init__()

        # --- check text input

        text_and_markdown = [i for i in (text, markdown) if i is not None]
        if len(text_and_markdown) > 1:
            raise TypeError("Either text or markdown must be given, not both.")

        # --- create per-item arrays/buffers

        # The position of each text block
        self.positions = Buffer(np.zeros((8, 3), np.float32))
        # The size of each text block
        # TODO: remove or rename to font_sizes
        self.sizes = Buffer(np.zeros((8,), np.float32))
        # self.colors = None

        # --- create per-glyph arrays/buffers

        # Index into the atlas that contains all glyphs
        self.glyph_atlas_indices = Buffer(np.zeros((16,), np.uint32))
        # Index into the block list above (i.e. the block.index)
        # TODO: I don't think we need this???
        self.glyph_block_indices = Buffer(np.zeros((16,), np.uint32))
        # Sub-position for glyph size, shaping, kerning, etc.
        self.glyph_positions = Buffer(np.zeros((16, 2), np.float32))
        self.glyph_sizes = Buffer(np.zeros((16,), np.float32))

        # --- init variables to help manage the glyph arrays

        # The number of allocated glyph slots.
        # This must be equal to _glyph_indices_top - _glyph_indices_gaps
        self._glyph_count = 0
        # The index marking the maximum used in the arrays. All elements higher than _glyph_indices_top are free.
        self._glyph_indices_top = 0
        # Free slots that are not in the contiguous space at the end of the arrays.
        self._glyph_indices_gaps = set()

        # Track what blocks need an update. This set is shared with the TextBlock instances.
        self._text_blocks = []  # List of TextBlock instances
        self._dirty_blocks = set()  # Set of ints (text_block.index)

        # --- other geomery-specific things

        self._aabb = np.zeros((2, 3), np.float32)

        # --- set propss

        # Font props
        self.font_size = font_size
        self.family = family
        self.direction = direction

        # Space props
        # TODO: fix reference to screen_space, and usage in examples
        if screen_space is not None:
            raise DeprecationWarning(
                "TextGeometry.screen_space is deprecated, use space_mode instead."
            )
        self.space_mode = space_mode

        # Layout props
        self.anchor = anchor
        self.anchor_offset = anchor_offset
        self.max_width = max_width
        self.line_height = line_height
        self.paragraph_spacing = paragraph_spacing
        self.text_align = text_align
        self.text_align_last = text_align_last

        # --- set initial content

        if text is not None:
            self.set_text(text)
        elif markdown is not None:
            self.set_markdown(markdown)
        else:
            pass  # TODO: ??

    # --- font properties

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
        # TODO: render_glyphs or only layout?
        self._trigger_blocks_update(layout=True)

    @property
    def family(self):
        """The font family to use.

        The name(s) of the font to prefer. If multiple names are given, they are
        preferred in the given order. Characters that are not supported by any
        of the given fonts are rendered with the default font (from the Noto
        Sans collection).
        """
        return self._font_props.family

    @family.setter
    def family(self, family):
        if family is None:
            self._font_props = textmodule.FontProps()
        else:
            self._font_props = textmodule.FontProps(family)
        self._trigger_blocks_update(render_glyphs=True)

    @property
    def direction(self):
        """The font direction overload."""
        return self._direction

    @direction.setter
    def direction(self, direction):
        if direction is None:
            self._direction = None
        elif direction in ("ltr", "rtl", "ttb", "btt"):
            self._direction = str(direction)
        else:
            raise ValueError(
                "TextGeometry direction must be None, 'ltr', 'rtl', 'ttb', or 'btt'."
            )
        self._trigger_blocks_update(render_glyphs=True)

    @property
    def space_mode(self):
        """The mode to render in ("screen" vs "model").

        Notes
        -----
        Regardless of choice, the local object's rotation and scale will still
        transform the text.

        """
        return self._store.space_mode

    @space_mode.setter
    def space_mode(self, value):
        # TODO: check input value, use a new enum
        self._store.space_mode = str(value)
        self._trigger_blocks_update(layout=True)

    # --- layout properties

    @property
    def anchor(self):
        """The position of the origin of the text.

        Represented as a string representing the vertical and horizontal anchors,
        separated by a dash, e.g. "top-left" or "bottom-center".

        * Vertical values: "top", "middle", "baseline", "bottom".
        * Horizontal values: "left", "center", "right".

        See :obj:`pygfx.utils.enums.TextAlign`:
        """
        return self._anchor

    @anchor.setter
    def anchor(self, anchor):
        # Init
        if anchor is None:
            anchor = "middle-center"
        elif not isinstance(anchor, str):
            raise TypeError("Text anchor must be str.")
        anchor = anchor.lower().strip().replace("-", "_")
        if anchor not in TextAnchor.__fields__:
            raise ValueError(f"Text anchor must be one of {TextAnchor}. Got {anchor!r}")
        self._anchor = TextAnchor[anchor]
        self._trigger_blocks_update(layout=True)

    @property
    def anchor_offset(self):
        """The offset (extra margin) for the 'top', 'bottom', 'left', and 'right' anchors."""
        return self._anchor_offset

    @anchor_offset.setter
    def anchor_offset(self, value):
        self._anchor_offset = float(value)
        self._trigger_blocks_update(layout=True)

    @property
    def max_width(self):
        """The maximum width of the text. Text will wrap if beyond this
        limit. The coordinate system that this applies to is the same
        as for ``font_size``. Set to 0 for no wrap. Default 0.

        TEXT WRAPPING IS NOT YET IMPLEMENTED
        TODO: wel toch?
        """
        return self._max_width

    @max_width.setter
    def max_width(self, width):
        self._max_width = float(width or 0)
        self._trigger_blocks_update(layout=True)

    @property
    def line_height(self):
        """The relative height of a line of text, used to set the
        distance between lines. Default 1.2.
        """
        return self._line_height

    @line_height.setter
    def line_height(self, height):
        self._line_height = float(height or 1.2)
        self._trigger_blocks_update(layout=True)

    @property
    def paragraph_spacing(self):
        """The extra space between two paragraphs.

        Measured in text units (like line height and font size).
        """
        return self._paragraph_spacing

    @paragraph_spacing.setter
    def paragraph_spacing(self, paragraph_spacing):
        self._paragraph_spacing = float(paragraph_spacing or 0)
        self._trigger_blocks_update(layout=True)

    @property
    def text_align(self):
        """Set the alignment of wrapped text. Default 'start'.

        See :obj:`pygfx.utils.enums.TextAlign`:

        Text alignment is ignored for top to bottom ('ttb') and
        bottom to top ('btt') directions.
        """
        return self._text_align

    @text_align.setter
    def text_align(self, align):
        if align is None:
            align = "start"
        if not isinstance(align, str):
            raise TypeError("Text align must be a None or str.")
        align = align.lower().replace("-", "_")
        if align not in TextAlign.__fields__:
            raise ValueError(f"Text align must be one of {TextAlign}. Got {align!r}.")
        if align == "auto":
            align = "left"
        self._text_align = TextAlign[align]
        self._trigger_blocks_update(layout=True)

    @property
    def text_align_last(self):
        """Set the alignment of the last line of text. Default "auto".

        See :obj:`pygfx.utils.enums.TextAlign`:

        Text alignment is ignored for top to bottom ('ttb') and
        bottom to top ('btt') directions.
        """
        return self._text_align_last

    @text_align_last.setter
    def text_align_last(self, align):
        if align is None:
            align = "auto"
        if not isinstance(align, str):
            raise TypeError("Text align_last must be a None or str.")
        align = align.lower().replace("-", "_")
        if align not in TextAlign.__fields__:
            raise ValueError(
                f"Text align_last must be one of {TextAlign}. Got {align!r}"
            )
        self._text_align_last = TextAlign[align]
        self._trigger_blocks_update(layout=True)

    # --- public methods

    # TODO: refine this API, maybe ensure_block_count() or something is sufficient

    def create_text_block(self):
        self._allocate_text_blocks(1)
        return self._text_blocks[-1]

    def create_text_blocks(self, n):
        self._allocate_text_blocks(n)
        return self._text_blocks[-n:]

    def get_text_block(self, index):
        """Get the TextBlock instance at the given index."""
        return self._text_blocks[index]

    def set_text(self, text):
        """Set the full text fir this TextGeometry.

        Each line (i.e. paragraph) results in one TextBlock.
        """
        if not isinstance(text, str):
            raise TypeError("The text should be str.")
        lines = text.splitlines()
        self._ensure_text_block_count(len(lines))
        for i, line in enumerate(lines):
            block = self._text_blocks[i]
            # Note that setting the blocks text is fast if it did not change
            block.set_text(line)

        # Disable unused text blocks
        for i in range(len(lines), len(self._text_blocks)):
            block = self._text_blocks[i]
            block.set_text("")

        # TODO: trigger a layout
        self._on_update_object()

    def set_markdown(self, text):
        self.set_text(text)

    def xxxset_markdown(self, markdown, family=None):
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
                    items[-1].ws_after += piece
            else:
                items.append(LayoutTextItem(piece, pieces_props[i]))
        if items and pending_whitespace:
            items[0].ws_before += pending_whitespace

        self._set_layout_items(items)
        return self

    # TODO: can be removed, I think
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

    def get_bounding_box(self):
        space_mode = self._store["space_mode"]
        if space_mode == "screen":
            # There is no sensible bounding box for text in screen space, except
            # for the anchor point. Although the point has no volume, it does
            # contribute to e.g. the scene's bounding box.
            return np.zeros((2, 3), np.float32)
        elif space_mode == "model":
            # A bounding box makes sense, and we calculated it during layout,
            # because we're already shifting rects there.
            return self._aabb
        elif space_mode == "labels":
            if self._aabb_rev == self.positions.rev:
                return self._aabb
            aabb = None
            # Get positions and check expected shape
            # TODO: only use positions that are in use!
            positions = self.positions.data
            aabb = np.array([positions.min(axis=0), positions.max(axis=0)], np.float32)
            # If positions contains xy, but not z, assume z=0
            if aabb.shape[1] == 2:
                aabb = np.column_stack([aabb, np.zeros((2, 1), np.float32)])
            self._aabb = aabb
            self._aabb_rev = self.positions.rev
            return self._aabb
        else:
            logger.warning(f"Unexpected space_mode {space_mode!r}")
            return None

    def get_bounding_sphere(self):
        space_mode = self._store["space_mode"]
        if space_mode == "screen":
            # There is no sensible bounding box for text in screen space, except
            # for the anchor point. Although the point has no volume, it does
            # contribute to e.g. the scene's bounding box.
            return np.zeros((4,), np.float32)
        elif space_mode == "model":
            # A bounding box makes sense, we can calculate it from the rect.
            mean = 0.5 * (self._aabb[1] + self._aabb[0])
            diag = np.norm(self._aabb[1] - self._aabb[0])
            return np.array([[mean[0], mean[1], mean[2], diag]], np.float32)
        elif space_mode == "labels":
            return super().get_bounding_sphere()
        else:
            logger.warning(f"Unexpected space_mode {space_mode!r}")
            return None

    # --- private methods

    def _on_update_object(self):
        # Is called right before the object is drawn;
        # gets called by Text._update_object()

        dirty_blocks = self._dirty_blocks
        if not dirty_blocks:
            return  # early exit

        # Update blocks
        need_high_level_layout = False
        for index in dirty_blocks:
            block = self._text_blocks[index]
            did_block_layout = block._update(self)
            need_high_level_layout |= did_block_layout

        # Reset
        dirty_blocks.clear()

        # Higher-level layout
        if need_high_level_layout:
            if self.space_mode in ("screen", "model"):
                apply_final_layout(self)

    # --- block management

    def _ensure_text_block_count(self, n):
        """Allocate new buffer if necessary."""
        current_size = len(self._text_blocks)
        if current_size < n or current_size > 4 * n:
            new_size = 2 ** int(np.ceil(np.log2(n)))
            new_size = max(8, new_size)
            self._allocate_text_blocks(new_size)

    def _allocate_text_blocks(self, n):
        """Allocate new buffers for text blocks with the given size."""
        smallest_n = min(n, len(self._text_blocks))
        # Create new buffers
        new_positions = np.zeros((n, 3), np.float32)
        new_sizes = np.zeros((n,), np.float32)
        # Copy data
        new_positions[:smallest_n] = self.positions.data[:smallest_n]
        new_sizes[:smallest_n] = self.sizes.data[:smallest_n]
        # Assign
        # TODO: I feel like resetting these buffers should be done on the geometry
        self.positions = Buffer(new_positions)
        self.sizes = Buffer(new_sizes)

        # Allocate / de-allocate text blocks and their glyphs
        while len(self._text_blocks) > n:
            block = self._text_blocks.pop()
            self._deallocate_glyphs(block.indices)
        while len(self._text_blocks) < n:
            block = TextBlock(len(self._text_blocks), self._dirty_blocks)
            self._text_blocks.append(block)

    def _trigger_blocks_update(self, layout=False, render_glyphs=False):
        for block in self._text_blocks:
            block._mark_dirty(layout=layout, render_glyphs=render_glyphs)

    # --- glyph array management

    def _glyphs_allocate(self, n):
        max_glyph_slots = self.glyph_positions.nitems

        # Need larger buffer?
        n_free = max_glyph_slots - self._glyph_count
        if n > n_free:
            self._glyphs_create_new_buffers(n)

        # Allocate indices
        if not self._glyph_indices_gaps:
            # Contiguous: indices is a range
            assert self._glyph_indices_top == self._glyph_count
            indices = range(self._glyph_indices_top, self._glyph_indices_top + n)
            self._glyph_count += n
            self._glyph_indices_top += n
        else:
            # First use gaps ...
            indices = np.empty((n,), np.uint32)
            n_from_gap = min(n, len(self._glyph_indices_gaps))
            for i in range(n_from_gap):
                indices[i] = self._glyph_indices_gaps.pop()
            self._glyph_count += n_from_gap
            # Then use indices at the end
            n -= n_from_gap
            if n > 0:
                indices[n_from_gap:] = range(
                    self._glyph_indices_top, self._glyph_indices_top + n
                )
                self._glyph_count += n
                self._glyph_indices_top += n

        return indices

    def _glyphs_deallocate(self, indices):
        # Nullify data
        self.glyph_block_indices[indices] = -1
        self.glyph_atlas_indices[indices] = 0
        # self.glyph_positions[indices] = 0.0
        # self.glyph_sizes[indices] = 0.0
        # Deallocate
        self._glyph_indices_gaps.update(indices)
        self._glyph_count -= len(indices)
        # TODO: Reduce buffer sizes from the Text object, re-packing all items
        # # Maybe reduce buffer size
        # max_glyph_slots = self.glyph_positions.nitems
        # if self._glyph_count < 0.25 * max_glyph_slots:
        #     self._glyphs_create_new_buffers()

    def _glyphs_create_new_buffers(self, extra_needed=0):
        assert extra_needed >= 0

        # Get new size
        need_size = self._glyph_indices_top + extra_needed
        new_size = 2 ** int(np.ceil(np.log2(need_size)))
        new_size = max(16, new_size)

        # Prepare new arrays
        glyph_block_indices = np.zeros((new_size,), np.uint32)
        glyph_atlas_indices = np.zeros((new_size,), np.uint32)
        glyph_positions = np.zeros((new_size, 2), np.float32)
        glyph_sizes = np.zeros((new_size,), np.float32)

        # Copy data over
        n = self._glyph_indices_top
        glyph_block_indices[:n] = self.glyph_block_indices.data[:n]
        glyph_atlas_indices[:n] = self.glyph_atlas_indices.data[:n]
        glyph_positions[:n] = self.glyph_positions.data[:n]
        glyph_sizes[:n] = self.glyph_sizes.data[:n]

        # Store
        self.glyph_block_indices = Buffer(glyph_block_indices)
        self.glyph_atlas_indices = Buffer(glyph_atlas_indices)
        self.glyph_positions = Buffer(glyph_positions)
        self.glyph_sizes = Buffer(glyph_sizes)


class TextBlock:
    """The TextBlock represents one block or paragraph of text.

    Text blocks are positioned using an entry in geometry.positions. This allows using it for e.g. a collection of text labels.

    """

    # TODO: __slots__ = []

    def __init__(self, index, dirty_blocks):
        self._index = index  # e.g. the index in geometry.positions
        self._dirty_blocks = dirty_blocks  # a set from the geometry

        self._text = ""
        self._need_layout = False
        self._need_render_glyphs = False

        self._text_items = []
        self._old_text_items = []

        # Used by layout
        self._nlines = 0
        self._rect = Rect()

    @property
    def index(self):
        """The index in the geometry.positions buffer."""
        return self._index

    def _mark_dirty(self, *, layout=False, render_glyphs=False):
        # Trigger ._update() being called right before the next draw
        self._dirty_blocks.add(self._index)
        self._need_layout |= bool(layout) | bool(render_glyphs)
        self._need_render_glyphs |= bool(render_glyphs)

    def _update(self, geometry):
        """Do the work to bring this block up-to-date. Be fast!"""

        # Reset flags
        # self._dirty_blocks.discard(self._index)  # no, geometry calls clear
        need_render_glyphs = self._need_render_glyphs
        need_layout = self._need_layout
        self._need_render_glyphs = False
        self._need_layout = False

        # De-allocate old item objects
        for item in self._old_text_items:
            item.sync_with_geometry(geometry, self._index)
        self._old_text_items = []

        # Update in-use item objects
        for item in self._text_items:
            if need_render_glyphs or item.need_render_glyphs:
                item.render_glyphs(geometry)

        # Layout
        if need_layout:
            apply_block_layout(geometry, self)

        # Item updates, and layout, may require syncing glyph data
        for item in self._text_items:
            if item.need_sync_with_geometry:
                item.sync_with_geometry(geometry, self._index)

        return need_layout  # i.e. did_layout

    def set_text(self, text):
        """Set the text for this TextBlock.

        This is called from ``TextGeometry.set_text()``, but can also be called directly.
        """
        if not isinstance(text, str):
            raise TypeError("TextBlock text should be str.")
        if text == self._text:
            return
        self._text = text
        self._mark_dirty(layout=True)

        def new_item(text, ws_before):
            if self._text_items:
                item = self._text_items.pop(0)
            else:
                item = TextItem()
            item.set_text(text)
            item.ws_before = ws_before
            return item

        # Split text in words
        items = []
        pending_whitespace = ""
        for kind, piece in textmodule.tokenize_text(text):
            if kind == "ws":
                pending_whitespace += piece
            elif kind == "nl":
                if pending_whitespace:
                    items.append(new_item("", pending_whitespace))
                    pending_whitespace = ""
                elif not items:
                    items.append(new_item("", ""))
                items[-1].nl_after += piece
            else:
                items.append(new_item(piece, pending_whitespace))
                pending_whitespace = ""
        if pending_whitespace:
            items.append(new_item("", pending_whitespace))

        # Store old items that need to be de-allocated
        for item in self._text_items:
            if len(item.glyph_indices):
                item.set_text("")
                self._old_text_items.append(item)

        # Store new worlds
        self._text_items = items

    def set_markdown(self, text):
        raise NotImplementedError()


class TextItem:
    """Represents one unit of text that moves as a whole to a new line when wrapped.
    This is an internal object (not public).
    """

    __slots__ = [
        "ascender",
        "atlas_indices",
        "descender",
        "direction",
        "extent",
        "glyph_count",
        "glyph_indices",
        "layout_offset",
        "margin_before",
        "need_render_glyphs",
        "need_sync_with_geometry",
        "nl_after",
        "offset",
        "positions",
        "sizes",
        "text",
        "ws_before",
    ]

    def __init__(self):
        # The text defines the arrays
        self.text = None

        # Whitespace attributes affect layout
        self.ws_before = ""
        self.nl_after = ""
        self.margin_before = 0  # is ws_before expressed as a float (in font units)

        # Flags to control precise updates
        self.need_render_glyphs = False
        self.need_sync_with_geometry = False

        # The text item has its own per-glyph arrays. These are copied into the geometries buffer arrays.
        self.atlas_indices = None
        self.positions = None
        self.sizes = None
        self.glyph_count = 0

        # The indices for slots in the arrays at the geometry. This value is managed by the geometry.
        self.glyph_indices = range(0)

        # Transform info when copying to he geometry buffers. Set during layout.
        self.offset = (1.0, 0.0, 0.0)
        self.layout_offset = None  # used by layout as a temp var

        # Metadata
        self.extent = 0
        self.ascender = 0
        self.descender = 0
        self.direction = None

    def set_text(self, text):
        if text != self.text:
            self.text = text
            self.need_render_glyphs = True
            self.margin_before = 0

    def set_offset(self, scale, dx, dy):
        offset = scale, dx, dy
        if offset != self.offset:
            self.offset = offset
            self.need_sync_with_geometry = True

    def render_glyphs(self, geometry):
        """Update the item's arrays."""

        self.need_render_glyphs = False
        self.need_sync_with_geometry = True

        font_props = geometry._font_props
        textengine = geometry._text_engine

        self.margin_before = 0
        if self.ws_before:
            for ws, font in textengine.select_font(self.ws_before, font_props):
                self.margin_before += textengine.get_ws_extent(ws, font)

        if not self.text:
            self.atlas_indices = None
            self.positions = None
            self.sizes = None
            self.glyph_count = 0
            return

        # Prepare containers for array
        atlas_indices_list = []
        positions_list = []
        sizes_list = []

        # Init meta data
        extent = ascender = descender = 0
        direction = "ltr"

        # Text rendering steps: font selection, shaping, glyph generation
        last_reverse_index = 0
        text_pieces = textengine.select_font(self.text, font_props)
        for text, font in text_pieces:
            unicode_indices, positions, meta = textengine.shape_text(
                text, font.filename, direction
            )
            atlas_indices = textengine.generate_glyph(unicode_indices, font.filename)
            encode_font_props_in_atlas_indices(atlas_indices, font_props, font)
            sizes = np.full((positions.shape[0],), 1.0, np.float32)
            extent = extent + meta["extent"]
            ascender = max(ascender, meta["ascender"])
            descender = min(descender, meta["descender"])  # note: descender is negative
            direction = meta["direction"]  # use last

            # TODO: if we have multiple pieces they need to be offset with the extent

            # Put in list, take direction into account
            if direction in ("rtl", "btt"):
                atlas_indices_list.insert(last_reverse_index, atlas_indices)
                positions_list.insert(last_reverse_index, positions)
                sizes_list.insert(last_reverse_index, sizes)
            else:
                atlas_indices_list.append(atlas_indices)
                positions_list.append(positions)
                sizes_list.append(sizes)
                last_reverse_index = len(atlas_indices_list)

        # Store meta data on the item
        self.extent = extent
        self.ascender = ascender
        self.descender = descender
        self.direction = direction

        # Store as a single array
        if len(atlas_indices_list) == 0:
            self.atlas_indices = None
            self.positions = None
            self.sizes = None
            self.glyph_count = 0
        elif len(atlas_indices_list) == 1:
            self.atlas_indices = atlas_indices_list[0]
            self.positions = positions_list[0]
            self.sizes = sizes_list[0]
            self.glyph_count = len(self.positions)
        else:
            self.atlas_indices = np.concatenate(atlas_indices_list, axis=0)
            self.positions = np.concatenate(positions_list, axis=0)
            self.sizes = np.concatenate(sizes_list, axis=0)
            self.glyph_count = len(self.positions)

    def sync_with_geometry(self, geometry, block_index):
        """Sync the item's arrays into the geometries buffers."""

        self.need_sync_with_geometry = False

        if self.glyph_count != len(self.glyph_indices):
            self._allocate_indices(geometry)
        if self.glyph_count > 0:
            self._sync_data(geometry, block_index)

    def _allocate_indices(self, geometry):
        glyph_count = self.glyph_count
        glyph_indices = self.glyph_indices

        current_glyph_indices_count = len(glyph_indices)
        if glyph_count < current_glyph_indices_count:
            new_indices = glyph_indices[:glyph_count].copy()
            indices_to_free = glyph_indices[glyph_count:]
            geometry._glyphs_deallocate(indices_to_free)
            self.glyph_indices = new_indices
        elif glyph_count > current_glyph_indices_count:
            extra_indices = geometry._glyphs_allocate(
                glyph_count - current_glyph_indices_count
            )
            new_indices = np.empty((glyph_count,), np.uint32)
            new_indices[:current_glyph_indices_count] = glyph_indices
            new_indices[current_glyph_indices_count:] = extra_indices
            self.glyph_indices = new_indices

    def _sync_data(self, geometry, block_index):
        indices = self.glyph_indices

        # Make the positioning absolute
        scale, dx, dy = self.offset
        positions = self.positions * scale + (dx, dy)
        sizes = self.sizes * scale

        # Write glyph data
        geometry.glyph_block_indices.data[indices] = block_index
        geometry.glyph_atlas_indices.data[indices] = self.atlas_indices
        geometry.glyph_positions.data[indices] = positions
        geometry.glyph_sizes.data[indices] = sizes

        # Trigger sync
        if isinstance(indices, range):
            count = indices.stop - indices.start
            geometry.glyph_block_indices.update_range(indices.start, count)
            geometry.glyph_atlas_indices.update_range(indices.start, count)
            geometry.glyph_positions.update_range(indices.start, count)
            geometry.glyph_sizes.update_range(indices.start, count)
        else:
            geometry.glyph_block_indices.update_indices(indices)
            geometry.glyph_atlas_indices.update_indices(indices)
            geometry.glyph_positions.update_indices(indices)
            geometry.glyph_sizes.update_indices(indices)


def encode_font_props_in_atlas_indices(atlas_indices, font_props, font):
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


class Rect:
    __slots__ = ["bottom", "left", "right", "top"]

    def __init__(self, left=0.0, right=0.0, top=0.0, bottom=0.0):
        self.left = left
        self.right = right
        self.top = top
        self.bottom = bottom

    def __repr__(self):
        return f"<Rect({self.left:0.5g}, {self.right:0.5g}, {self.top:0.5g}, {self.bottom:0.5g})>"

    @property
    def width(self):
        return self.right - self.left

    def shift(self, dx, dy):
        self.left = self.left + dx
        self.right = self.right + dx
        self.top = self.top + dy
        self.bottom = self.bottom + dy

    def get_offset_for_anchor(self, anchor, anchor_offset):
        v_anchor, h_anchor = anchor.split("-")

        if h_anchor == "left":
            dx = -self.left + anchor_offset
        elif h_anchor == "right":
            dx = -self.right - anchor_offset
        else:  # center or justify
            dx = -0.5 * (self.left + self.right)

        if v_anchor == "top":
            dy = -self.top - anchor_offset
        elif v_anchor == "baseline":
            dy = -anchor_offset
        elif v_anchor == "bottom":
            dy = -self.bottom + anchor_offset
        else:  # middle
            dy = -0.5 * (self.top + self.bottom)

        return dx, dy


def apply_block_layout(geometry, text_block):
    """The layout step. Updates positions and sizes to finalize the geometry."""

    # TODO: move to geometry, so it can be overloaded, see text_snake example.
    # TODO: maybe move inplementation to utils.text.layout because its so long ...

    items = text_block._text_items

    if not items:
        text_block._nlines = 0
        text_block._rect = Rect()
        return

    # Obtain layout attributes
    font_size = geometry._font_size
    line_height = geometry._line_height * font_size  # like CSS
    paragraph_spacing = geometry._paragraph_spacing * font_size
    max_width = geometry._max_width
    anchor = geometry._anchor
    text_align = geometry._text_align
    text_align_last = geometry._text_align_last
    direction = geometry._direction or "ltr"
    anchor_offset = geometry._anchor_offset

    geometrty_does_layout = geometry.space_mode in ("screen", "model", "world")

    # Resolve text_align_last
    if text_align_last == "auto":
        if text_align == "justify":
            text_align_last = "start"
        elif text_align == "justify_all":
            text_align_last = "justify"
        else:
            text_align_last = text_align
    if text_align == "justify_all":
        text_align = "justify"

    # Resolve text align to real directions
    is_horizontal = direction is None or direction in ("ltr", "rtl")
    if is_horizontal:
        if direction == "ltr":
            map = {"start": "left", "end": "right"}
        elif direction == "rtl":
            map = {"start": "right", "end": "left"}
        text_align = map.get(text_align, text_align)
        text_align_last = map.get(text_align_last, text_align_last)
    else:
        # The algorightm doesn't support text alignment for ttb and btt yet
        text_align = "left"
        text_align_last = "left"

    assert text_align in ("left", "right", "center", "justify")
    assert text_align_last in ("left", "right", "center", "justify")

    # Prepare

    # The offset is used to track the position as we wrap the text
    offset = [0, 0]

    # The current rect represents the bounding box of each line, relative to the line.
    # Vertically, point zero is at the baseline, and the rect spans from ascender (top, positive) to descender (bottom, negative).
    current_rect = Rect()  # left, right, top, bottom

    # The current line holds the text items for the line being processed.
    current_line = []

    lines = []  # list of lists of TextItems
    rects = []  # List of Rects

    def make_new_line(n_new_lines=1, n_new_paragraphs=0):
        nonlocal current_line, current_rect
        skip = n_new_lines * line_height + n_new_paragraphs * paragraph_spacing
        if is_horizontal:
            offset[1] -= skip
            offset[0] = 0
        else:
            offset[1] = 0
            offset[0] += skip
        if current_line:
            lines.append(current_line)
            rects.append(current_rect)
            current_line = []
            current_rect = Rect()

    # Resolve position and sizes

    for item in items:
        # Get item width and determine if we need a new line
        apply_margin = True
        if max_width > 0 and current_line:
            item_width = (item.margin_before + item.extent) * font_size
            if offset[0] + item_width > max_width:
                make_new_line()
                apply_margin = False

        # Apply whitespace offset
        if apply_margin:
            offset[0] += item.margin_before * font_size

        # Add item and store its initial offset, which we use later on
        current_line.append(item)
        if is_horizontal:
            item.layout_offset = tuple(offset)
        else:
            item.layout_offset = tuple(offset[::-1])

        # Prepare for next
        offset[0] += item.extent * font_size

        # Update rect
        if is_horizontal:
            current_rect.left = 0
            current_rect.right = offset[0]
            current_rect.top = max(current_rect.top, item.ascender * font_size)
            current_rect.bottom = min(current_rect.bottom, item.descender * font_size)
        else:
            current_rect.top = 0
            current_rect.bottom = offset[0]
            current_rect.right = max(current_rect.right, item.ascender * font_size)
            current_rect.left = min(current_rect.left, item.descender * font_size)

        # The item can have newlines too. Does not happen when using geometry.set_text(),
        # but can happen when using TextBlock.set_text().
        if item.nl_after:
            make_new_line(len(item.nl_after), 1)

    if current_line:
        make_new_line()

    # # If there's just one line ... its the last
    # if len(lines) == 1:
    #     text_align = text_align_last

    # Calculate block rect. The top is positive, the bottom is negative (descender).

    block_rect = Rect()
    for rect in rects:
        block_rect.left = min(block_rect.left, rect.left)
        block_rect.right = max(block_rect.right, rect.right)
    block_rect.top = rects[0].top
    block_rect.bottom = (len(rects) - 1) * line_height + rects[-1].bottom

    if text_align == "justify" or text_align_last == "justify":
        block_rect.right = max_width

    # Resolve newlines at the end of the text
    # block_rect.bottom = min(block_rect.bottom, offset[1])

    # Determine horizontal anchor

    if geometrty_does_layout:
        # If the geometry does its layout, it's far easier to *not* to the anchoring here,
        # except to anchor according to text alignment.
        anchor_offset_x, anchor_offset_y = block_rect.get_offset_for_anchor(
            f"baseline-{text_align}", 0
        )
    else:
        # Full layout done here, including anchoring.
        anchor_offset_x, anchor_offset_y = block_rect.get_offset_for_anchor(
            anchor, anchor_offset
        )

    # Shift block rect for anchoring
    block_rect.shift(anchor_offset_x, anchor_offset_y)

    # Align the text, i.e. shift individual lines so they fit inside the block rect according to the current alignment

    num_lines = len(lines)
    align = text_align
    for i, (line, rect) in enumerate(zip(lines, rects)):
        if i == num_lines - 1:
            align = text_align_last

        line_length = rect.right - rect.left

        extra_space_per_word = 0
        length_to_add = 0
        if align == "justify":
            length_to_add = max_width - line_length
            nwords = len(line)
            if nwords > 1:
                extra_space_per_word = length_to_add / (nwords - 1)
            else:
                length_to_add = 0
        if align == "center":
            line_offset_x = 0.5 * (block_rect.right - block_rect.left) - 0.5 * (
                rect.right - rect.left + length_to_add
            )
        elif align == "right":
            line_offset_x = (block_rect.right - block_rect.left) - (
                rect.right - rect.left + length_to_add
            )
        else:  # elif align == "left":
            line_offset_x = 0

        for j, item in enumerate(line):
            dx = (
                item.layout_offset[0]
                + anchor_offset_x
                + line_offset_x
                + j * extra_space_per_word
            )
            dy = item.layout_offset[1] + anchor_offset_y
            item.set_offset(font_size, dx, dy)

    # Update block's rect. Used by the final layout and to calculate bounding boxes.
    text_block._rect = block_rect
    text_block._nlines = len(rects)


def apply_final_layout(geometry):
    text_blocks = [block for block in geometry._text_blocks if block._text]
    # TODO: code below assumes that unused text blocks are at the end

    if not text_blocks:
        geometry._aabb = np.zeros((2, 3), np.float32)
        return

    # TODO: apply alias via enums
    anchor = geometry._anchor
    anchor_offset = geometry._anchor_offset
    line_height = geometry._line_height * geometry._font_size  # like CSS
    paragraph_spacing = geometry._paragraph_spacing * geometry._font_size
    direction = geometry._direction  # noqa - TODO: should probably use this

    # Calculate offsets to put the blocks beneath each-other, as well as the full rect.
    # Note that the distance between anchor points is independent on the anchor-mode.
    y_offsets = np.zeros((len(text_blocks),), np.float32)
    offset = 0
    total_rect = Rect()
    for i, block in enumerate(text_blocks):
        y_offsets[i] = offset
        offset -= block._nlines * line_height + paragraph_spacing
        total_rect.left = min(total_rect.left, block._rect.left)
        total_rect.right = max(total_rect.right, block._rect.right)
    total_rect.top = text_blocks[0]._rect.top
    total_rect.bottom = y_offsets[-1] + text_blocks[-1]._rect.bottom

    # Get anchor offset
    # Note that the anchoring is dead-simple because the blocks are anchoring based on text_align.
    anchor_offset_x, anchor_offset_y = total_rect.get_offset_for_anchor(
        anchor, anchor_offset
    )

    # Shift bounding box rect, and store for geomerty bounding box.
    # Note how we swap top and bottom here, because bottom has smaller values than top.
    total_rect.shift(anchor_offset_x, anchor_offset_y)
    geometry._aabb = np.array(
        [
            (total_rect.left, total_rect.bottom, 0),
            (total_rect.right, total_rect.top, 0),
        ],
        np.float32,
    )

    # Update positions
    geometry.positions.data[: len(y_offsets), 0] = anchor_offset_x
    geometry.positions.data[: len(y_offsets), 1] = y_offsets + anchor_offset_y
    geometry.positions.update_range(0, len(y_offsets))
