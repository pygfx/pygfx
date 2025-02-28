"""
This module implements text geometry. This is where the text rendering comes
together. Most steps in the text rendering process come from pygfx.utils.text,
though most of the alignment is implemented here.

For details about the text rendering process, see pygfx/utils/text/README.md


## On text geometry, text blocks, text items, and text pieces

The geomety maintains the text. It holds and manages the arrays for the glyph
locations and atlas indices. It is also the main entrypoint for the user.

Text is divided into multiple blocks, typically one per line/paragraph, similar
to how Qt splits text in blocks in its editor component. This makes it easy to
edit one block, and then merely shift the other blocks to update the layout. In
the multi-text scenario, these blocks can be individually controlled and
positioned by the user.

Each text block again divides its text into multiple items. Each item is a unit
of text that is moved as a whole during layout. Each word typically becomes one
item. Each item is created from one or more text pieces which each can have a
different format (bold, italic, size).

## On layout

Layout is performed on each block, shifting the text items into position based
on text_align, anchor and direction. This positoning is done by offsetting the
item's array of glyph positions. The offset is applied when the item's positions
are copied into the geometry's big glyph_positions buffer.

The TextGeometry also performs a high level layout by positioning the blocks.
The MultiTextGeometry does not do this, as the user is responsible for
positioning the blocks.
"""

from typing import List, Union

import numpy as np

from ..resources import Buffer
from ..utils import text as textmodule
from ..utils.enums import TextAlign, TextAnchor
from ._base import Geometry


# Allow anchor to be written without a dash, for backward compat.
ANCHOR_ALIASES = {anchor.replace("-", ""): anchor for anchor in TextAnchor}
ANCHOR_ALIASES["center"] = ANCHOR_ALIASES["middle"] = "middle-center"

# We cache the extents of small whitespace strings to improve performance
WHITESPACE_EXTENTS = {}


class TextEngine:
    """A small abstraction that allows subclasses of TextGeometry to use a different text engine."""

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

    def get_ws_extent(self, ws, font, direction):
        """Get the extent of a piece of whitespace text. Results of small strings are cached."""
        key = (font.filename, direction)
        map = WHITESPACE_EXTENTS.setdefault(key, {})
        # Try on the full ws
        try:
            return map[ws]
        except KeyError:
            pass
        # Calculate extent on base components
        extent = 0
        for c in ws:
            try:
                extent += map[c]
            except KeyError:
                meta = self.shape_text(c, font.filename, direction)[2]
                map[c] = meta["extent"]
                extent += map[c]
        # Calculate relatively short ws strings (e.g. indentation in code)
        if len(ws) <= 20:
            map[ws] = extent
        return extent


class TextGeometry(Geometry):
    """Geometry for representing text.

    The ``TextGeometry`` creates and stores the geometry to render text. It supports plain text
    as well as formatting with (a subset of) markdown.

    Updates to text are relatively efficient, because text is internally divided
    into text blocks (for lines/paragraphs). Use a ``MultiTextGeometry`` if you
    want to directly control the text blocks.

    Parameters
    ----------
    text : str | list[str]
        The plain text to render (optional). The text is split in one TextBlock per line,
        unless a list is given, in which case each (str) item become a TextBlock.
    markdown : str | list[str]
        The text to render, formatted as markdown (optional).
        See ``set_markdown()`` for details on the supported formatting.
        The text is split in one TextBlock per line,
        unless a list is given, in which case each (str) item become a TextBlock.
    font_size : float
        The size of the font, in object coordinates or pixel screen coordinates,
        depending on the value of the ``screen_space`` property. Default 12.
    family : str, tuple
        The name(s) of the font to prefer.
    direction : str | None
        The text direction overload.
    screen_space : bool
        Whether the text is rendered in model space or in screen space (like a label).
        Default False (i.e. model-space).
    anchor : str | TextAnchor
        The position of the origin of the text. Default "middle-center".
    anchor_offset : float
        The offset (extra margin) for the 'top', 'bottom', 'left', and 'right' anchors.
    max_width : float
        The maximum width of the text. Words are wrapped if necessary. Default zero (no wrapping).
    line_height : float
        A factor to scale the distance between lines. A value of 1 means the
        "native" font's line distance. Default 1.2.
    paragraph_spacing : float
        An extra space between paragraphs. Default 0.
    text_align : str | TextAlign
        The horizontal alignment of the text. Can be "start",
        "end", "left", "right", "center", "justify" or "justify_all". Default
        "start". Text alignment is ignored for vertical text (direction 'ttb' or 'btt').
    text_align_last: str | TextAlign
        The horizontal alignment of the last line of the content element. Default "auto".
    """

    _text_engine = TextEngine()
    is_multi = False

    def __init__(
        self,
        text=None,
        *,
        markdown=None,
        font_size=12,
        family=None,
        direction=None,
        screen_space=False,
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
        # self.colors = Buffer(np.zeros((8,4), np.float32))-> we could later implement per-block colors

        # --- create per-glyph arrays/buffers

        # Index into the atlas that contains all glyphs
        self.glyph_atlas_indices = Buffer(np.zeros((16,), np.uint32))
        # Index into the block list above (i.e. the block.index)
        self.glyph_block_indices = Buffer(np.zeros((16,), np.uint32))
        # Sub-position for glyph size, shaping, kerning, etc.
        self.glyph_positions = Buffer(np.zeros((16, 2), np.float32))
        self.glyph_sizes = Buffer(np.zeros((16,), np.float32))

        # --- init variables to help manage the glyph arrays

        # The number of allocated glyph slots.
        # This must be equal to _glyph_indices_top - len(_glyph_indices_gaps)
        self._glyph_count = 0
        # The index marking the maximum used in the arrays. All elements higher than _glyph_indices_top are free.
        self._glyph_indices_top = 0
        # Free slots that are not in the contiguous space at the end of the arrays.
        self._glyph_indices_gaps = set()

        # Track what blocks need an update. This set is shared with the TextBlock instances.
        self._text_blocks = []  # List of TextBlock instances. May not match length of positions.
        self._dirty_blocks = set()  # Set of ints (text_block.index)

        # --- other geomery-specific things

        self._aabb = np.zeros((2, 3), np.float32)

        # --- set propss

        # Font props
        self.font_size = font_size
        self.family = family
        self.direction = direction
        self.screen_space = screen_space

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

    # --- font properties

    @property
    def font_size(self):
        """The font size.

        For text rendered in screen space (``screen_space`` property is set),
        the size is in logical pixels, and the object's local transform affects
        the final text size.

        For text rendered in world space (``screen_space`` property is *not*
        set), the size is in object coordinates, and the the object's
        world-transform affects the final text size.

        Note that font size is indicative only. Final glyph size further depends on the
        font family, as glyphs may be smaller (or larger) than the indicative
        size. Final glyph size may further vary based on additional formatting
        applied a particular subsection.

        """
        return self._font_size

    @font_size.setter
    def font_size(self, value):
        self._font_size = float(value)
        self._trigger_blocks_update(layout=True)  # only need re-layout

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
        """The font direction overload.

        If not set (i.e. set to None), the text direction is determined
        automatically based on the script, selecting either 'ltr' or 'rtl'
        (Asian scripts are rendered left-to-right by default).

        If set, valid values are 'ltr', 'rtl', 'ttb', or 'btt'.
        Can also specify two values, separated by a dash (e.g. 'ttb-rtl')
        to specify the word-direction and line-direction, respectively.
        """
        return self._direction

    @direction.setter
    def direction(self, direction):
        direction = direction or ""
        word_direction, _, line_direction = direction.partition("-")
        valid_directions = ("", "ltr", "rtl", "ttb", "btt")
        if not (
            word_direction in valid_directions and line_direction in valid_directions
        ):
            raise ValueError(
                "TextGeometry direction components must be None, 'ltr', 'rtl', 'ttb', or 'btt'."
            )
        if not word_direction and not line_direction:
            self._direction = ""
        elif word_direction and line_direction:
            self._direction = f"{word_direction}-{line_direction}"
        elif not line_direction:
            m = {"ltr": "ttb", "rtl": "ttb", "ttb": "rtl", "btt": "rtl"}
            line_direction = m[word_direction]
            self._direction = f"{word_direction}-{line_direction}"
        else:
            self._direction = f"{word_direction}-{line_direction}"

        self._trigger_blocks_update(render_glyphs=True)

    @property
    def screen_space(self):
        """Whether to render text in screen space.

        * False: Render in model-pace (making it part of the scene), and having a bounding box.
        * True: Render in screen-space (like a label).

        Note that in both cases, the local object's rotation and scale will
        still transform the text.
        """
        return self._store.screen_space

    @screen_space.setter
    def screen_space(self, screen_space):
        self._store.screen_space = bool(screen_space)
        self._trigger_blocks_update(layout=True)

    # --- layout properties

    @property
    def anchor(self):
        """The position of the origin of the text.

        Represented as a string representing the vertical and horizontal anchors,
        separated by a dash, e.g. "top-left" or "bottom-center".

        * Vertical values: "top", "middle", "baseline", "bottom".
        * Horizontal values: "left", "center", "right".

        See :obj:`pygfx.utils.enums.TextAnchor`:
        """
        return self._anchor

    @anchor.setter
    def anchor(self, anchor):
        # Init
        if anchor is None:
            anchor = "middle-center"
        elif not isinstance(anchor, str):
            raise TypeError("Text anchor must be str.")
        anchor = ANCHOR_ALIASES.get(anchor, anchor)
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
        """The maximum width of the text.

        Text will wrap if beyond this limit. The coordinate system that this
        applies to is the same as for ``font_size``. Set to 0 for no wrap (default).

        For vertical text, this value indicates the maximum height.
        """
        return self._max_width

    @max_width.setter
    def max_width(self, width):
        self._max_width = float(width or 0)
        self._trigger_blocks_update(layout=True)

    @property
    def line_height(self):
        """The relative height of a line of text.

        Used to set the distance between lines. Default 1.2.
        For vertical text this also defines the distance between the (vertical) lines.
        """
        return self._line_height

    @line_height.setter
    def line_height(self, height):
        self._line_height = float(height or 1.2)
        self._trigger_blocks_update(layout=True)

    @property
    def paragraph_spacing(self):
        """The extra margin between two paragraphs.

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

        Text alignment is ignored for vertical text (direction 'ttb' or 'btt').
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
            align = "start"
        self._text_align = TextAlign[align]
        self._trigger_blocks_update(layout=True)

    @property
    def text_align_last(self):
        """Set the alignment of the last line of text. Default "auto".

        See :obj:`pygfx.utils.enums.TextAlign`:

        Text alignment is ignored for vertical text (direction 'ttb' or 'btt').
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

    def set_text(self, text: Union[str, List[str]]):
        """Set the full text for this TextGeometry.

        Each line (i.e. paragraph) results in one TextBlock, unless a list is given,
        in which case each (str) item become a TextBlock.

        On subsequent calls, blocks are re-used, and blocks that did not change
        have near-zero overhead (only lines/blocks that changed need updating).
        """
        if isinstance(text, str):
            str_per_block = text.splitlines()
        elif isinstance(text, list):
            str_per_block = text
        else:
            raise TypeError("TextGeometry text should be str.")
        self._ensure_text_block_count(len(str_per_block))
        for i, s in enumerate(str_per_block):
            # Note that setting the blocks text is fast if it did not change
            self._text_blocks[i].set_text(s)

        # Do a layout now, so the bounding box is up-to-date
        self._on_update_object()

    def set_markdown(self, text: Union[str, List[str]]):
        """Set the full text, formatted as markdown.

        The supported markdown features are:

        * ``**bold** and *italic* text`` is supported for words, word-parts,
          and (partial) sentences, but not multiple lines (formatting state does
          not cross line boundaries).
        * ``# h1``, ``## h2``, ``### h3``, etc.
        * ``* bullet points``.

        Each line (i.e. paragraph) results in one TextBlock, unless a list is given,
        in which case each (str) item become a TextBlock.

        On subsequent calls, blocks are re-used, and blocks that did not change
        have near-zero overhead (only lines/blocks that changed need updating).
        """
        if isinstance(text, str):
            str_per_block = text.splitlines()
        elif isinstance(text, list):
            str_per_block = text
        else:
            raise TypeError("TextGeometry markdown should be str.")
        self._ensure_text_block_count(len(str_per_block))
        for i, s in enumerate(str_per_block):
            self._text_blocks[i].set_markdown(s)

        # Do a layout now, so the bounding box is up-to-date
        self._on_update_object()

    def get_bounding_box(self):
        screen_space = self._store["screen_space"]
        if screen_space:
            # There is no sensible bounding box for text in screen space, except
            # for the anchor point. Although the point has no volume, it does
            # contribute to e.g. the scene's bounding box.
            return np.zeros((2, 3), np.float32)
        else:
            # A bounding box makes sense, and we calculated it during layout,
            # because we're already shifting rects there.
            return self._aabb

    def get_bounding_sphere(self):
        screen_space = self._store["screen_space"]
        if screen_space:
            # There is no sensible bounding box for text in screen space, except
            # for the anchor point. Although the point has no volume, it does
            # contribute to e.g. the scene's bounding box.
            return np.zeros((4,), np.float32)
        else:
            # A bounding box makes sense, we can calculate it from the rect.
            mean = 0.5 * (self._aabb[1] + self._aabb[0])
            diag = np.norm(self._aabb[1] - self._aabb[0])
            return np.array([[mean[0], mean[1], mean[2], diag]], np.float32)

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
            try:
                block = self._text_blocks[index]
            except IndexError:
                continue  # block was removed after being marked dirty
            did_block_layout = block._update(self)
            need_high_level_layout |= did_block_layout

        # Reset
        dirty_blocks.clear()

        # Update drawing range. Note that the "gaps" are still rendered.
        self.draw_range = 0, self._glyph_indices_top

        # Higher-level layout
        if need_high_level_layout:
            self._layout_blocks()

    def _layout_blocks(self):
        total_rect = apply_high_level_layout(self)
        self._aabb = np.array(
            [
                (total_rect.left, total_rect.bottom, 0),
                (total_rect.right, total_rect.top, 0),
            ],
            np.float32,
        )

    # --- block management

    def _trigger_blocks_update(self, layout=False, render_glyphs=False):
        for block in self._text_blocks:
            block._mark_dirty(layout=layout, render_glyphs=render_glyphs)

    def _ensure_text_block_count(self, n):
        """Allocate new buffer if necessary."""

        # Make sure the underlying buffers are large enough
        current_buffer_size = self.positions.nitems
        if current_buffer_size < n or current_buffer_size > 4 * n:
            new_size = 2 ** int(np.ceil(np.log2(max(n, 1))))
            new_size = max(8, new_size)
            self._allocate_block_buffers(new_size)

        # Add or remove blocks
        while len(self._text_blocks) > n:
            block = self._text_blocks.pop(-1)
            block._clear(self)
        while len(self._text_blocks) < n:
            block = TextBlock(len(self._text_blocks), self._dirty_blocks)
            self._text_blocks.append(block)

    def _allocate_block_buffers(self, n):
        """Allocate new buffers for text blocks with the given size."""
        smallest_n = min(n, len(self._text_blocks))
        new_positions = np.zeros((n, 3), np.float32)
        new_positions[:smallest_n] = self.positions.data[:smallest_n]
        self.positions = Buffer(new_positions)

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
        # These glyphs will still end up in the vertex shader,
        # but it will discard early by producing degeneate triangles.
        # Clear data, for sanity
        self.glyph_block_indices.data[indices] = 0
        self.glyph_atlas_indices.data[indices] = 0
        self.glyph_sizes.data[indices] = 0.0
        self.glyph_positions.data[indices] = 0.0
        # Only update the sizes, shader uses this to check empty slots
        self.glyph_sizes.update_indices(indices)
        # Deallocate
        self._glyph_count -= len(indices)
        # Small optimization to avoid gaps
        if indices.min() == self._glyph_indices_top - len(indices):
            self._glyph_indices_top -= len(indices)
        else:
            self._glyph_indices_gaps.update(indices)
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


class MultiTextGeometry(TextGeometry):
    """Geometry for representing multiple text blocks.

    The ``MultiTextGeometry`` manages a collection of ``TextBlock`` objects,
    for which the text (or markdown) can be individually set, and which can
    be individually positioned. Each text block has the same layout support
    as the ``TextGeometry``.

    Most properties are defined by the geometry, i.e. shared across all text
    objects for that geometry. But this may change in the future to allow more
    flexibility.
    """

    is_multi = True

    def set_text_block_count(self, n):
        """Set the number of text blocks to n.

        Use this if you want to use text blocks directly and you know how many
        blocks you need beforehand. After this, get access to the blocks
        using ``get_text_block()``.
        """
        self._ensure_text_block_count(n)

    def get_text_block_count(self):
        """Get how many text blocks this geometry has."""
        return len(self._text_blocks)

    def create_text_block(self):
        """Create a text block and return it.

        The text block count is increased by one.
        """
        self._ensure_text_block_count(len(self._text_blocks) + 1)
        return self._text_blocks[-1]

    def create_text_blocks(self, n):
        """Create n text blocks and return as a list.

        The text block count is increased by n.
        """
        self._ensure_text_block_count(len(self._text_blocks) + n)
        return self._text_blocks[-n:]

    def get_text_block(self, index):
        """Get the TextBlock instance at the given index.

        The block's position is stored in ``geometry.positions.data[index]``.
        """
        return self._text_blocks[index]

    def _layout_blocks(self):
        total_rect = Rect()
        for block in self._text_blocks:
            total_rect.left = min(total_rect.left, block._rect.left)
            total_rect.right = max(total_rect.right, block._rect.right)
            total_rect.top = max(total_rect.top, block._rect.top)
            total_rect.bottom = min(total_rect.bottom, block._rect.bottom)

        self._aabb = np.array(
            [
                (total_rect.left, total_rect.bottom, 0),
                (total_rect.right, total_rect.top, 0),
            ],
            np.float32,
        )

    def get_bounding_box(self):
        screen_space = self._store["screen_space"]
        if screen_space:
            if not self._text_blocks:
                return None
            if self._aabb_rev == self.positions.rev:
                return self._aabb
            aabb = None
            # Get positions and check expected shape
            positions = self.positions.data[: len(self._text_blocks)]
            aabb = np.array([positions.min(axis=0), positions.max(axis=0)], np.float32)
            # If positions contains xy, but not z, assume z=0
            if aabb.shape[1] == 2:
                aabb = np.column_stack([aabb, np.zeros((2, 1), np.float32)])
            self._aabb = aabb
            self._aabb_rev = self.positions.rev
            return self._aabb
        else:
            # A bounding box makes sense, and we calculated it during layout,
            # because we're already shifting rects there.
            return self._aabb

    def get_bounding_sphere(self):
        screen_space = self._store["screen_space"]
        if screen_space:
            positions = self.positions.data[: len(self._text_blocks)]
            center = positions.mean(axis=0)
            distances = np.linalg.norm(positions - center, axis=0)
            radius = float(distances.max())
            if len(center) == 2:
                return np.array([center[0], center[1], 0.0, radius], np.float32)
            else:
                return np.array([center[0], center[1], center[2], radius], np.float32)
        else:
            mean = 0.5 * (self._aabb[1] + self._aabb[0])
            diag = np.norm(self._aabb[1] - self._aabb[0])
            return np.array([[mean[0], mean[1], mean[2], diag]], np.float32)


class TextBlock:
    """The TextBlock represents one block or paragraph of text.

    Users can obtain instances of this class from the ``MultiTextGeometry``.

    Text blocks are positioned using an entry in ``geometry.positions``.
    The ``TextGeometry`` uses text blocks internally to do efficient re-layout when text is updated.
    With the ``MultiTextGeometry`` the text blocks are positioned by the user or external code.
    """

    def __init__(self, index, dirty_blocks):
        self._index = index  # e.g. the index in geometry.positions
        self._dirty_blocks = dirty_blocks  # a set from the geometry

        self._input = None
        self._need_layout = False
        self._need_render_glyphs = False
        self._pending_position = None

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

        # Update position?
        if self._pending_position is not None:
            geometry.positions.data[self.index] = self._pending_position
            geometry.positions.update_indices(self.index)
            self._pending_position = None

        # Reset flags
        # self._dirty_blocks.discard(self._index)  # no, geometry calls clear
        need_render_glyphs = self._need_render_glyphs
        need_layout = self._need_layout
        self._need_render_glyphs = False
        self._need_layout = False

        # De-allocate old item objects
        if self._old_text_items:
            for item in self._old_text_items:
                item.clear(geometry)
            self._old_text_items = []

        # Quick exit
        if not (need_render_glyphs or need_layout):
            return False

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

    def _clear(self, geometry):
        if self._old_text_items:
            for item in self._old_text_items:
                item.clear(geometry)
            self._old_text_items = []
        for item in self._text_items:
            item.clear(geometry)
        self._text_items = []
        self._input = None

    def set_position(self, x, y, z=0):
        self._mark_dirty()
        self._pending_position = float(x), float(y), float(z)

    def set_text(self, text: str):
        """Set the text for this TextBlock (as a string).

        This is called from ``TextGeometry.set_text()``, but can also be called directly.
        Note that in contrast to ``TextGeometry.set_text()``, setting the text on a TextBlock
        does not result in a re-layout (i.e. the bounding box is not updated).
        """

        if not isinstance(text, str):
            raise TypeError("TextBlock text should be str.")

        input = "text", text
        if input == self._input:
            return
        self._input = input
        self._mark_dirty(layout=True)

        # Split text in words
        text_parts = textmodule.tokenize_text(text)
        self._text_parts_to_items(text_parts)

    def set_markdown(self, text: str):
        """Set the markdown for this TextBlock (as a string).

        This is called from ``TextGeometry.set_markdown()``, but can also be called directly.
        Note that in contrast to ``TextGeometry.set_markdown()``, setting the text on a TextBlock
        does not result in a re-layout (i.e. the bounding box is not updated).
        """

        if not isinstance(text, str):
            raise TypeError("TextBlock markdown should be str.")

        input = "md", text
        if input == self._input:
            return
        self._input = input
        self._mark_dirty(layout=True)

        # Split text in parts using a tokenizer
        text_parts = list(textmodule.tokenize_markdown(text))

        is_newline = True
        is_bold = is_italic = 0
        max_i = len(text_parts) - 1

        for i in range(0, len(text_parts)):
            kind, text = text_parts[i]

            # Detect bullets
            if is_newline and i < max_i and text_parts[i + 1][0] == "ws":
                if text in ("*", "-"):
                    text_parts[i] = "bullet", "  •  "  # ltr and rtl compatible
                    text_parts[i + 1] = "ws", ""  # remove whitespace
                if text.startswith("#") and len(text.strip("#")) == 0:
                    header_level = len(text)
                    text_parts[i] = f"fmt:h{header_level}", text
                    text_parts[i + 1] = "ws", ""  # remove whitespace
            if kind == "nl":
                is_newline = True

            elif kind != "ws":
                is_newline = False

            # Detect bold / italics
            if kind == "stars":
                # Get what surrounding parts look like
                prev_is_wordlike = next_is_wordlike = False
                if i > 0:
                    prev_is_wordlike = text_parts[i - 1][0] != "ws"
                if i < max_i:
                    next_is_wordlike = text_parts[i + 1][0] != "ws"
                # Decide how to format
                if not prev_is_wordlike and next_is_wordlike:
                    # Might be a beginning
                    if text == "**":
                        text_parts[i] = "fmt:+b", text
                        is_bold += 1
                    elif text == "*":
                        text_parts[i] = "fmt:+i", text
                        is_italic += 1
                elif prev_is_wordlike and not next_is_wordlike:
                    # Might be an end
                    if text == "**":
                        text_parts[i] = "fmt:-b", text
                        is_bold = max(0, is_bold - 1)
                    elif text == "*":
                        text_parts[i] = "fmt:-i", text
                        is_italic = max(0, is_italic - 1)
                elif prev_is_wordlike and next_is_wordlike:
                    if text == "**":
                        if is_bold:
                            text_parts[i] = "fmt:-b", text
                            is_bold = max(0, is_bold - 1)
                        else:
                            text_parts[i] = "fmt:+b", text
                            is_bold += 1
                    elif text == "*" and is_italic:
                        if is_italic:
                            text_parts[i] = "fmt:-i", text
                            is_italic = max(0, is_italic - 1)
                        else:
                            text_parts[i] = "fmt:+i", text
                            is_italic += 1

        # Produce text items
        self._text_parts_to_items(text_parts)

    def _text_parts_to_items(self, iter_of_kind_text):
        # A TextItem represents one "word"; one thing held together during layout.
        # Multiple pieces can go into one TextItem, mostly when the word has multiple different formatting in it.
        pending_whitespace = ""
        pending_pieces = []
        new_items = []

        def add_piece(format, text):
            pending_pieces.append((format, text))

        def flush_pieces(force=False):
            nonlocal pending_whitespace
            if pending_pieces or force:
                if self._text_items:
                    item = self._text_items.pop(0)
                else:
                    item = TextItem()
                item.set_text_pieces(tuple(pending_pieces))
                item.ws_before = pending_whitespace
                pending_pieces.clear()
                new_items.append(item)
                pending_whitespace = ""

        # In the code below, we resolve format-modifiers to an 'absolute' format.
        # These are both implementation details, but let's document them here for clarity:
        #
        # Modifiers:
        # +b: make bold
        # -b: unbold
        # +i: make italic
        # -i: make italic
        # h1 h2 h3 h4: headers
        format = {}
        bold_level = italic_level = 0

        # Process the parts to create TextItem objects
        for kind, text in iter_of_kind_text:
            if kind.startswith("fmt:"):
                modifier = kind.partition(":")[-1]
                if modifier == "+b":
                    bold_level += 1
                    format["weight"] = 300
                elif modifier == "-b":
                    bold_level -= 1
                    if not bold_level:
                        format.pop("weight", None)
                elif modifier == "+i":
                    italic_level += 1
                    format["slant"] = True
                elif modifier == "-i":
                    italic_level -= 1
                    if not italic_level:
                        format.pop("slant", None)
                elif modifier.startswith("h"):
                    level = int(modifier[1:])
                    format["size"] = [1, 2.0, 1.5, 1.25][level] if level <= 3 else 1.1
            elif kind == "ws":
                flush_pieces()
                pending_whitespace += text
            elif kind == "nl":
                format.clear()
                bold_level = italic_level = 0
                flush_pieces()
                if pending_whitespace or not new_items:
                    flush_pieces(force=True)
                new_items[-1].nl_after += text
            else:
                add_piece(format.copy(), text)

        flush_pieces()
        if pending_whitespace:
            flush_pieces(force=True)

        # Store old items that need to be de-allocated
        for item in self._text_items:
            if len(item.glyph_indices):
                self._old_text_items.append(item)

        # Store new worlds
        self._text_items = new_items


class TextItem:
    """Represents one unit of text that moves as a whole to a new line when wrapped (usually a word).
    This is a low-level internal object (not public).
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
        "pieces",
        "positions",
        "sizes",
        "ws_before",
    ]

    def __init__(self):
        # The text defines the arrays
        self.pieces = None

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
        self.glyph_indices = np.zeros((0,), np.uint32)

        # Transform info when copying to he geometry buffers. Set during layout.
        self.offset = (1.0, 0.0, 0.0)
        self.layout_offset = None  # used by layout as a temp var

        # Metadata
        self.extent = 0
        self.ascender = 0
        self.descender = 0
        self.direction = None

    def set_text_pieces(self, pieces):
        # The pieces arg should be [(format, text), (format, text), ...]
        if pieces != self.pieces:
            self.pieces = pieces
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

        # The direction may be forced by the geometry. If the geometry's direction is
        # None (default), the first piece that is word-like will determine the direction
        # for this text item.
        direction = geometry._direction.partition("-")[0] or None

        if not self.pieces:
            self.atlas_indices = None
            self.positions = None
            self.sizes = None
            self.glyph_count = 0
            if self.ws_before:
                font = textengine.select_font(" ", font_props)[0][1]
                self.margin_before = textengine.get_ws_extent(
                    self.ws_before, font, direction
                )
            return

        # Prepare containers for array
        atlas_indices_list = []
        positions_list = []
        sizes_list = []

        # Init meta data
        extent = ascender = descender = 0
        calculate_margin_before = bool(self.ws_before)

        # Text rendering steps: font selection, shaping, glyph generation
        last_reverse_index = 0
        for format, text2 in self.pieces:
            for text, font in textengine.select_font(text2, font_props):
                rsize = format.get("size", 1.0)
                weight, slant = format.get("weight", 0), format.get("slant", False)
                unicode_indices, positions, meta = textengine.shape_text(
                    text, font.filename, direction
                )
                atlas_indices = textengine.generate_glyph(
                    unicode_indices, font.filename
                )
                encode_font_props_in_atlas_indices(atlas_indices, weight, slant)
                if rsize != 1.0:
                    positions *= rsize
                if extent:
                    positions[:, 0] += extent  # put pieces next to each-other
                sizes = np.full((positions.shape[0],), rsize, np.float32)
                extent = extent + meta["extent"] * rsize
                ascender = max(ascender, meta["ascender"] * rsize)
                descender = min(descender, meta["descender"] * rsize)  # is neg
                # The first piece that has actual words (not numbers or punctuation) defines direction
                if direction is None and meta["script"]:
                    direction = meta["direction"]
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

                # Calculate margin_before based on the font and direction of the first piece
                if calculate_margin_before:
                    calculate_margin_before = False
                    self.margin_before = textengine.get_ws_extent(
                        self.ws_before, font, direction
                    )

        # Store meta data on the item
        self.extent = extent
        self.ascender = ascender
        self.descender = descender
        self.direction = direction  # Can be None

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
        geometry.glyph_block_indices.update_indices(indices)
        geometry.glyph_atlas_indices.update_indices(indices)
        geometry.glyph_positions.update_indices(indices)
        geometry.glyph_sizes.update_indices(indices)

    def clear(self, geometry):
        if len(self.glyph_indices):
            geometry._glyphs_deallocate(self.glyph_indices)
            self.glyph_indices = np.zeros((0,), np.uint32)


def encode_font_props_in_atlas_indices(atlas_indices, weight, slant):
    # We could put font properties in their own buffer(s), but to
    # safe memory, we encode them in the top bits of the atlas
    # indices. This seems like a good place, because these top bits
    # won't be used (2**24 is more than enough glyphs), and the
    # glyph index is a rather "opaque" value to the user anyway.
    # You can think of the new glyph index as the index to the glyph
    # in the atlas, plus props to tweak its appearance.
    if slant:
        atlas_indices += 0x08000000
    # weight_offset = font_props.weight - font.weight
    weight_offset = weight
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

    @property
    def height(self):
        return self.top - self.bottom

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


# ----- Layout functions


def apply_block_layout(geometry, text_block):
    """The layout step. Updates positions and sizes to finalize the geometry."""

    items = text_block._text_items

    if not items:
        text_block._nlines = 1  # an empty line also takes the space of ine line
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
    anchor_offset = geometry._anchor_offset

    # Get the direction to apply
    word_direction, _, line_direction = geometry._direction.partition("-")
    line_direction = line_direction or "ttb"

    # If not overriden by the geometry, the block direction is defined by the direction of the
    # first item that has a direction. Only items with actual words/script have a direction.
    # (Thanks to Harfbuz we can actually distinguish between script / no-script.)
    ordered_items = items
    if not word_direction:
        # Find direction
        for item in items:
            if item.direction:
                word_direction = item.direction
                break
        word_direction = word_direction or "ltr"
        # Re-order to allow subsentences in other scripts.
        # Note that in this case we only have ltr and rtl, because vertical
        # text only happens when forced by the geometry, and then everything is the same direction.
        ordered_items = []
        i = 0
        for item in items:
            if item.direction and item.direction != word_direction:
                ordered_items.insert(i, item)
            else:
                ordered_items.append(item)
                i = len(ordered_items)

    word_direction_is_horizontal = word_direction in ("ltr", "rtl")
    line_direction_is_vertical = line_direction in ("ttb", "btt")

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
    if line_direction_is_vertical:
        text_align_map = {"start": "left", "end": "right"}
        if word_direction == "rtl":
            text_align_map = {"start": "right", "end": "left"}
        text_align = text_align_map.get(text_align, text_align)
        text_align_last = text_align_map.get(text_align_last, text_align_last)
    else:
        # For horizonal line direction (i.e. vertical word-direction)
        # we don't support text align.
        text_align = text_align_last = "left" if line_direction == "ltr" else "right"

    # If we wrote the above correctly, we have a clear subset of alignment vars
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
        # Update position
        skip = n_new_lines * line_height + n_new_paragraphs * paragraph_spacing
        if line_direction == "ttb":
            offset[1] -= skip
            offset[0] = 0
        elif line_direction == "btt":
            offset[1] += skip
            offset[0] = 0
        elif line_direction == "rtl":
            offset[1] = 0
            offset[0] -= skip
        elif line_direction == "ltr":
            offset[1] = 0
            offset[0] += skip
        # Add the line
        if current_line:
            lines.append(current_line)
            rects.append(current_rect)
            current_line = []
            current_rect = Rect()

    # Resolve position and sizes
    for item in ordered_items:
        # Get item width and determine if we need a new line
        apply_whitespace_margin = True
        if max_width > 0 and current_line:
            if word_direction_is_horizontal:
                current_width = current_rect.width
            else:
                current_width = current_rect.height

            item_width = (item.margin_before + item.extent) * font_size
            if current_width + item_width > max_width:
                make_new_line()
                apply_whitespace_margin = False

        if word_direction == "ltr":
            # Update offset
            if apply_whitespace_margin:
                offset[0] += item.margin_before * font_size
            item.layout_offset = tuple(offset)
            offset[0] += item.extent * font_size
            # Update rect
            current_rect.left = 0
            current_rect.right = offset[0]
            current_rect.top = max(current_rect.top, item.ascender * font_size)
            current_rect.bottom = min(current_rect.bottom, item.descender * font_size)
        elif word_direction == "rtl":
            # Update offset
            if apply_whitespace_margin:
                offset[0] -= item.margin_before * font_size
            offset[0] -= item.extent * font_size
            item.layout_offset = tuple(offset)
            # Update rect
            current_rect.left = offset[0]
            current_rect.right = 0
            current_rect.top = max(current_rect.top, item.ascender * font_size)
            current_rect.bottom = min(current_rect.bottom, item.descender * font_size)
        elif word_direction == "ttb":
            # Update offset
            if apply_whitespace_margin:
                offset[1] -= item.margin_before * font_size
            item.layout_offset = tuple(offset)
            offset[1] -= item.extent * font_size
            # Update rect
            current_rect.top = 0
            current_rect.bottom = offset[1]
            current_rect.left = item.descender * font_size
            current_rect.right = item.ascender * font_size
        elif word_direction == "btt":
            # Update offset
            if apply_whitespace_margin:
                offset[1] += item.margin_before * font_size
            offset[1] += item.extent * font_size
            item.layout_offset = tuple(offset)
            # Update rect
            current_rect.top = offset[1]
            current_rect.bottom = 0
            current_rect.left = item.descender * font_size
            current_rect.right = item.ascender * font_size

        current_line.append(item)

        # The item can have newlines too. Does not happen when using geometry.set_text(),
        # but can happen when using TextBlock.set_text().
        if item.nl_after:
            make_new_line(len(item.nl_after), 1)

    # Properly end the loop
    if current_line:
        make_new_line()

    # Calculate block rect. The top is positive, the bottom is negative (descender).
    block_rect = Rect()
    for line, rect in zip(lines, rects):
        # For rtl, align each line so left is at the origin
        if word_direction == "rtl":
            shift = -rect.left
            rect.shift(shift, 0)
            for item in line:
                layout_offset = item.layout_offset
                item.layout_offset = layout_offset[0] + shift, layout_offset[1]
        # Aggregate
        if line_direction_is_vertical:
            block_rect.left = min(block_rect.left, rect.left)
            block_rect.right = max(block_rect.right, rect.right)
        else:
            block_rect.top = max(block_rect.top, rect.top)
            block_rect.bottom = min(block_rect.bottom, rect.bottom)

    if line_direction == "ttb":
        block_rect.top = rects[0].top
        block_rect.bottom = offset[1] + line_height + rects[-1].bottom
    elif line_direction == "btt":
        block_rect.top = offset[1] - line_height + rects[-1].top
        block_rect.bottom = rects[0].bottom
    elif line_direction == "rtl":
        block_rect.left = offset[0] + line_height + rects[-1].left
        block_rect.right = rects[0].right
    elif line_direction == "ltr":
        block_rect.left = rects[0].left
        block_rect.right = offset[0] - line_height + rects[-1].right

    if text_align == "justify" or text_align_last == "justify":
        if max_width > 0:
            block_rect.right = max_width
        else:
            # Within a block, support justify without max-width
            max_width = block_rect.right

    # Determine horizontal anchor
    if geometry.is_multi:
        # Full layout done here, including anchoring.
        anchor_offset_x, anchor_offset_y = block_rect.get_offset_for_anchor(
            anchor, anchor_offset
        )
    else:
        # If the geometry does its layout, it's far easier to *not* do the anchoring here,
        # except to anchor according to text alignment.
        anchor_offset_x, anchor_offset_y = block_rect.get_offset_for_anchor(
            f"baseline-{text_align}", 0
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
        block_length = block_rect.right - block_rect.left

        extra_space_per_word = 0
        if not word_direction_is_horizontal:
            line_offset_x = 0
        elif align == "justify":
            line_offset_x = 0
            length_to_add = max_width - line_length
            nwords = len(line)
            if nwords > 1:
                extra_space_per_word = length_to_add / (nwords - 1)
        elif align == "center":
            line_offset_x = 0.5 * block_length - 0.5 * line_length
        elif align == "right":
            line_offset_x = block_length - line_length
        else:  # elif align == "left":
            line_offset_x = 0

        for j, item in enumerate(line):
            if word_direction == "rtl":
                j = len(line) - j - 1
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


def apply_high_level_layout(geometry):
    text_blocks = geometry._text_blocks

    if not text_blocks:
        geometry._aabb = np.zeros((2, 3), np.float32)
        return

    anchor = geometry._anchor
    anchor_offset = geometry._anchor_offset
    line_height = geometry._line_height * geometry._font_size  # like CSS
    par_spacing = geometry._paragraph_spacing * geometry._font_size

    # Get line direction.
    line_direction = geometry._direction.partition("-")[2] or "ttb"
    line_direction_is_vertical = line_direction in ("ttb", "btt")

    # Calculate offsets to put the blocks beneath each-other, as well as the full rect.
    # Note that the distance between anchor points is independent on the anchor-mode.
    offsets = np.zeros((len(text_blocks),), np.float32)
    offset = 0
    total_rect = Rect()
    if line_direction == "ttb":
        for i, block in enumerate(text_blocks):
            rect = block._rect
            offsets[i] = offset
            offset -= max(block._nlines * line_height, 0.5 * rect.height) + par_spacing
            total_rect.left = min(total_rect.left, rect.left)
            total_rect.right = max(total_rect.right, rect.right)
        total_rect.top = text_blocks[0]._rect.top
        total_rect.bottom = offsets[-1] + text_blocks[-1]._rect.bottom
    elif line_direction == "btt":
        for i, block in enumerate(text_blocks):
            rect = block._rect
            offsets[i] = offset
            offset += max(block._nlines * line_height, 0.5 * rect.height) + par_spacing
            total_rect.left = min(total_rect.left, rect.left)
            total_rect.right = max(total_rect.right, rect.right)
        total_rect.top = offsets[-1] + text_blocks[-1]._rect.top
        total_rect.bottom = text_blocks[0]._rect.bottom
    elif line_direction == "rtl":
        for i, block in enumerate(text_blocks):
            rect = block._rect
            offsets[i] = offset
            offset -= max(block._nlines * line_height, 0.5 * rect.width) + par_spacing
            total_rect.bottom = min(total_rect.bottom, rect.bottom)
            total_rect.top = max(total_rect.top, rect.top)
        total_rect.left = offsets[-1] + text_blocks[-1]._rect.left
        total_rect.right = text_blocks[0]._rect.right
    elif line_direction == "ltr":
        for i, block in enumerate(text_blocks):
            rect = block._rect
            offsets[i] = offset
            offset += max(block._nlines * line_height, 0.5 * rect.width) + par_spacing
            total_rect.bottom = min(total_rect.bottom, rect.bottom)
            total_rect.top = max(total_rect.top, rect.top)
        total_rect.left = text_blocks[0]._rect.left
        total_rect.right = offsets[-1] + text_blocks[-1]._rect.right

    # Get anchor offset
    # Note that the anchoring is dead-simple because the blocks are anchoring based on text_align.
    anchor_offset_x, anchor_offset_y = total_rect.get_offset_for_anchor(
        anchor, anchor_offset
    )

    # Shift bounding box rect, and store for geomerty bounding box.
    # Note how we swap top and bottom here, because bottom has smaller values than top.
    total_rect.shift(anchor_offset_x, anchor_offset_y)

    # Update positions
    if line_direction_is_vertical:
        geometry.positions.data[: len(offsets), 0] = anchor_offset_x
        geometry.positions.data[: len(offsets), 1] = offsets + anchor_offset_y
    else:
        geometry.positions.data[: len(offsets), 0] = offsets + anchor_offset_x
        geometry.positions.data[: len(offsets), 1] = anchor_offset_y
    geometry.positions.update_range(0, len(offsets))

    return total_rect
