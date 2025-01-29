import pylinalg as la
import numpy as np

from ..resources import Buffer
from ..geometries._text import TextGeometry, TextItem
from ..materials._text import TextMaterial
from ..utils import text as textmodule
from ._base import WorldObject


class Text(WorldObject):
    """A text.

    See :class:``pygfx.TextGeometry`` for details.

    Parameters
    ----------
    text : str | optional
        The text to display.
    font_size : float
        The size of the font, in object coordinates or pixel screen coordinates,
        depending on the value of the ``screen_space`` property. Default 12.
    family : str, tuple
        The name(s) of the font to prefer. If multiple names are given, they are
        preferred in the given order. Characters that are not supported by any
        of the given fonts are rendered with the default font (from the Noto
        Sans collection).
    space : enums.CoordSpace
        The coordinate space in which the text expands ("screen", "model"), default "screen".
    position_mode : str
        How ``TextItem`` objects are positioned. With "auto" the layout is performed automatically (in the same space as ``space``).
        If "model" the positioning is done in model-space, a bit as if its a point set with labels (i.e. TextItem's) as markers.
    """

    uniform_type = dict(
        WorldObject.uniform_type,
        rot_scale_transform="4x4xf4",
    )

    def __init__(
        self,
        geometry=None,
        material=None,
        *,
        text=None,
        font_size=12.0,
        family=None,
        space="screen",
        position_mode="auto",
        max_width: float = 0,
        **kwargs,
    ):
        if geometry is None:
            geometry = TextGeometry()
        elif not isinstance(geometry, TextGeometry):
            raise TypeError("Text must have geometry of type TextGeometry.")
        if material is None:
            material = TextMaterial()

        super().__init__(geometry, material, **kwargs)
        self._text_items = []

        self.family = family
        self.font_size = font_size

        if text is not None:
            self.set_text(text)

    @property
    def family(self):
        """The font family to use.

        The name(s) of the font to prefer. If multiple names are given, they are
        preferred in the given order. Characters that are not supported by any
        of the given fonts are rendered with the default font (from the Noto
        Sans collection).
        """
        return self.geometry._font_props.family

    @family.setter
    def family(self, family):
        if family is None:
            self.geometry._font_props = textmodule.FontProps()
        else:
            self.geometry._font_props = textmodule.FontProps(family)
        self._trigger_items_update()

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
        return self.geomtry._font_size

    @font_size.setter
    def font_size(self, value):
        self.geometry._font_size = float(value)
        self._trigger_items_update()

    @property
    def max_width(self):
        """The maximum width of the text. Text will wrap if beyond this
        limit. The coordinate system that this applies to is the same
        as for ``font_size``. Set to 0 for no wrap. Default 0.
        """
        return self.geometry._max_width

    @max_width.setter
    def max_width(self, max_width):
        self.geometry._max_width = float(max_width)
        self._trigger_items_update()

    @property
    def line_height(self):
        """The relative height of a line of text, used to set the
        distance between lines. Default 1.2.
        """
        return self.geometry._line_height

    @line_height.setter
    def line_height(self, height):
        self.geometry._line_height = float(height or 1.2)
        self._trigger_items_update()

    @property
    def text_align(self):
        """Set the alignment of wrapped text. Can be start, end, or center.
        Default "start".

        Text alignment is ignored for top to bottom ('ttb') and
        bottom to top ('btt') directions.
        """
        return self.geometry._text_align

    @text_align.setter
    def text_align(self, align):
        if align is None:
            align = "start"
        if not isinstance(align, str):
            raise TypeError("text-align must be a None or str.")
        align = align.lower()
        if align not in _TEXT_ALIGNMENTS:
            raise ValueError(f"Align must be one of {_TEXT_ALIGNMENTS}. Got {align}.")
        self.geometry._text_align = align
        self._trigger_items_update()

    @property
    def text_align_last(self):
        """Set the alignment of the last line of text.
        Default "auto".

        Text alignment is ignored for top to bottom ('ttb') and
        bottom to top ('btt') directions.
        """
        return self.geometry._text_align_last

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
        self.geometry._text_align_last = align
        self._trigger_items_update()

    @property
    def anchor(self):
        """The position of the origin of the text. This is a string
        representing the vertical and horizontal anchors, separated by
        a dash, e.g. "top-left" or "bottom-center".

        * Vertical values: "top", "middle", "baseline", "bottom".
        * Horizontal values: "left", "center", "right".
        """
        return self.geometry._anchor

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
            raise ValueError(f"Invalid anchor value '{anchor}'") from None
        # Apply
        self.geometry._anchor = f"{anchory}-{anchorx}"
        self._trigger_items_update()

    @property
    def anchor_offset(self):
        """The offset (extra margin) for the 'top', 'bottom', 'left', and 'right' anchors."""
        return self.geometry._anchor_offset

    @anchor_offset.setter
    def anchor_offset(self, value):
        self.geometry._anchor_offset = float(value)
        self._trigger_items_update()

    def set_text(self, text):
        """Set the text for this TextItem."""
        # TODO: could be a settable property?
        # Convert text to words

        if text == self._text or text is None:
            return
        if not isinstance(text, str):
            raise TypeError("TextItem text should be str.")
        self._text = text

        self._text2words()

    # ---

    def _trigger_items_update(self):
        for item in self._text_items:
            item._mark_dirty()

    def _ensure_text_item_count(self, n):
        """Allocate new buffer if necessary."""
        current_size = len(self._text_items)
        if current_size < n or current_size > 4 * n:
            new_size = 2 ** int(np.ceil(np.log2(n)))
            new_size = max(8, new_size)
            self._allocate_text_items(new_size)

    def _allocate_text_items(self, n):
        """Allocate new buffers for text items with the given size."""
        smallest_n = min(n, len(self._text_items))
        # Create new buffers
        new_positions = np.zeros((n, 3), np.float32)
        new_sizes = np.zeros((n,), np.float32)
        # Copy data
        new_positions[:smallest_n] = self.geometry.positions.data[:smallest_n]
        new_sizes[:smallest_n] = self.geometry.sizes.data[:smallest_n]
        # Assign
        # TODO: I feel like resetting these buffers should be done on the geometry
        self.geometry.positions = Buffer(new_positions)
        self.geometry.sizes = Buffer(new_sizes)

        # Allocate / de-allocate text items and their glyphs
        while len(self._text_items) > n:
            item = self._text_items.pop()
            self._deallocate_glyphs(item.indices)
        while len(self._text_items) < n:
            item = TextItem(self.geometry, len(self._text_items))
            self._text_items.append(item)

    def _update_object(self):
        super()._update_object()

        dirty_items = self.geometry._dirty_items
        # Exit early
        if not dirty_items:
            return
        # Update items
        for index in dirty_items:
            item = self._text_items[index]
            item._update()
        # Reset
        dirty_items.clear()

    def _update_world_transform(self):
        # Update when the world transform has changed
        super()._update_world_transform()
        # When rendering in screen space, the world transform is used
        # to establish the point in the scene where the text is placed.
        # The only part of the local transform that is used is the
        # position. Therefore, we also keep a transform containing the
        # local rotation and scale, so that these can be applied to the
        # text in screen coordinates.
        matrix = la.mat_compose((0, 0, 0), self.local.rotation, self.local.scale)
        self.uniform_buffer.data["rot_scale_transform"] = matrix.T

    def create_text_item(self):
        self._allocate_text_items(1)
        return self._text_items[-1]

    def create_text_items(self, n):
        self._allocate_text_items(n)
        return self._text_items[-n:]

    def get_text_item(self, index):
        return self._text_items[index]

    def set_text(self, text):
        lines = text.splitlines()
        self._ensure_text_item_count(len(lines))
        for i, line in enumerate(lines):
            item = self._text_items[i]
            item.set_text(line)

        # Disable unused text items
        for i in range(len(lines), len(self._text_items)):
            item = self._text_items[i]
            item.set_text("")

        # TODO: de-allocate word objects somehow

    def set_markdown(self, markdown, family=None):
        raise NotImplementedError()

        # TODO all other layout props


class LayoutText(Text):
    pass  # ?


class MultiText(Text):
    pass
