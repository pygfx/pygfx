import numpy as np

from ._base import Geometry
from ..resources import Buffer


"""
From https://www.slideshare.net/NicolasRougier1/siggraph-2018-digital-typography

Text rendering can be divided into the following steps:

* Start with Unicode text.
* Itemization: cut in parts (each has own font, formatting and even shaping)
* Reordering: To deal with mix of LTR and RTL languages.
* Shaping: out comes set of positions and glyph indices.
* Positioning: justification and other positioning tweaks.
* Rendering.


The text_geometry function takes care of itemization. E.g. when markdown
is given as "hello *world*" it will produce two TextPart objects, one for
the "hello" and one for "world" in bold.
TODO: do we also split words into parts so we can justify easier?

The same function will also re-order the text parts if necessary.

The TextPart object is responsible for the shaping. From the unicode text
it produces glyph indices and positions. There may be less glyphs than
Unicode characters because of ligatures etc. It also makes sure to add
the glyph to the global glyph atlas.

The TextGeometry takes the text parts and combines them into a single
string of glyphs. It is responsible for the positioning.

Finally, the renderer will render the glyphs on screen.

"""


def text_geometry(
    text=None, markdown=None, max_width=None, line_height=None, text_align=None
):
    """Generate text geometry."""

    # === Itemization - generate a list of TextParts objects
    parts = []
    if text:
        if not isinstance(text, str):
            raise TypeError("Text must be a Unicode string.")
        parts.append(TextPart(text))
    if markdown:
        raise NotImplementedError()

    # === Reordering - put parts in the right order
    pass  # we're assuming LTR for now

    return TextGeometry(
        parts, max_width=max_width, line_height=line_height, text_align=text_align
    )


class TextPart:
    """A text part represents a piece of text that is formatted in a uniform
    way. One piece of text can consists of multiple parts. This class
    is responsible for shaping the text.
    """

    def __init__(self, text, font=None, font_size=12, weight=400, italic=False):

        # === Shaping - generate indices and positions

        self.indices = np.zeros((len(text),), np.uint32)
        self.positions = np.zeros((len(text), 3), np.float32)

        for i in range(len(text)):
            self.positions[i] = i * font_size, 0, 0


class TextGeometry(Geometry):
    def __init__(self, text_parts, max_width=0, line_height=1.2, text_align="left"):

        # Check incoming parts
        self._text_parts = tuple(text_parts)
        for part in self._text_parts:
            if not isinstance(part, TextPart):
                raise TypeError("TextGeometry only accepts TextPart objects.")

        # Set props
        self.max_width = max_width
        self.line_height = line_height
        self.text_align = text_align

        # Compose the parts in a single geometry
        indices_arrays = []
        positions_arrays = []
        glyph_count = 0
        for part in self._text_parts:
            assert part.indices.dtype == np.uint32
            assert part.positions.dtype == np.float32
            assert part.positions.shape == (part.indices.size, 3)
            part.offset = glyph_count
            glyph_count += part.indices.size
            indices_arrays.append(part.indices)
            positions_arrays.append(part.positions)

        # Store
        super().__init__(
            indices=Buffer(np.concatenate(indices_arrays, 0)),
            positions=Buffer(np.concatenate(positions_arrays, 0)),
        )

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

        for part in self._text_parts:
            positions = part.positions.copy()
            # todo: tweak positions
            if not (positions == part.positions).all():
                i1, i2 = part.offset, part.offset + positions.shape[0]
                self.posisions.data[i1:i2] = positions
                self.positions.update_range(i1, i2)
