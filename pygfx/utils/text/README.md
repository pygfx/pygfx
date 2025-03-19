# Text rendering in pygfx

Text rendering is a notoriously complex task. In pygfx we draw inspiration
from Vispy, Matplotlib and the work of Nicolas Rougier
(https://www.slideshare.net/NicolasRougier1/siggraph-2018-digital-typography).

## Text rendering steps

This is a high-level overview of the steps of our text rendering
process. The steps are listed below, and explained in more detail in
the following subsections.

* Entrypoint & itemisation
* Font selection
* Shaping
* Glyph generation
* Layout
* Rendering

### Entrypoint & itemisation

We provide the user with an API to produce text. This can be plain text,
Markdown, Html, Math, musical notes, etc. In addition, the user may
provide font properties. This is the API entry point.

Depending on the situation, the text is then cut in pieces. E.g. if the
input is markdown `"hello *world*"`  this is separated into two parts:
one with regular text and one with the bold text. Each part can be
provided with a different font-family. In practice, each word becomes
a separate text item, which allows moving words around during the
layout.

The result of this step is a list of text items: each a Unicode string
and associated font properties.

### Font selection

The text items can be seen as a request to render the text in a certain
way. In the font selection step, the most appropriate font file is
selected. It may be needed to split a text item into more pieces, e.g.
when an emoticon is present at the end of a word. For this step to
function, a system is needed that can detect what fonts are present and
what properties they have.

The result of this step is a list of `(text, fontfile)` tuples.

### Shaping

Shaping is the process of converting a list of characters into a list
of glyphs and their positions. You can do this the naïve way using
FreeType, or the proper way using Harfbuzz. The shaping process takes
into account kerning and glyph-replacements such as ligatures and
diacritics, and is crucial to make e.g. Arabic text look right.

The result of this step is a list of glyph-indices (indices in the font
file) and positions, as well as some meta-data about the font such as
the direction the line height.

### Glyph generation

Each glyph is now read from the font file (using its glyph-index) and
a representation of it is generated and stored in a global atlas. We
use a signed distance field (SDF) to realise high quality rendering.
Glyphs that are already in the atlas can be re-used.

This step exchanges the glyph index (plus font file) for an atlas index.

### Layout

In the previous steps, the incoming list of text items has been
converted to a list of glyph items. These contain an array of glyph
indices, an array of positions, and several attributes needed to do the
layout.

In the layout step, the glyph items are positioned. It performs line
breaks, text wrapping, alignment, justification, anchoring, etc. The
re-ordering of items (if LTR scripts are used) can be seen as
pre-processing for the layout procedure.

The result of this step is a complete pygfx geometry ready for rendering.

### Rendering

Each glyph represents a small rectangular area, to which the contents
of the atlas must be rendered. Each rectangle has an origin in the
atlas, a size (in pixels), and an offset for positioning the rectangle
relative to the glyph's origin.

The SDF rendering offers some interesting features. For one, the glyph
can be rendered at any size and angle. Further, the glyph can be
adjusted for its weight (to a certain degree), and provided with an
outline. It's also quite trivial to implement slanting to approximate
italic text.

## What happens where

The API entrypoint is the `Text` (and `MultiText`) object, which can be found in
the `objects/_text.py` module. In that module we perform all the
steps above, except rendering. The actual implementation of most steps
is in the `utils/text` subpackage.

## Details

### Font properties

Font properties specify how a font looks:

* family: a font name or list of font names (fallbacks).
* style: regular, italic, slanted.
* weight: the boldness.

These properties are used to select a font, but in our case are also
used to influence the rendering, since we can approximate weight and style
pretty well.

### Sizing

The glyphs are rendered at a certain "reference glyph size". The
resulting SDF glyphs are each of a different size (a "." takes less
room than a "W") posing a challenge to efficiently pack them in the
atlas.

The Text object has a `font_size` property. For text rendered in screen
space, this is the font size in logical pixels. For text rendered in
world space, the size is expressed in world units. The object's scale
also affects the final size.

In the shader, both the per-glyph size and reference glyph size are used to
calculate to positions the rectangles.



