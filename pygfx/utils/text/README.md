# The text rendering process in PyGfx

Text rendering is a notoriously complex task. In PyGfx we draw inspiration
from Vispy, Matplotlib and the work of Nicolas Rougier
(https://www.slideshare.net/NicolasRougier1/siggraph-2018-digital-typography).

## Text rendering steps

This is a high-level overview of the steps of our text rendering process.

### Entrypoint & itemisation

We provide the user with an API to produce text. This can be plain text, Markdown, Html, Math, musical notes, etc. In addition, the user may provide certain font properties (TODO: which ones?). This is the API entry point. 

Depending on the situation, the text is then cut in pieces. E.g. if the input is markdown `"hello *world*"`  this is separated into two parts: one with regular text and one with the bold text. Each part can even use different scripts and have a different font-famly.

TODO: will we separate each word?

The result of this step is a list of text items: each a Unicode string and associated font properties.

### Font selection

*happens for each text item*

The appropriate font file is selected based on the font properties. Not all font properties are used here, only the ones that define what file should be used. This selection process tries to find the best match. E.g. if a weight of 900 is requested but there is only a font file for that family with weight 700 (the standard bold), then this one is used.

Since it's likely that multiple text items use the same font, the results of font selection should be cached.

### Shaping & glyph generation

*happens for each text item*

The text of each item is converted into a list of glyph-indices and positions. A glyph-index is the index of the glyph in the font file. The number of glyphs may not match the length of the text. This is because of e.g. ligatures and diacritics.

Each glyph is now read from the font file (using its glyph-index) and a representation of it is generated and stored in a global atlas. In our case this is a scalar distance field (SDF). The result is the index into the atlas.

From an API point of view, this step exchanges ...

TODO: not sure yet if these should be two steps or one

### Positioning

Each text item now has a list of atlas-indices and matching (relative) positions. These items are then composed so that they form the complete text. This is where word-wrapping, justification etc. are applied. The end result is a single list of atlas-indices and matching positions.

It may also be necessary to reorder the items to deal with a mix of LTR and RTL languages.

TODO: let's mention exactly what text props are used here.

### Rendering

The positions define small quads that are positioned in the vertex shader. In the fragment shader the atlas-index is used to sample the glyph representation from the global atlas, so that it can be rendered to the screen.

## What happens where

The API entrypoint is the `text_geometry()` function that will produce a `TextGeometry` object. Both can be found in the `geometries/_text.py` module. In that module we implement all the steps above, but in a high-level way - the actual implementation of most steps are in the `utils/text` subpackage.

