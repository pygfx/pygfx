"""
The six stages of text rendering (also see the README in this subpackage):

* Entrypoint & itemisation
* Font selection
* Shaping
* Glyph generation
* Layout
* Rendering

This namespace contains a function for the three central stages.
Itemisation and layout are implemented in the TextGeometry.
Rendering is implemented in the TextShader.
"""

from ._fontfinder import FontFile  # noqa: F401
from ._fontmanager import FontProps, FontManager, font_manager  # noqa: F401
from ._atlas import glyph_atlas  # noqa: F401
from ._tokenizers import tokenize_text, tokenize_markdown  # noqa: F401
from . import _sdf
from . import _shaper


select_font = font_manager.select_font  # Font selection
shape_text = _shaper.shape_text  # Shaping
generate_glyph = _sdf.generate_glyph  # Glyph generation
