"""
Text shaping with Harfbuzz and Freetype.

Relevant links:
* https://harfbuzz.github.io/
* https://freetype.org/freetype2/docs/glyphs/glyphs-3.html

"""

import freetype
import numpy as np

from ._atlas import REF_GLYPH_SIZE


def shape_text(text, font_filename):
    """Text shaping. Takes a Unicode string, and replace it with a list
    of glyphs, positioned to represent the given text in a good way.
    Depending on the algorithm used, this takes care of kerning,
    ligatures, and much more.

    Parameters:
        text (str): the text to shape.
        font_filename (str): the font to shape for.

    Returns:
        glyph_indices (list): the indices of the glyphs in the font.
        positions (ndarray): the positions at which these glyphs must be placed.
        full_width (float): the full width of the text.
        meta (dict): additional information about this text.

    All returned distances are measured in unit font_size.
    """
    return shape_text_hb(text, font_filename)
    # return shape_text_ft(text, font_filename)


def shape_text_hb(text, font_filename):

    import uharfbuzz as hb  # noqa

    # Prepare buffer
    buf = hb.Buffer()
    buf.add_str(text)
    buf.guess_segment_properties()

    # Load font
    # todo: cache font objects, or is this not necessary? Benchmark!
    blob = hb.Blob.from_file_path(font_filename)
    face = hb.Face(blob)
    font = hb.Font(face)
    font.scale = REF_GLYPH_SIZE, REF_GLYPH_SIZE

    # Add space so we can measure the space between words
    buf.add_str(" ")

    # Shape!
    hb.shape(font, buf)

    glyph_infos = buf.glyph_infos
    glyph_positions = buf.glyph_positions
    n_glyphs = len(glyph_infos) - 1

    # Get glyph indices, these can be different from the text's Unicode code points
    glyph_indices = [glyph_infos[i].codepoint for i in range(n_glyphs)]

    # Convert advances to positions
    positions = np.zeros((n_glyphs, 2), np.float32)
    pen_x = 0
    for i in range(n_glyphs):
        pos = glyph_positions[i]
        positions[i] = pen_x + pos.x_offset, pos.y_offset
        pen_x += pos.x_advance

    # Normalize (make everything unit font size)
    normalized_positions = positions / REF_GLYPH_SIZE
    full_width = pen_x / REF_GLYPH_SIZE
    space_width = glyph_positions[-1].x_advance / REF_GLYPH_SIZE

    # todo: for line height I think we can use font.get_font_extents("rtl")

    meta = {
        "space_width": space_width,
        "script": buf.script,
        "direction": buf.direction,
    }

    return glyph_indices, normalized_positions, full_width, meta


def shape_text_ft(text, font_filename):

    # Load font
    # todo: cache Face objects, or is this not necessary? Benchmark!
    face = freetype.Face(font_filename)
    face.set_pixel_sizes(REF_GLYPH_SIZE, REF_GLYPH_SIZE)

    # With Freetype we simply replace each char for a glyph.
    glyph_indices = [face.get_char_index(c) for c in text]

    # We get the advance of each glyph (can be in font units or 16.16 format)
    advances = [face.get_advance(i, freetype.FT_LOAD_DEFAULT) for i in glyph_indices]
    advances = [(x / 65536 if x > 65536 * 10 else x) for x in advances]

    # Convert advances to positions
    positions = np.zeros((len(advances), 2), np.float32)
    pen_x = 0
    prev = " "
    for i in range(len(advances)):
        c = text[i]
        kerning = face.get_kerning(prev, c, freetype.FT_KERNING_UNSCALED)
        pen_x += kerning.x / 64
        positions[i] = pen_x, 0
        pen_x += advances[i]
        prev = c

    # Normalize (make everything unit font size)
    normalized_positions = positions / REF_GLYPH_SIZE
    full_width = pen_x / REF_GLYPH_SIZE
    space_width = get_advance_for_space_ft(face) / REF_GLYPH_SIZE

    # todo: use the line_gap as the reference line_height
    line_gap = face.height

    meta = {
        "space_width": space_width,
        "script": "",
        "direction": "ltr",
    }

    return glyph_indices, normalized_positions, full_width, meta


def get_advance_for_space_ft(face):
    glyph_index = face.get_char_index(" ")
    advance = face.get_advance(glyph_index, freetype.FT_LOAD_DEFAULT)
    return advance / 65536 if advance > 65536 * 10 else advance
