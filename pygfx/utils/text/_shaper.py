import freetype
import numpy as np

from ._atlas import atlas


reference_size = 200

font_cache = {}


# https://freetype.org/freetype2/docs/glyphs/glyphs-3.html


def shape_text(text, font_filename):

    # Get font index
    try:
        font_index = font_cache[font_filename]
    except KeyError:
        font_index = len(font_cache) + 1
        font_cache[font_filename] = font_index

    face = freetype.Face(font_filename)
    # todo: cache Face objects, or is this not necessary? Benchmark!

    # Set to a good size for the SDF
    face.set_pixel_sizes(reference_size, reference_size)
    scale = face.size.x_scale  # == y_scale

    # === Shaping
    glyph_indices = [face.get_char_index(c) for c in text]
    advances = [face.get_advance(i, freetype.FT_LOAD_DEFAULT) for i in glyph_indices]
    # Advances can be in font units or 16.16 format
    advances = [(x / 65536 if x > 65536 * 10 else x) for x in advances]

    # Convert advances to positions
    positions = np.zeros((len(advances), 2), np.float32)
    pen_x = 0
    for i in range(len(advances)):
        positions[i] = pen_x, 0
        pen_x += advances[i] / reference_size

    # todo: I'm confused about what units apply where, check visvis, vispy, or maybe just use Harfbuzz, maybe that's more clear.

    # todo: kerning
    # todo: use the line_gap as the reference line_height
    line_gap = face.height

    def generate_glyph(glyph_index):
        face.load_char(glyph_index, freetype.FT_LOAD_DEFAULT)
        face.glyph.render(freetype.FT_RENDER_MODE_SDF)
        bitmap = face.glyph.bitmap
        # a = np.array(bitmap.buffer, np.uint8).reshape(bitmap.rows, bitmap.width)
        a = np.zeros((atlas.glyph_size, atlas.glyph_size), np.uint8)
        a[:, 20:-20] = 255
        return a

    # === Glyph generation
    altas_indices = np.zeros((len(glyph_indices),), np.uint32)
    coverage = np.zeros((len(glyph_indices), 2), np.float32)
    for i, glyph_index in enumerate(glyph_indices):
        glyph_hash = (font_index, glyph_index)
        atlas_index = atlas.register_glyph(glyph_hash, generate_glyph, glyph_index)
        altas_indices[i] = atlas_index
        coverage[i] = 1, 1  # a.shape[1] / reference_size, a.shape[0] / reference_size

    return altas_indices, positions, coverage
