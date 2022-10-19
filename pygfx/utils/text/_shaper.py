import freetype
import numpy as np

from ._atlas import glyph_atlas


font_cache = {}


# https://freetype.org/freetype2/docs/glyphs/glyphs-3.html


def shape_text_and_generate_glyph(text, font_filename):

    # todo: split this up in two? Let's see after we've implemented Harfbuzz

    # The purpose of this function is to perform the shaping and glyph generation.
    # These are (now) combined because the glyph generation needs to apply
    # corrections to the positions created by the shaping step.
    #
    # The result is:
    # * An array of indices of the glyphs into the atlas.
    # * An array of 2D positions in unit font space. These can be multiplied with
    #   the font size to get the pixel positions.
    # * An array of 2D coverage numbers, because most glyphs don't occupy
    #   the whole glyph square of the atlas.
    #
    # The glyph metrics place the origin at the baseline. When a bitmap
    # is generated, the bitmap's origin is not the glyphs origin, so
    # we need to correct for this, otherwise the characters are not on
    # the same baseline. The glyph bitmap that is generated varies in
    # size depending on the glyph. We put it in the atlas where each
    # glyph has the same size. Because we put the bitmap in the
    # upperleft corner, this all works well. It does mean that the
    # fragment shader is processing a lot of empty pixels. We prevent
    # this by also letting the shader know the coverage: how much the
    # glyph covers its region in the canvas.
    #
    #  -----------
    # |           |
    # |           |
    # |           |
    # | o         |
    # |           |
    #  -----------

    # Get font index (so we can make it part of the glyph hash)
    try:
        font_index = font_cache[font_filename]
    except KeyError:
        font_index = len(font_cache) + 1
        font_cache[font_filename] = font_index

    # Load font
    # todo: cache Face objects, or is this not necessary? Benchmark!
    face = freetype.Face(font_filename)

    # Set size to match the atlas, a bit less because some glyphs actually become larger
    # todo: set reference size *a lot* lower and check if rendering still works as expected - to test that our scaling is ok
    reference_size = glyph_atlas.glyph_size - 6
    face.set_pixel_sizes(reference_size, reference_size)

    # === Shaping

    # With Freetype we simply replace each char for a glyph.
    glyph_indices = [face.get_char_index(c) for c in text]
    # We get the advance of each glyph
    advances = [face.get_advance(i, freetype.FT_LOAD_DEFAULT) for i in glyph_indices]
    # Advances can be in font units or 16.16 format
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

    # It looks like by calling set_pixel_sizes(), the metrics we use
    # (advances and kerning) are expressed in pixels too. If these would
    # be e.g. points, then we'd convert here.
    positions_in_pixels = positions

    # todo: use the line_gap as the reference line_height
    line_gap = face.height

    # === Glyph generation

    # Prepare
    altas_indices = np.zeros((len(glyph_indices),), np.uint32)
    coverage = np.zeros((len(glyph_indices), 2), np.float32)
    bitmap_offsets = np.zeros((len(glyph_indices), 2), np.float32)

    # Loop over the glyphs
    for i, glyph_index in enumerate(glyph_indices):
        # Get hash and glyph info
        glyph_hash = (font_index, glyph_index)
        info = glyph_atlas.get_glyph_info(glyph_hash)
        if not info:
            info = glyph_atlas.set_glyph(glyph_hash, *generate_glyph(face, glyph_index))
        # Apply geometry
        altas_indices[i] = info["index"]
        coverage[i] = info["coverage"]
        bitmap_offsets[i] = info["offset"]

    # Finalize by making everything unit font size
    positions = (positions_in_pixels + bitmap_offsets) / reference_size
    coverage /= reference_size  # todo: should coverage be scaled like this?
    full_width = pen_x / reference_size
    space_width = get_advance_for_space(face) / reference_size

    # todo: I think we can encode coverage using an uint8

    return altas_indices, positions, coverage, space_width, full_width


def get_advance_for_space(face):
    glyph_index = face.get_char_index(" ")
    advance = face.get_advance(glyph_index, freetype.FT_LOAD_DEFAULT)
    return advance / 65536 if advance > 65536 * 10 else advance


def generate_glyph(face, glyph_index):
    # This only gets called for glyphs that are not in the atlas yet.
    gs = glyph_atlas.glyph_size
    # Load the glyph bitmap
    face.load_glyph(glyph_index, freetype.FT_LOAD_DEFAULT)
    try:
        face.glyph.render(freetype.FT_RENDER_MODE_SDF)
    except Exception:  # Freetype refuses SDF for spaces ?
        face.glyph.render(freetype.FT_RENDER_MODE_NORMAL)
    bitmap = face.glyph.bitmap

    # Put in an array of the right size
    glyph = np.zeros((gs, gs), np.uint8)
    a = np.array(bitmap.buffer, np.uint8).reshape(bitmap.rows, bitmap.width)
    if a.shape[0] > gs or a.shape[1] > gs:
        name = face.get_glyph_name(glyph_index).decode()
        print(
            f"Warning: glyph {glyph_index} ({name}) was cropped from {a.shape[1]}x{a.shape[0]} to {gs}x{gs}."
        )
        a = a[:gs, :gs]
    glyph[: a.shape[0], : a.shape[1]] = a
    # Extract other info
    info = {
        "advance": face.glyph.linearHoriAdvance / 65536,
        # "advance": face.glyph.advance.x / 64,  -> less precize
        "offset": (face.glyph.bitmap_left, face.glyph.bitmap_top),
        "coverage": (a.shape[1], a.shape[0]),
    }
    return glyph, info
