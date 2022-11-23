import numpy as np
import freetype

from ._atlas import glyph_atlas


# A little cache so we can assign numbers to fonts
fontname_cache = {}


REF_GLYPH_SIZE = 48  # 48px == 64pt


def generate_glyph(glyph_indices, font_filename):
    """Generate a glyph for the given glyph indices.

    Parameters:
        glyph_indices (list): the indices in the font to render a glyph for.
        font_filename (str): the font to use.

    This generates SDF glyphs and puts them in the atlas. The indices
    of where the glyphs are in the atlas are returned. Glyphs already
    present in the atlas are of course reused.
    """

    # Notes on the rect:
    #
    # The glyph metrics place the origin at the baseline. When a bitmap
    # is generated, the bitmap's origin is not the glyphs origin, so
    # we need to correct for this, otherwise the characters are not on
    # the same baseline. Further, the glyph bitmap that is generated
    # varies in size depending on the glyph. We put it in the atlas
    # where each glyph has the same size. Because we put the bitmap in
    # the upperleft corner, this all works well. However, we don't want
    # to process all those empty pixels in the fragment shader.
    #
    # Both issues are solved by exposing the rect of each glyph, which
    # is stored in a buffer along with the atlas texture.

    # Get font index (so we can make it part of the glyph hash)
    try:
        font_index = fontname_cache[font_filename]
    except KeyError:
        font_index = len(fontname_cache) + 1
        fontname_cache[font_filename] = font_index

    face = freetype.Face(font_filename)
    face.set_pixel_sizes(REF_GLYPH_SIZE, REF_GLYPH_SIZE)

    atlas_indices = np.zeros((len(glyph_indices),), np.uint32)
    for i in range(len(glyph_indices)):
        glyph_index = int(glyph_indices[i])
        glyph_hash = (font_index, glyph_index)
        index = glyph_atlas.get_index_from_hash(glyph_hash)
        if index is None:
            glyph, offset = _generate_glyph(face, glyph_index)
            index = glyph_atlas.store_region_with_hash(glyph_hash, glyph, offset)
        atlas_indices[i] = index

    return atlas_indices


def _generate_glyph(face, glyph_index):
    # This only gets called for glyphs that are not in the atlas yet.

    # Load the glyph bitmap
    face.load_glyph(glyph_index, freetype.FT_LOAD_DEFAULT)

    # We first render in bitmap mode. Doing this forces the SDF renderer
    # to use the BSDF approach instead of the outline approach. See
    # http://freetype.org/freetype2/docs/reference/ft2-base_interface.html
    # We do this because there are quite a few artifacts in e.g. Arabic
    # and emoticons, due to sharp and/or intersecting Bezier curves.
    # At the time I write this (07-11-2022) there are some fixes in
    # FreeType's codebase, but freetype (and freetype-py) have not had
    # a release that includes these improvements. Once it has, we can
    # remove this line and uncomment the line in the except-clause below.
    face.glyph.render(freetype.FT_RENDER_MODE_NORMAL)

    # Render SDF
    try:
        face.glyph.render(freetype.FT_RENDER_MODE_SDF)
    except Exception:  # Freetype refuses SDF for spaces ?
        pass  # face.glyph.render(freetype.FT_RENDER_MODE_NORMAL)

    # Convert to numpy array
    bitmap = face.glyph.bitmap
    glyph = np.array(bitmap.buffer, np.uint8).reshape(bitmap.rows, bitmap.width)
    offset = face.glyph.bitmap_left, face.glyph.bitmap_top

    return glyph, offset
