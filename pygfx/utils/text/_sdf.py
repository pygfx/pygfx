import numpy as np
import freetype

from ._atlas import glyph_atlas
from ._shaper import CACHE_FT


# A little cache so we can assign numbers to fonts
fontname_cache = {}


def generate_glyph(glyph_indices, font_filename):
    """Generate a glyph for the given glyph indices.

    Parameters:
        glyph_indices (list): the indices in the font to render a glyph for.
        font_filename (str): the font to use.

    This generates SDF glyphs and puts them in the atlas. The indices
    of where the glyphs are in the atlas are returned. Glyphs already
    present in the atlas are reused.
    """

    # The glyph metrics place the origin at the baseline. When a bitmap
    # is generated, the bitmap's origin is not the glyphs origin, so
    # we need to correct for this, otherwise the characters are not on
    # the same baseline. Further, the glyph bitmap that is generated
    # varies in size depending on the glyph. The rect info to account for
    # the above points is stored in the per-glyph buffer of the atlas.

    # Get font index (so we can make it part of the glyph hash)
    try:
        font_index = fontname_cache[font_filename]
    except KeyError:
        font_index = len(fontname_cache) + 1
        fontname_cache[font_filename] = font_index

    # Get the face object. We will not need it if all glyphs are already
    # in the atlas, but because of the cache it is fast, and this way
    # we keep the face alive in the cache.
    face = CACHE_FT[font_filename]

    atlas_indices = np.zeros((len(glyph_indices),), np.uint32)
    for i in range(len(glyph_indices)):
        glyph_index = int(glyph_indices[i])
        glyph_hash = (font_index, glyph_index)
        index = glyph_atlas.get_index_from_hash(glyph_hash)
        if index is None:
            glyph, offset = _generate_sdf(face, glyph_index)
            index = glyph_atlas.store_region_with_hash(glyph_hash, glyph, offset)
        atlas_indices[i] = index

    return atlas_indices


def _generate_sdf(face, glyph_index):
    """Generate the SDF bitmap."""
    # This only gets called for glyphs that are not in the atlas yet.

    # FreeType has two ways to render an SDF. The old way is based on
    # the bitmap, and the new way renders from the geometry directly.
    # The new way is the default, and the old way can be selected by
    # first rendering the glyph in bitmap mode. See
    # http://freetype.org/freetype2/docs/reference/ft2-base_interface.html
    #
    # Currently the new way produces quite a few artifacts in e.g. Arabic
    # and emoticons, due to sharp and/or intersecting Bezier curves.
    #
    # Some timing data:
    # about 0.2 ms to create a simple bitmap.
    # about 0.8 ms to create an SDF from the bitmap in the old/stable way (no artifacts)
    # about 1.9 ms to create an SDF the new/fancy way (with artifacts)
    #
    # So, at the time I write this (November 2022) the SDF that goes
    # via the bitmap is both more stable and faster. We might want to
    # revisit this in a year or so.

    # Load the glyph bitmap
    face.load_glyph(glyph_index, freetype.FT_LOAD_DEFAULT)

    # Render bitmap. This forces the SDF to be based off the bitmap, being more stable
    face.glyph.render(freetype.FT_RENDER_MODE_NORMAL)

    # Render SDF
    try:
        face.glyph.render(freetype.FT_RENDER_MODE_SDF)
    except Exception:  # Freetype refuses SDF for spaces ?
        pass

    # Convert to numpy array
    bitmap = face.glyph.bitmap
    glyph = np.array(bitmap.buffer, np.uint8).reshape(bitmap.rows, bitmap.width)
    offset = face.glyph.bitmap_left, face.glyph.bitmap_top

    return glyph, offset
