import numpy as np
import freetype

from ._atlas import glyph_atlas


# A little cache so we can assign numbers to fonts
fontname_cache = {}


# Set size to match the atlas, a bit less because some glyphs actually become larger
# todo: this could be different for each glyph, otherwise we have to set this
# quite low to also support the big arabic/chinese chars, and thereby waste valuable atlas space.
REF_GLYPH_SIZE = glyph_atlas.glyph_size - 8  # 64 -8 == 56px == 42pt


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
        info = glyph_atlas.get_glyph_info(glyph_hash)
        if not info:
            info = glyph_atlas.set_glyph(
                glyph_hash, *_generate_glyph(face, glyph_index)
            )
        atlas_indices[i] = info["index"]

    return atlas_indices


def _generate_glyph(face, glyph_index):
    # This only gets called for glyphs that are not in the atlas yet.

    gs = glyph_atlas.glyph_size

    # Load the glyph bitmap
    face.load_glyph(glyph_index, freetype.FT_LOAD_DEFAULT)

    try:
        face.glyph.render(freetype.FT_RENDER_MODE_SDF)
    except Exception:  # Freetype refuses SDF for spaces ?
        face.glyph.render(freetype.FT_RENDER_MODE_NORMAL)
    bitmap = face.glyph.bitmap

    # Make the bitmap smaller if it does not fit in the atlas slot.
    # The REF_GLYPH_SIZE should be set such that this does not happen.
    # But when it does, the result is simply a cut-off glyph.
    a = np.array(bitmap.buffer, np.uint8).reshape(bitmap.rows, bitmap.width)
    if a.shape[0] > gs or a.shape[1] > gs:
        try:
            name = face.get_glyph_name(glyph_index).decode()
        except Exception:
            name = "?"
        size1 = f"{a.shape[1]}x{a.shape[0]}"
        msg = f"Warning: glyph {glyph_index} ({name}) was cropped from {size1} to {gs}x{gs}."
        print(msg)
        a = a[:gs, :gs]

    # Put in an array of the right size
    glyph = np.zeros((gs, gs), np.uint8)
    glyph[: a.shape[0], : a.shape[1]] = a

    # Extract other info
    info = {
        "advance": face.glyph.linearHoriAdvance / 65536,
        # "advance": face.glyph.advance.x / 64,  -> less precize
        "rect": (face.glyph.bitmap_left, face.glyph.bitmap_top, a.shape[1], a.shape[0]),
    }

    return glyph, info
