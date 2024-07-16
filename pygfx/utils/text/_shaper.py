"""
Text shaping with Harfbuzz and Freetype.

Relevant links:
* https://harfbuzz.github.io/
* https://freetype.org/freetype2/docs/glyphs/glyphs-3.html

"""

import time
import freetype
import uharfbuzz
import numpy as np


# Determine reference size. This affects the size of the SDF bitmap.
REF_GLYPH_SIZE = 48  # 48px == 64pt


def shape_text(text, font_filename, direction=None):
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
        meta (dict): additional information about this text (width, direction, and more).

    All returned distances are measured in unit font_size.
    """
    return shape_text_hb(text, font_filename, direction)
    # return shape_text_ft(text, font_filename)


class TemporalCache:
    """A simple cache that drops old items based on time."""

    def __init__(self, lifetime, *, getter, minimum_items=0):
        self._ref_lifetime = lifetime
        self._minimum_items = minimum_items
        self._cache = {}
        self._lifetimes = {}
        self._getter = getter

    def __getitem__(self, key):
        """Gets the object corresponding to the given key.

        Will reset the item's lifetime.
        Getting triggers the lifetimes of all items to be checked.
        """
        try:
            res = self._cache[key]
        except KeyError:
            res = self._getter(key)
            self._cache[key] = res

        self._lifetimes[key] = time.time()
        self.check_lifetimes()
        return res

    def __contains__(self, key):
        return key in self._cache

    def __len__(self):
        return len(self._cache)

    def check_lifetimes(self):
        """Check the lifetimes of all objects."""
        # We want this to be fast. I played with only checking a subset,
        # so make it faster if there are say 10k items in the set. This
        # makes it faster for that case, but the added complexity makes
        # it slower already for 100 items, which is way more realistic
        # font count. So let's keep things simple :)
        pretty_old = time.time() - self._ref_lifetime
        old_items = sorted(
            ((key, lt) for key, lt in self._lifetimes.items() if lt < pretty_old),
            key=lambda x: x[1],
        )
        max_items_to_remove = max(len(self._cache) - self._minimum_items, 0)

        for key, lt in old_items[:max_items_to_remove]:
            self._lifetimes.pop(key, None)
            self._cache.pop(key, None)


# Caches to store HB and FT fonts/faces. Caches are scary, because in
# a long-running process in which many fonts come by, we don't want to
# hold/leak memory. For HB it take just 16 KB per font. For this case
# a regular dict would probably be fine. But for FT the font can take
# substantial memory (e.g. a Chinese font can take close to 1MB). This
# is why we use a temporal cache, that drops unused items after 10s.

# However, in many cases, users are probably only using a small
# subset of fonts so if there are fewer than say 5 fonts,
# we don't want to continuously evict them.
# PyGFX uses the vendored Noto fonts (up to 5 ttf files)
# And the user application may use their own 5 in a basic usecase.
# So we allow them to load approximately 20 font files without eviction


def get_hb_font(font_filename):
    ref_size = REF_GLYPH_SIZE

    blob = uharfbuzz.Blob.from_file_path(font_filename)
    face = uharfbuzz.Face(blob)
    font = uharfbuzz.Font(face)
    font.scale = ref_size, ref_size

    return blob, face, font


CACHE_HB = TemporalCache(
    lifetime=10,
    getter=get_hb_font,
    minimum_items=20,
)


def get_ft_face(font_filename):
    ref_size = REF_GLYPH_SIZE

    face = freetype.Face(font_filename)
    face.set_pixel_sizes(ref_size, ref_size)

    return face


CACHE_FT = TemporalCache(
    lifetime=10,
    getter=get_ft_face,
    minimum_items=20,
)


def shape_text_hb(text, font_filename, direction=None):
    """Shape text with Harfbuzz."""

    ref_size = REF_GLYPH_SIZE

    # Prepare buffer
    buf = uharfbuzz.Buffer()
    buf.add_str(text)
    buf.guess_segment_properties()

    is_horizontal = True
    if direction is not None:
        buf.direction = direction
        is_horizontal = direction in ("ltr", "rtl")

    # Load font, maybe from the cache
    blob, face, font = CACHE_HB[font_filename]

    # Shape!
    uharfbuzz.shape(font, buf)

    glyph_infos = buf.glyph_infos
    glyph_positions = buf.glyph_positions
    n_glyphs = len(glyph_infos)

    # Get glyph indices (these can be different from the text's Unicode
    # code points) and convert advances to positions.
    glyph_indices = np.zeros((n_glyphs,), np.uint32)
    positions = np.zeros((n_glyphs, 2), np.float32)
    pen_x = pen_y = 0
    for i in range(n_glyphs):
        glyph_indices[i] = glyph_infos[i].codepoint
        pos = glyph_positions[i]
        positions[i] = (
            (pen_x + pos.x_offset) / ref_size,
            (pen_y + pos.y_offset) / ref_size,
        )
        pen_x += pos.x_advance
        pen_y += pos.y_advance

    # Get font extents
    font_ext = font.get_font_extents(buf.direction)

    meta = {
        "extent": (pen_x if is_horizontal else pen_y) / ref_size,
        "ascender": font_ext.ascender / ref_size,
        "descender": font_ext.descender / ref_size,
        "direction": buf.direction,
        "script": buf.script,
    }

    return glyph_indices, positions, meta


def shape_text_ft(text, font_filename, direction=None):
    """Shape text with FreeType.

    This function is tested but not actually used. It is provided for
    completeness, as a possible fallback. FreeType supports basic
    shaping with kerning but not much more than that (e.g. no glyph
    replacements).
    """

    # assert direction is None  # just ignore the given direction ...

    ref_size = REF_GLYPH_SIZE

    # Load font face
    face = CACHE_FT[font_filename]

    # With Freetype we simply replace each char for a glyph.
    n_glyphs = len(text)
    glyph_indices = np.array([face.get_char_index(c) for c in text], np.uint32)

    # We get the advance of each glyph (can be in font units or 16.16 format)
    advances = [
        face.get_advance(int(i), freetype.FT_LOAD_DEFAULT) for i in glyph_indices
    ]
    advances = [(x / 65536 if x > 65536 * 10 else x) for x in advances]

    # Convert advances to positions
    positions = np.zeros((n_glyphs, 2), np.float32)
    pen_x = 0
    prev = " "
    for i in range(n_glyphs):
        c = text[i]
        kerning = face.get_kerning(prev, c, freetype.FT_KERNING_UNSCALED)
        pen_x += kerning.x / 64
        positions[i] = pen_x / ref_size, 0
        pen_x += advances[i]
        prev = c

    meta = {
        "extent": pen_x / ref_size,
        "ascender": face.ascender / face.units_per_EM,
        "descender": face.descender / face.units_per_EM,
        "direction": "ltr",
        "script": "",
    }

    return glyph_indices, positions, meta
