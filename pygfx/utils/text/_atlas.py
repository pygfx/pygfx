import threading

import numpy as np
import freetype

from ...resources import Texture, Buffer


# todo: TextItems should release glyphs when disposed.


class GlyphAtlas:
    """A global atlas for glyphs (thread-safe)."""

    def __init__(self):

        self._lock = threading.RLock()

        # Keep track of the index for each glyph
        self._hash2index = {}  # hash -> dict
        self._hash2info = {}
        self._index2hash = {}
        self._free_indices = set()

        # Init array
        self._slots_shape = 0, 0
        self._array = np.zeros((0, 0), np.uint8)
        self._rects = np.zeros((0, 4), np.int32)
        self._ensure_capacity(400)

    @property
    def glyph_size(self):
        """The used glyph size."""
        return 64  # 64 seems to be a reasonable size for SDF's

    @property
    def capacity(self):
        """The number of glyphs that the atlas can currently hold."""
        return self._slots_shape[0] * self._slots_shape[1]

    def _ensure_capacity(self, capacity):
        """Ensure a certain minimal capacity. Creates a new array,
        with at least the given capacity. The real capacity may be
        higher due to memory layout.

        This can also be used to reduce the texture size, but since the
        indices of existing glyphs remain valid, this may be restricted
        by existing glyphs with a high index. If the real capacity
        remains unchanged, this is a no-op.
        """

        # note: if this method is made public, it must be wrapped in the lock.

        # Get actual capacity and layout
        min_capacity = max(self._index2hash, default=0) + 1
        the_capacity = max([capacity, min_capacity, 4])
        ncols = nrows = int(np.ceil(the_capacity**0.5))

        if (nrows * ncols) != self.capacity:
            self._set_new_array(nrows, ncols)

    def _set_new_array(self, nrows, ncols):

        gs = self.glyph_size
        used_indices = list(self._index2hash.keys())

        # Bookkeeping
        self._free_indices = set(range(self.capacity)) - set(used_indices)
        self._slots_shape = nrows, ncols

        # Create new array
        array1 = self._array
        array2 = np.zeros((gs * nrows, gs * ncols), np.uint8)
        self._array = array2

        # Copy over the existing glyphs, so that the glyph indices stay valid
        rows_cols1 = [self._row_col_from_index(i) for i in used_indices]
        rows_cols2 = [self._row_col_from_index(i) for i in used_indices]
        for row_col1, row_col2 in zip(rows_cols1, rows_cols2):
            row1, col1 = row_col1
            row2, col2 = row_col2
            glyph = array1[gs * row1 : gs * (row1 + 1), gs * col1 : gs * (col1 + 1)]
            array2[gs * row2 : gs * (row2 + 1), gs * col2 : gs * (col2 + 1)] = glyph

        # Also create new rect array
        rects1 = self._rects
        rects2 = np.zeros((self.capacity, 4), np.int32)
        self._rects = rects2

        # Copy data in rects
        max_i = min(rects1.shape[0], rects2.shape[0])
        rects2[:max_i] = rects1[:max_i]

    def _row_col_from_index(self, index):
        ncols = self._array.shape[1] // self.glyph_size  # == int(self.capacity ** 0.5)
        row = index // ncols
        col = index % ncols
        return row, col

    def _set_glyph_slot(self, index, glyph):
        gs = self.glyph_size
        row, col = self._row_col_from_index(index)
        self._array[gs * row : gs * (row + 1), gs * col : gs * (col + 1)] = glyph
        return row, col

    def get_glyph_info(self, hash):
        """Get info for the glyph corresponding to the given hash.
        If present, returns a dict with at least the field 'index'. Otherwise
        returns None.
        """
        with self._lock:
            try:
                return self._hash2index[hash]
            except KeyError:
                return None

    def set_glyph(self, hash, glyph, info=None):
        """Set the glyph corresponding to the given hash, and return
        the glyph info dict. If the glyph is already set, this method
        returns fast. The given glyph must be an array with shape
        (glyph_size, glyph_size). If info has a 'rect' field, that rect
        is stored in the rects array.
        """

        with self._lock:

            try:
                # If the hash is known, take the shortcut
                return self._hash2index[hash]
            except KeyError:
                # Otherwise, continue below
                pass

            # Check glyph array
            gs = self.glyph_size
            if not (isinstance(glyph, np.ndarray) and glyph.shape == (gs, gs)):
                raise ValueError(f"Glyph array must be a {gs}x{gs} array.")

            # We might need a bigger atlas array
            if not self._free_indices:
                self._ensure_capacity(self.capacity * 2)

            # Get index and construct info
            index = min(self._free_indices)
            info = (info or {}).copy()
            info["index"] = index

            # Update rects if given
            if "rect" in info:
                self._rects[index] = info["rect"]

            # Bookkeeping
            self._free_indices.remove(index)
            self._hash2index[hash] = info
            self._index2hash[index] = hash

            # Store
            self._set_glyph_slot(index, glyph)

            return info

    def free_slot(self, index):
        """Free up a glyph slot."""
        # Note that the array data is not nullified
        with self._lock:
            hash = self._index2hash[index]
            self._hash2index.pop(hash)
            self._index2hash.pop(index)
            self._free_indices.add(index)


class PyGfxGlyphAtlas(GlyphAtlas):
    """A textured pygfx-specific subclass of the GlyphAtlas."""

    @property
    def _gfx_texture_view(self):
        """The texture view for the atlas. The Shared object exposes this object
        in a trackable way.
        """
        return self._texture_view

    @property
    def _gfx_rects_buffer(self):
        """A buffer containing the rects for the glyphs. These indicate the offset
        relative to the glyphs origin, and the (used) size of the glyph, in pixels.
        """
        return self._rects_buffer

    def _set_new_array(self, *args):
        super()._set_new_array(*args)
        # Create texture view
        self._texture = Texture(self._array, dim=2)
        self._texture_view = self._texture.get_view(filter="linear")
        # Create buffer
        self._rects_buffer = Buffer(self._rects)

    def _set_glyph_slot(self, index, glyph):
        row, col = super()._set_glyph_slot(index, glyph)
        gs = self.glyph_size
        self._texture.update_range((gs * row, gs * col, 0), (gs, gs, 1))
        self._rects_buffer.update_range(index, index + 1)


# A little cache so we can assign numbers to fonts
fontname_cache = {}


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


glyph_atlas = PyGfxGlyphAtlas()


# Set size to match the atlas, a bit less because some glyphs actually become larger
# todo: this could be different for each glyph, otherwise we have to set this
# quite low to also support the big arabic/chinese chars, and thereby waste valuable atlas space.
REF_GLYPH_SIZE = glyph_atlas.glyph_size - 8  # 64 -8 == 56px == 42pt
