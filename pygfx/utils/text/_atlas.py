import threading

import numpy as np

from ...resources import Texture


# todo: how to invalidate the pipeline of objects that use the atlas when the texture is changed?
# todo: TextItems should release glyphs when disposed.


class GlyphAtlas:
    """A global atlas for glyphs (thread-safe)."""

    def __init__(self):

        self._lock = threading.RLock()

        # Keep track of the index for each glyph
        self._hash2index = {}
        self._index2hash = {}
        self._free_indices = set()

        # Init array
        self._slots_shape = 0, 0
        self._array = np.zeros((0, 0), np.uint8)
        self._ensure_capacity(10)  # todo: more

    @property
    def glyph_size(self):
        """The used glyph size."""
        return 64  # todo: what's a good size?

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

        # Collect info that we need for copying
        array1 = self._array
        rows_cols1 = [self._row_col_from_index(i) for i in used_indices]

        # Apply - create new array
        array2 = np.zeros((gs * nrows, gs * ncols), np.uint8)
        self._slots_shape = nrows, ncols
        self._array = array2
        self._free_indices = set(range(self.capacity)) - set(used_indices)

        # Copy over the existing glyphs, so that the glyph indices stay valid
        rows_cols2 = [self._row_col_from_index(i) for i in used_indices]
        for row_col1, row_col2 in zip(rows_cols1, rows_cols2):
            row1, col1 = row_col1
            row2, col2 = row_col2
            glyph = array1[gs * row1 : gs * (row1 + 1), gs * col1 : gs * (col1 + 1)]
            array2[gs * row2 : gs * (row2 + 1), gs * col2 : gs * (col2 + 1)] = glyph

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

    def register_glyph(self, hash, glyph_creator, *args):
        """Register a glyph in the atlas and return the index.
        If the glyph is already present, this method returns fast.
        Otherwise it will call the given glyph_creator function to
        obtain a glyph (glyph_size x glyph_size array).
        """
        with self._lock:

            try:
                # If the hash is known, take the shortcut
                return self._hash2index[hash]
            except KeyError:
                # Otherwise, continue below
                pass

            # Obtain the glyph
            gs = self.glyph_size
            glyph = glyph_creator(*args)
            if not (isinstance(glyph, np.ndarray) and glyph.shape == (gs, gs)):
                raise ValueError(f"Glyph must be a {gs}x{gs} array.")

            # We might need a bigger array
            if not self._free_indices:
                self._ensure_capacity(self.capacity * 2)

            # Get index and do bookkeeping
            index = min(self._free_indices)
            self._free_indices.remove(index)
            self._hash2index[hash] = index
            self._index2hash[index] = hash

            # Store
            self._set_glyph_slot(index, glyph)

            return index

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
    def texture_view(self):
        """The texture view for the atlas."""
        return self._texture_view

    def _set_new_array(self, *args):
        super()._set_new_array(*args)
        self._texture = Texture(self._array, dim=2)
        self._texture_view = self._texture.get_view(filter="linear")

    def _set_glyph_slot(self, index, glyph):
        row, col = super()._set_glyph_slot(index, glyph)
        gs = self.glyph_size
        self._texture.update_range((gs * row, gs * col, 0), (gs, gs, 1))


atlas = PyGfxGlyphAtlas()
