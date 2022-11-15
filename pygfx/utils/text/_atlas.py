import threading

import numpy as np

from ...resources import Texture, Buffer


# todo: TextItems should release glyphs when disposed.


class RectPacker:
    """
    The algorithm is based on the article by Jukka Jylänki : "A Thousand Ways
    to Pack the Bin - A Practical Approach to Two-Dimensional Rectangle Bin
    Packing", February 27, 2010. More precisely, this is an implementation of
    the Skyline Bottom-Left algorithm based on C++ sources provided by Jukka
    Jylänki at: http://clb.demon.fi/files/RectangleBinPack/.
    The Python code was copied from the Vispy project.
    """

    # Note: could be extended to keep track of waste-space in a separate data structure (a.k.a. waste map improvement)

    def _reset_packer(self):
        self._atlas_nodes = [(0, 0, self._array.shape[1])]

    def _select_region(self, width, height):
        """Find a free region of given size and allocate it.
        Returns the newly allocated region as (x, y, w, h), or None on failure.
        """

        # Find best place to fit it
        best_height = best_width = np.inf
        best_index = -1
        for i in range(len(self._atlas_nodes)):
            y = self._fit_region(i, width, height)
            if y >= 0:
                node = self._atlas_nodes[i]
                if y + height < best_height or (
                    y + height == best_height and node[2] < best_width
                ):
                    best_height = y + height
                    best_index = i
                    best_width = node[2]
                    region = node[0], y, width, height
        if best_index == -1:
            return None

        # Create the node and update the data structure
        node = region[0], region[1] + height, width
        self._atlas_nodes.insert(best_index, node)
        i = best_index + 1
        while i < len(self._atlas_nodes):
            node = self._atlas_nodes[i]
            prev_node = self._atlas_nodes[i - 1]
            if node[0] < prev_node[0] + prev_node[2]:
                shrink = prev_node[0] + prev_node[2] - node[0]
                x, y, w = self._atlas_nodes[i]
                self._atlas_nodes[i] = x + shrink, y, w - shrink
                if self._atlas_nodes[i][2] <= 0:
                    del self._atlas_nodes[i]
                    i -= 1
                else:
                    break
            else:
                break
            i += 1

        # Merge nodes
        i = 0
        while i < len(self._atlas_nodes) - 1:
            node = self._atlas_nodes[i]
            next_node = self._atlas_nodes[i + 1]
            if node[1] == next_node[1]:
                self._atlas_nodes[i] = node[0], node[1], node[2] + next_node[2]
                del self._atlas_nodes[i + 1]
            else:
                i += 1

        return region

    def _fit_region(self, index, width, height):
        """Test if region (width, height) fit into self._atlas_nodes[index].
        Returns y on success, or -1 on failure."""
        node = self._atlas_nodes[index]
        x, y = node[0], node[1]
        shape = self._array.shape
        width_left = width
        if x + width > shape[1]:
            return -1
        i = index
        while width_left > 0:
            node = self._atlas_nodes[i]
            y = max(y, node[1])
            if y + height > shape[0]:
                return -1
            width_left -= node[2]
            i += 1
        return y


class GlyphAtlas(RectPacker):
    """A global atlas for glyphs (thread-safe).

    This object allows allocating regions, which are packed into a large
    internal array. When the internal array is full, it is resized and
    repacked. This means that packed regions are moved to a new location
    (defragmentation). Therefore the actual location must be looked up
    via the infos array. The internal array is also reduced when the
    capacity is much larger than the need.
    """

    def __init__(self, initial_infos_size=8, initial_array_size=8):
        self._lock = threading.RLock()

        # Keep track of the index for each glyph
        self._hash2index = {}  # hash -> index
        self._index2hash = {}

        # Indices monotonically increase, but can also be reused from freed regions
        self._index_counter = 0
        self._freed_indices = set()

        # Stats
        self._region_count = 0
        self._allocated_area = 0
        self._freed_area = 0

        # The per-glyph information (used in the shader)
        self._info_dtype = [
            ("origin", np.int32, 2),
            ("size", np.int32, 2),
            ("offset", np.float32, 2),
        ]

        # Init arrays
        self._infos = self._array = np.zeros((0,), np.void)
        self._initial_array_size = initial_array_size
        self._set_new_infos_array(initial_infos_size)
        self._set_new_glyphs_array(initial_array_size)

    @property
    def region_count(self):
        """The number of regions allocated (excluding freed)."""
        return self._region_count

    @property
    def total_area(self):
        """The total area that the atlas could potentially hold."""
        shape = self._array.shape
        return shape[1] * shape[0]

    @property
    def allocated_area(self):
        """The allocated area (excluding the area of freed regions)."""
        return self._allocated_area

    def _set_new_infos_array(self, size):
        """Create a new array to store per-glyph info."""
        assert size > 0 and size > self._infos.shape[0]

        # Allocate new infos array
        infos1 = self._infos
        infos2 = np.zeros((size,), self._info_dtype)
        self._infos = infos2

        # Make a copy
        infos2[: infos1.shape[0]] = infos1

    def _set_new_glyphs_array(self, size):
        """Create a new array to store the glyphs."""
        assert size > 0

        # Create new array
        array1 = self._array
        array2 = np.zeros((size, size), np.uint8)
        self._array = array2

        # We're going to pack it up fresh
        self._reset_packer()

        # Copy all rect regions to their new location
        allocated_area = 0
        for index in range(self._index_counter):
            info = self._infos[index]
            x1, y1 = info["origin"]
            w1, h1 = info["size"]
            if not w1 or not h1:
                continue  # freed region
            x2, y2, w2, h2 = self._select_region(w1, h1)
            info["origin"] = x2, y2
            array2[y2 : y2 + h2, x2 : x2 + w2] = array1[y1 : y1 + h1, x1 : x1 + w1]
            allocated_area += w2 * h2

        assert allocated_area == self._allocated_area

        self._freed_area = 0
        self._freed_indices.clear()

    def allocate_region(self, w, h):
        """Allocate a region of the given size. Returns the index for
        the new region. The internal array may be repacked and/or
        resized if necessary.
        """
        with self._lock:

            # Select a region
            rect = self._select_region(w, h)

            # If the array is full the above returns None. If so, resize and try again.
            if rect is None:
                if (
                    self._allocated_area
                    and self._freed_area >= 0.5 * self._allocated_area
                ):
                    # Resize to same size: repack only
                    self._set_new_glyphs_array(self._array.shape[0])
                else:
                    # Increase size with about a factor 2
                    rounder = int(8 * np.ceil(max(w, h) / 8))
                    current_size = self._array.shape[0]
                    new_size = (2 * current_size * current_size) ** 0.5
                    new_size = int(rounder * np.ceil(new_size / rounder))
                    self._set_new_glyphs_array(new_size)
                # Try again
                rect = self._select_region(w, h)
                assert rect

            # Select an index
            if self._freed_indices:
                # Reuse an index that was previously freed.
                index = self._freed_indices.pop()
            else:
                # Create a new index
                index = self._index_counter
                self._index_counter += 1
                # Make sure that the per-glyph arrays are large enough
                while index >= self._infos.shape[0]:
                    self._set_new_infos_array(self._infos.shape[0] * 2)

            # Store the rectangle
            info = self._infos[index]
            info["origin"] = rect[:2]
            info["size"] = rect[2:]
            info["offset"] = 0, 0  # set to zero just in case

            # Bookkeeping
            self._region_count += 1
            self._allocated_area += w * h

            return index

    def set_region(self, index, region):
        """Set the region data for index."""
        with self._lock:
            info = self._infos[index]
            x, y = info["origin"]
            w, h = info["size"]
            self._array[y : y + h, x : x + w] = region

    def get_region(self, index):
        """Return a copy of the region corresponding to index."""
        with self._lock:
            info = self._infos[index]
            x, y = info["origin"]
            w, h = info["size"]
            return self._array[y : y + h, x : x + w].copy()

    def get_index_from_hash(self, hash):
        """Get the glyph index from the hash, or None on failure."""
        with self._lock:
            try:
                return self._hash2index[hash]
            except KeyError:
                return None

    def store_region_with_hash(self, hash, region, offset=(0, 0)):
        """Set the region corresponding to the given hash, and return the index."""
        with self._lock:

            # Sanitize
            assert isinstance(region, np.ndarray)
            assert hash not in self._hash2index

            # Reserve a slot
            w, h = region.shape[1], region.shape[0]
            index = self.allocate_region(w, h)

            # Store
            self.set_region(index, region)
            self._infos[index]["offset"] = offset

            # Bookkeeping
            self._hash2index[hash] = index
            self._index2hash[index] = hash

            return index

    def free_region(self, index):
        """Free up a region slot."""
        # Note that the array data is not nullified
        with self._lock:
            # Free in data structure
            info = self._infos[index]
            x, y = info["origin"]
            w, h = info["size"]
            info["size"] = 0, 0
            self._region_count -= 1
            self._freed_area += w * h
            self._allocated_area -= w * h
            self._freed_indices.add(index)
            # Clear hash data
            hash = self._index2hash.pop(index, None)
            if hash is not None:
                self._hash2index.pop(hash)
            # Maybe reduce size
            if self._allocated_area < 0.25 * self.total_area:
                rounder = 8
                current_size = self._array.shape[0]
                if current_size > self._initial_array_size:
                    new_size = (0.5 * current_size * current_size) ** 0.5
                    new_size = int(rounder * np.ceil(new_size / rounder))
                    new_size = max(new_size, self._initial_array_size)
                    self._set_new_glyphs_array(new_size)


class PyGfxGlyphAtlas(GlyphAtlas):
    """A textured pygfx-specific subclass of the GlyphAtlas."""

    @property
    def _gfx_texture_view(self):
        """The texture view for the atlas. The Shared object exposes this object
        in a trackable way.
        """
        return self._texture_view

    @property
    def _gfx_info_buffer(self):
        """A buffer containing the info for the glyphs (origin, size, offset)."""
        return self._infos_buffer

    def _set_new_glyphs_array(self, *args):
        super()._set_new_glyphs_array(*args)
        self._texture = Texture(self._array, dim=2)
        self._texture_view = self._texture.get_view(filter="linear")

    def _set_new_infos_array(self, *args):
        super()._set_new_infos_array(*args)
        self._infos_buffer = Buffer(self._infos)
        # self._offsets_buffer = Buffer(self._offsets)

    def set_region(self, index, glyph):
        with self._lock:
            super().set_region(index, glyph)
            info = self._infos[index]
            x, y = info["origin"]
            w, h = info["size"]
            self._texture.update_range((y, x, 0), (h, w, 1))
            self._infos_buffer.update_range(index, index + 1)


glyph_atlas = PyGfxGlyphAtlas()
