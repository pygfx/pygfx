import threading

import numpy as np

from ...resources import Texture, Buffer


def generate_size_table(max_size=8192):
    # 8192 is the default wgpu max texture size
    size = 0
    ref_area = 256
    while size < max_size:
        while size * size < ref_area:
            size += 8
        size = min(size, max_size)
        yield (size, size * size, ref_area)
        ref_area *= 2


# A table of predefined atlas sizes, resulting in ± factor-of-two increasing area.
# Each item in the table is a tuple (size, area, ref_area).
# Sizes are rounded to multiples of 8.
SIZES = list(generate_size_table())


def get_suitable_size(approximate_area):
    """Get the atlas size such that it has about the given area."""
    for i in range(1, len(SIZES)):
        size2, area2, _ = SIZES[i]
        if area2 >= approximate_area:
            size1, area1, _ = SIZES[i - 1]
            diff1 = np.abs(area1 - approximate_area)
            diff2 = np.abs(area2 - approximate_area)
            return size1 if diff1 < diff2 else size2
    else:
        return SIZES[-1][0]


class RectPacker:
    """
    This Python code was copied from the Vispy project.
    The algorithm is based on the article by Jukka Jylänki : "A Thousand Ways
    to Pack the Bin - A Practical Approach to Two-Dimensional Rectangle Bin
    Packing", February 27, 2010. More precisely, this is an implementation of
    the Skyline Bottom-Left algorithm based on C++ sources provided by Jukka
    Jylänki at: http://clb.demon.fi/files/RectangleBinPack/.
    """

    # Note: this could be extended to keep track of waste-space in a
    # separate data structure (a.k.a. waste map improvement). Similarly,
    # we could track freed regions for re-use. Right now we keep things
    # simple, and repack to re-claim freed regions.

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
    capacity is much larger than needed.
    """

    def __init__(self, initial_infos_size=1024, initial_array_size=1024):
        self._lock = threading.RLock()

        # Keep track of the index for each glyph
        self._hash2index = {}  # hash -> index
        self._index2hash = {}

        # Indices monotonically increase, but can also be reused from freed regions
        self._index_counter = 0
        self._free_indices = set()

        # Stats
        # The allocated_area represents the sum of areas that the packer has allocated,
        # not including the areas of freed regions. The free_area consists of regions
        # that were once allocated. The waste space in the atlas due to inefficient
        # packing is not counted.
        self._region_count = 0
        self._allocated_area = 0
        self._free_area = 0

        # Props to influence the behavior
        self.clear_free_regions = False
        self.downscale_ratio = 0.25

        # The per-glyph information (used in the shader)
        self._info_dtype = [
            ("origin", np.int32, 2),
            ("size", np.int32, 2),
            ("offset", np.float32, 2),
        ]

        # Init arrays
        self._infos = self._array = np.zeros((0,), np.void)
        self._initial_array_size = get_suitable_size(initial_array_size**2)
        self._set_new_infos_array(initial_infos_size)
        self._set_new_glyphs_array(self._initial_array_size)

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
        """Pack all glyphs up in a new array. If the size is unchanged,
        repacks the glyphs in the current array.
        """
        assert size > 0

        if size == self._array.shape[0]:
            # Keep the array, we'll repack only
            array1 = self._array.copy()
            array2 = self._array
            array2.fill(0)
        else:
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

        self._free_area = 0
        self._allocated_area = allocated_area
        self._free_indices.clear()

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
                # If one third or more is empty, we repack only
                if self._free_area and self._free_area >= 0.5 * self._allocated_area:
                    self._set_new_glyphs_array(self._array.shape[0])
                    rect = self._select_region(w, h)
                # Increase size until we have enough space. Note that the requested
                # region might simply require more than double the original size.
                new_size = 0
                while rect is None:
                    new_size = get_suitable_size(self.total_area * 2)
                    if new_size**2 == self.total_area:
                        # You'd have to stretch things *a lot*, but this can happen in theory
                        raise RuntimeError(
                            "Glyph atlas is out of space and cannot be larger."
                        )
                    self._set_new_glyphs_array(new_size)
                    rect = self._select_region(w, h)

            # Select an index
            if self._free_indices:
                # Reuse an index that was previously freed.
                # Note that we only re-use the index (and thus the slot in the infos array),
                # but not the slot in the glyph array. Therefore _free_area does not change.
                index = self._free_indices.pop()
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
        # We mask the index to only use the first 24 bits. In pygfx we use
        # the upper 8 bits to store font-prop info for the shader.
        index = int(index) & 0x0FFF
        with self._lock:
            # Zero out the region data
            if self.clear_free_regions:
                self.set_region(index, 0)
            # Free in data structure
            assert index < self._index_counter, "Invalid index to free"
            info = self._infos[index]
            x, y = info["origin"]
            w, h = info["size"]
            info["size"] = 0, 0
            self._free_indices.add(index)
            # Bookkeeping
            self._region_count -= 1
            self._free_area += w * h
            self._allocated_area -= w * h
            # Clear hash data
            hash = self._index2hash.pop(index, None)
            if hash is not None:
                self._hash2index.pop(hash)
            # Reduce size?
            if self.downscale_ratio:
                total_area = self._allocated_area + self._free_area
                if self._allocated_area <= self.downscale_ratio * total_area:
                    current_size = self._array.shape[0]
                    if current_size > self._initial_array_size:
                        new_size = get_suitable_size(self.total_area / 2)
                        new_size = max(new_size, self._initial_array_size)
                        if new_size != current_size:
                            self._set_new_glyphs_array(new_size)


class PygfxGlyphAtlas(GlyphAtlas):
    """A textured pygfx-specific subclass of the GlyphAtlas."""

    @property
    def texture(self):
        """The texture for the atlas. The Shared object exposes this object
        in a trackable way.
        """
        return self._texture

    @property
    def info_buffer(self):
        """A buffer containing the info for the glyphs (origin, size, offset)."""
        return self._infos_buffer

    def _set_new_glyphs_array(self, *args):
        # Do the normal behavior
        array = self._array
        super()._set_new_glyphs_array(*args)
        # Create resource object if needed
        if self._array is not array:
            self._texture = Texture(self._array, dim=2, force_contiguous=True)
        # Schedule an update. Note that the infos array is updated due to repacking.
        w, h = self._array.shape[1], self._array.shape[0]
        self._texture.update_range((0, 0, 0), (w, h, 1))
        self._infos_buffer.update_range(0, self._infos.shape[0])

    def _set_new_infos_array(self, *args):
        # Do the normal behavior
        infos = self._infos
        super()._set_new_infos_array(*args)
        # Create resource object if needed
        if self._infos is not infos:
            self._infos_buffer = Buffer(self._infos, force_contiguous=True)
        # Schedule an update
        self._infos_buffer.update_range(0, self._infos.shape[0])

    def set_region(self, index, glyph):
        with self._lock:
            super().set_region(index, glyph)
            info = self._infos[index]
            x, y = info["origin"]
            w, h = info["size"]
            self._texture.update_range((x, y, 0), (w, h, 1))
            self._infos_buffer.update_range(index, index + 1)


glyph_atlas = PygfxGlyphAtlas()
