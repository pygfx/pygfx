"""Utils for the Buffer and Texture classes."""

import logging
from math import ceil, log2


logger = logging.getLogger("pygfx")


STRUCT_FORMAT_ALIASES = {"c": "B", "l": "i", "L": "I"}

# Map memoryview format to short-numpy format.
FORMAT_MAP = {
    "b": "i1",
    "B": "u1",
    "h": "i2",
    "H": "u2",
    "i": "i4",
    "I": "u4",
    "e": "f2",
    "f": "f4",
}


def get_item_format_from_memoryview(mem):
    """Get the per-item format specifier from a memoryview.
    Returns None if the format appears to be a structured array.
    Raises an error if GPU-incompatible dtypes are used (64 bit).
    """

    # Uniform buffers are scalars with a structured dtype.
    # But can also create storage buffers with complex formats.
    if len(mem.format) > 2:
        return None

    # GPUs generally don't support 64-bit buffers or textures.
    # Note: the Python docs say that l and L are 32 bit, but converting
    # a int64 numpy array to a memoryview gives a format of 'l' instead
    # of 'q' on some systems/configs? So we need to check the itemsize.
    if mem.itemsize == 8:
        kind = "64-bit"
        if mem.format in "fd":
            kind = "float64"
        elif mem.format in "ilq":
            kind = "int64"
        elif mem.format in "ILQ":
            kind = "uint64"
        raise ValueError(
            f"A dtype of {kind} is not supported for buffers, use a 32-bit variant instead."
        )

    # Get normalized format
    format = str(mem.format)
    format = STRUCT_FORMAT_ALIASES.get(format, format)

    if format not in FORMAT_MAP:
        raise TypeError(
            f"Cannot convert {format!r} to wgpu format. You should provide data with a different dtype."
        )
    return FORMAT_MAP[format]


def get_alignment_multiplier(bytes_per_element=1, align=16):
    bytes_per_element = bytes_per_element or 1
    factor_max = log2(align)
    if not factor_max.is_integer():
        raise ValueError("align must be factor of two")
    for factor in range(int(factor_max) + 1):
        mult = 2**factor
        extra = (mult * bytes_per_element) % align
        if not extra:
            return mult


def calculate_buffer_chunk_size(buffer_size, **kwargs):
    """Same as calculate_texture_chunk_size, but for just one dimension."""
    res = calculate_texture_chunk_size((int(buffer_size), 1, 1), **kwargs)
    return res[0]


def calculate_texture_chunk_size(
    tex_size,
    *,
    bytes_per_element=1,
    byte_align=16,
    target_chunk_count=20,
    min_chunk_size=2**8,
    max_chunk_size=2**20,
):
    """Calculate the sizes to divide a texture in chunks.

    This algorithm tries to approach the target_chunk_count, divide the chunks equally
    over the dimensions, take alignment into account, as well as bounds on the chunk size.
    """
    # Check / normalize inputs
    tex_size = tuple(int(x) for x in tex_size)
    assert len(tex_size) == 3
    bytes_per_element = int(bytes_per_element)
    byte_align = int(byte_align)
    target_chunk_count = int(target_chunk_count)

    # Calculate the element-alignment to apply for the first dimension
    element_align = get_alignment_multiplier(bytes_per_element, byte_align)

    # Prepare result
    chunk_size = [1, 1, 1]

    # Determine order for iterating over the dimensions. Do the smaller
    # once first, so we can see how much chunk-need we have left for the
    # larger dimensions.
    indices = [(tex_size[i], i) for i in range(3)]
    indices.sort()

    element_count = tex_size[0] * tex_size[1] * tex_size[2]
    byte_count = element_count * bytes_per_element

    remaining_element_count = element_count
    remaining_chunk_count = target_chunk_count  # can be float
    remaining_chunk_count = min(remaining_chunk_count, byte_count / min_chunk_size)
    remaining_chunk_count = max(remaining_chunk_count, byte_count / max_chunk_size)
    remaining_dims = 3

    # Iterate ...
    for this_size, index in indices:
        if this_size == 1:
            remaining_dims -= 1
            continue

        # Get approximate (float) chunk size, bases on (remaining) elements
        power = 1 / remaining_dims  # sqrt3 / sqrt2 / 1
        approx_chunk_size = (remaining_element_count / remaining_chunk_count) ** power

        # Apply limits
        approx_chunk_size = min(this_size, max(1, approx_chunk_size))

        # Get the approximate chunk count, and snap the approximate
        # chunk_size to it. This is to have a more or less equal
        # distribution of the chunks (i.e. avoiding a half-chunk at the
        # end). Note that we don't try to get an exact integer number of
        # chunks, becaus this_might be hard/impossible (this_size might
        # be a prime number). A ceil works better here, otherwise a chunk size
        # can easily snap to the full size for smaller dimensions.
        approx_chunk_count = ceil(this_size / approx_chunk_size)
        approx_chunk_size = this_size / approx_chunk_count

        # If this is the first index, apply byte alignment. Otherwise, just round.
        # Use ceil, because we'd rather have chunks that are slightly too large, than
        # chunks that are slightly to small (and thus needing an extra chunk).
        if index == 0:
            this_chunk_size = ceil(approx_chunk_size / element_align) * element_align
        else:
            this_chunk_size = ceil(approx_chunk_size)

        # Apply limits
        this_chunk_size = min(this_size, max(1, this_chunk_size))
        this_chunk_count = this_size / this_chunk_size  # precise, but possibly float

        # Save
        chunk_size[index] = this_chunk_size

        # Next
        remaining_dims -= 1
        remaining_chunk_count /= this_chunk_count
        remaining_element_count /= this_size

    return tuple(chunk_size)


class ChunkBlock:
    """Little helper object."""

    __slots__ = ["x", "y", "z", "nx", "ny", "nz"]

    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
        self.nx = 1
        self.ny = 1
        self.nz = 1

    def get_offset(self):
        return self.x, self.y, self.z

    def get_size(self):
        return self.nx, self.ny, self.nz


def get_merged_blocks_from_mask_3d(chunk_mask):
    """Algorithm to get a list of chunk descriptions from the mask, with chunks merged.

    Returns a list of objects having fields x, y, z, nx, ny, nz.
    """

    # This algorithm only needs a 3D mask, making it easy to test and maintain.
    #
    # The problem can be defined relatively easy: given a 3D boolean mask, get a
    # list of blocks representing the elements that are "on". Neighbouring
    # on-elements must be combined in a single block (merging). It is ok if a
    # few off-elements are present in a block (aggresive merging). It is not ok
    # to omit any on-elements.
    #
    # From benchmarks we learned that textures suffer a larger performance
    # penalty for chunking than buffers do, except for chunks split in the last
    # dimension. This makes sense because it results in non-contiguous pieces of
    # data. This means that merging chunks is especially important. However,
    # being multi-dimensional, it is also more challenging :) We first process a
    # row (x dimension), merging pretty aggressively, allowing one chunk in
    # between. Then for a stack of rows (y dimension), we merge chunks if they
    # match. Then we repeat this for the depth (z dimension).

    chunk_count_z, chunk_count_y, chunk_count_x = chunk_mask.shape

    chunk_blocks_z_new = []
    chunk_blocks_y_before = []

    for z in range(chunk_count_z):
        chunk_blocks_y_new = []
        chunk_blocks_x_before = []

        for y in range(chunk_count_y):
            chunk_blocks_x_new = []

            # Collect (merged) chunks for the x-dimension
            gap = 999
            for x in range(chunk_count_x):
                dirty = chunk_mask[z, y, x]
                if dirty:
                    if gap <= 1:
                        chunk_blocks_x_new[-1].nx += gap + 1
                        gap = 0
                    else:
                        chunk_blocks_x_new.append(ChunkBlock(x, y, z))
                        gap = 0
                else:
                    gap += 1

            # Merge in the y-dimension
            chunk_blocks_x_composed = []
            for new_block in chunk_blocks_x_new:
                block_match = None
                for prev_block in chunk_blocks_x_before:
                    if prev_block.x == new_block.x and prev_block.nx == new_block.nx:
                        block_match = prev_block
                        break
                if block_match is not None:
                    block_match.ny += 1
                    chunk_blocks_x_composed.append(prev_block)
                else:
                    chunk_blocks_x_composed.append(new_block)
                    chunk_blocks_y_new.append(new_block)
            chunk_blocks_x_before = chunk_blocks_x_composed

        # Merge in the z-dimension
        chunk_blocks_y_composed = []
        for new_block in chunk_blocks_y_new:
            block_match = None
            for prev_block in chunk_blocks_y_before:
                if (
                    prev_block.x == new_block.x
                    and prev_block.y == new_block.y
                    and prev_block.nx == new_block.nx
                    and prev_block.ny == new_block.ny
                ):
                    block_match = prev_block
                    break
            if block_match is not None:
                block_match.nz += 1
                chunk_blocks_y_composed.append(prev_block)
            else:
                chunk_blocks_y_composed.append(new_block)
                chunk_blocks_z_new.append(new_block)
        chunk_blocks_y_before = chunk_blocks_y_composed

    return chunk_blocks_z_new
