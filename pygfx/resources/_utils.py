"""Utils for the Buffer and Texture classes."""

import time
import sys
import logging
from math import ceil, log2

import numpy as np

logger = logging.getLogger("pygfx")


def get_element_format_from_numpy_array(array):
    """Get the per-element format specifier from a numpy array.
    Returns None if the format appears to be a structured array.
    Raises an error if GPU-incompatible dtypes are used (64 bit).
    """

    # Uniform buffers are scalars with a structured dtype.
    # But can also create storage buffers with complex formats.
    if array.dtype.kind not in "iuf":
        return None

    # GPUs generally don't support 64-bit buffers or textures.
    # Note: the Python docs say that l and L are 32 bit, but converting
    # a int64 numpy array to a memoryview gives a format of 'l' instead
    # of 'q' on some systems/configs? So we need to check the itemsize.
    if array.itemsize == 8:
        raise ValueError(
            f"A dtype of {array.dtype.name} is not supported for buffers, use a 32-bit variant instead."
        )

    return array.dtype.str.lstrip("<>=|")


SYS_ENDIANNESS = "<" if sys.byteorder == "little" else ">"


def is_little_endian(arr):
    """Get whether the given array is little endian."""
    byteorder = arr.dtype.byteorder
    if byteorder == "=":
        byteorder = SYS_ENDIANNESS
    elif byteorder == "|":  # single-byte dtype
        byteorder = "<"
    return byteorder == "<"


def make_little_endian(arr):
    """Get a copy of the array that has the same dtype but little endian."""
    return arr.astype(arr.dtype.newbyteorder("<"))


def check_data_is_clean_for_performance(kind, arr):
    """Check that data is c_contiguous and little endian. Raise an error if not."""
    missing_props = []
    if not is_little_endian(arr):
        missing_props.append("little endian")
    if not arr.flags.c_contiguous:
        missing_props.append("c_contiguous")
    if missing_props:
        raise ValueError(
            f"Given {kind} data is not {', '.join(missing_props)} (enforced because force_contiguous is set)."
        )


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
    target_chunk_count=32,
    min_chunk_bytes=2**8,
    max_chunk_bytes=2**20,
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

    min_chunk_size_x = int(min_chunk_bytes / bytes_per_element)
    min_chunk_sizes = [
        min_chunk_size_x,
        max(1, int(min_chunk_size_x / tex_size[0])),
        max(1, int(min_chunk_size_x / tex_size[0] / tex_size[1])),
    ]

    max_chunk_elements = max_chunk_bytes / bytes_per_element

    # Prepare result
    chunk_size = [1, 1, 1]

    for index in (0, 1, 2):
        min_chunk_size = min_chunk_sizes[index]
        max_chunk_size = tex_size[index]

        # Get approximate chunk size, with limits applied
        approx_chunk_size = max_chunk_size / target_chunk_count
        approx_chunk_size = min(approx_chunk_size, max_chunk_elements)
        approx_chunk_size = max(approx_chunk_size, min_chunk_size)

        # Get the approximate chunk count, and snap the approximate
        # chunk_size to it. This is to have a more or less equal
        # distribution of the chunks (i.e. avoiding a half-chunk at the
        # end). Note that we don't try to get an exact integer number of
        # chunks, becaus this_might be hard/impossible (max_chunk_size might
        # be a prime number). A ceil works better here, otherwise a chunk size
        # can easily snap to the full size for smaller dimensions.
        approx_chunk_count = ceil(max_chunk_size / approx_chunk_size)
        approx_chunk_size = max_chunk_size / approx_chunk_count

        if index == 0:
            # Special care for x
            # If the texture is 3D, taking non-full rows (in x) is doubly-non-contiguous.
            # Benchmarks have shown that its better to not chunk in x in this case.
            if tex_size[2] > 1:
                this_chunk_size = tex_size[index]
            else:
                this_chunk_size = (
                    ceil(approx_chunk_size / element_align) * element_align
                )
        else:
            this_chunk_size = ceil(approx_chunk_size)

        # Apply final limits
        this_chunk_size = min(max_chunk_size, max(1, this_chunk_size))

        # Save
        chunk_size[index] = this_chunk_size
        max_chunk_elements /= this_chunk_size

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

    def __repr__(self):
        return f"<Chunkblock ({self.x}, {self.y}, {self.z}), ({self.nx}, {self.ny}, {self.nz})>"

    def get_offset(self):
        return self.x, self.y, self.z

    def get_size(self):
        return self.nx, self.ny, self.nz


def get_merged_blocks_from_mask_1d(chunk_mask):
    """Algorithm to get a list of chunk descriptions from an 1D mask, with chunks merged.

    Return list of (offset, size) tuples.
    """

    blocks = []
    size = chunk_mask.size
    i = 0
    while i < size:
        if chunk_mask[i]:
            x = i
            nx = 1
            i += 1
            while i < size and chunk_mask[i]:
                nx += 1
                i += 1
            blocks.append((x, nx))
        else:
            i += 1
    return blocks


def get_merged_blocks_from_mask_3d(chunk_mask):
    """Algorithm to get a list of chunk descriptions from a 3D mask, with chunks merged.

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
    # data. Therefore the non-contiguous dimension is merged in full when it
    # contains sufficient on-values.
    #
    # This implementation leverages numpy to generate chunks, which I found
    # was faster (for average chunk counts) than a more Python-isch approach.

    t0 = time.perf_counter()  # noqa

    chunk_count_z, chunk_count_y, chunk_count_x = chunk_mask.shape

    # Get dimensionality, so we know whether to more agrressily merge chunks to get a contiguous chunk.
    ndims = 1
    if chunk_mask.shape[1] > 1:
        ndims = 2
    if chunk_mask.shape[0] > 1:
        ndims = 3

    # Note: don't fill gaps between chunks; this only hurts performance!
    # Even for the non-contiguous dimensions.

    # Get a mask to merge all chunks along a dim
    if ndims >= 2:
        mask_x = chunk_mask.sum(axis=2) >= 0.25 * chunk_count_x
    if ndims == 3:
        mask_y = chunk_mask.sum(axis=1) >= 0.25 * chunk_count_y

    chunks = []

    while True:
        # Get start of next block
        zz, yy, xx = np.where(chunk_mask)
        if not zz.size:
            break

        z, y, x = zz[0], yy[0], xx[0]
        nz, ny, nx = 1, 1, 1

        # Increase block size (i.e. merging)
        still_changing = True
        while still_changing:
            still_changing = False

            # Extend chunk in x-dimension
            while x + nx < chunk_count_x:
                extra = chunk_mask[z : z + nz, y : y + ny, x + nx : x + nx + 1]
                count = np.count_nonzero(extra)
                if count >= 0.5 * ny * nz:
                    nx += 1
                    still_changing = True
                else:
                    break

            # Extend chunk in y-dimension
            while y + ny < chunk_count_y:
                extra = chunk_mask[z : z + nz, y + ny : y + ny + 1, x : x + nx]
                count = np.count_nonzero(extra)
                if count >= 0.5 * nx * nz:
                    ny += 1
                    still_changing = True
                else:
                    break

            # Extend chunk in z-dimension
            while z + nz < chunk_count_z:
                extra = chunk_mask[z + nz : z + nz + 1, y : y + ny, x : x + nx]
                count = np.count_nonzero(extra)
                if count >= 0.5 * nx * ny:
                    nz += 1
                    still_changing = True
                else:
                    break

        # Extend to full size if its sufficiently filled
        if ndims >= 2:
            if mask_x[z, y] or nx >= 0.25 * chunk_count_x:
                x, nx = 0, chunk_count_x
        if ndims == 3:
            if mask_y[z, x] or ny >= 0.25 * chunk_count_y:
                y, ny = 0, chunk_count_y

        # Create the detected chunk
        chunk = ChunkBlock(x, y, z)
        chunk.nx = nx
        chunk.ny = ny
        chunk.nz = nz
        chunks.append(chunk)

        # Update the mask
        chunk_mask[z : z + nz, y : y + ny, x : x + nx] = False

    # print(chunk_mask.shape, len(chunks), f"{(time.perf_counter() - t0)*1000:0.2f}ms")
    return chunks
