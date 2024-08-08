from math import floor, ceil

import numpy as np

from ._base import Resource
from ._utils import (
    get_element_format_from_numpy_array,
    calculate_texture_chunk_size,
    get_merged_blocks_from_mask_3d,
    check_data_is_clean_for_performance,
    is_little_endian,
    make_little_endian,
    logger,
)


class Texture(Resource):
    """The Texture represents structured 1D, 2D or 3D data on the GPU.

    A texture can be used to represent e.g. image data or colormaps. They can
    also serve as a render target (for the renderer). Supports texture stacks,
    cube textures, and mipmapping.

    Parameters
    ----------
    data : array | None
        The initial data of the texture. It must support the buffer-protocol,
        (e.g. a bytes or numpy array). If None, ``size`` and ``forma`` must be
        provided. The data will be accessible at ``texture.data``, no copies are
        made. The dtype must be compatible with wgpu texture formats.
    dim : int
        The dimensionality of the array (1, 2 or 3).
    size : tuple | None
        The extent ``(width, height, depth)`` of the array. If None, it is
        derived from `dim` and the shape of the data. The texture can also
        represent a stack of images by setting ``dim=2`` and ``depth > 1``, or a
        cube image by setting ``dim=2`` and ``depth==6``.
    format : None | str | ElementFormat | wgpu.TextureFormat
        A format string describing the pixel/voxel format. This can follow
        pygfx' ``ElementFormat`` e.g. "1xf4" for intensity, "3xu1" for rgb, etc.
        Can also be wgpu's ``TextureFormat``. Optional: if None, it is
        automatically determined from the data.
    colorspace : str
        If this data is used as color, it is interpreted to be in this
        colorspace. Can be "srgb" or "physical". Default "srgb".
    generate_mipmaps : bool
        If True, automatically generates mipmaps when transferring data to the
        GPU. Default False.
    chunk_size : None | tuple | int
        The chunk size to use for uploading data to the GPU, expressed in elements (not bytes).
        When None (default) an optimal chunk size is determined automatically.
        A 3-tuple can be given to provide a size for each dimension, or an integer
        to apply for all dimensions.
    force_contiguous : bool
        When set to true, the texture goes into a stricter mode, forcing set data
        to be c_contiguous. This ensures optimal upload performance for cases when
        the data changes often.
    usage : int | wgpu.TextureUsage
        The wgpu ``usage`` flag for this texture. Optional: typically pygfx can
        derive how the texture is used and apply the appropriate flag. In cases
        where it doesn't this param provides an override. This is a bitmask flag
        (values are OR'd).

    Performance tips:

    * If the given data is not c_contiguous, extra memory-copies may be needed
      at upload time, which reduces performance when the data is changed often.
    * RGB textures do not exist in wgpu, but are emulated with an RGBA texture.
      This may introduce extra memory copies, which reduces performance when
      data is changed often.
    * Setting ``force_contiguous`` ensures that the set data is contiguous and
      not RGB, it is recommended to use this when the texture data is dynamic.
    """

    def __init__(
        self,
        data=None,
        *,
        dim,
        size=None,
        format=None,
        colorspace="srgb",
        generate_mipmaps=False,
        chunk_size=None,
        force_contiguous=False,
        usage=0,
    ):
        super().__init__()
        Resource._rev += 1
        self._rev = Resource._rev

        # Attributes for internal use, updated by other parts of pygfx.
        self._wgpu_object = None
        self._wgpu_usage = int(usage)
        self._wgpu_mip_level_count = 1
        self._wgpu_emulate_rgb = False

        # Init
        self._data = None
        self._force_contiguous = bool(force_contiguous)
        assert dim in (1, 2, 3)
        self._store.dim = int(dim)
        self._colorspace = (colorspace or "srgb").lower()
        assert self._colorspace in ("srgb", "physical")
        self._generate_mipmaps = bool(generate_mipmaps)

        # Normalize size
        size = None if size is None else (int(size[0]), int(size[1]), int(size[2]))

        # Process data
        if data is not None:
            # Store data and view, and do some basic checks.
            # The view is a numpy array, but we go via memoryview to ensure data follows the buffer protocol.
            self._data = data
            self._view = view = np.asarray(memoryview(data))
            if self._force_contiguous:
                check_data_is_clean_for_performance("texture", view)
            the_nbytes = view.nbytes
            # Establish texture size
            if size is not None:
                the_size = size
            else:
                the_size = size_from_array(view, dim)
            reshape_array(view, the_size)
            # Establish format
            element_format = get_element_format_from_numpy_array(view)
            if element_format is None:
                raise ValueError(
                    f"Unsupported dtype/format for texture data: {view.dtype}"
                )
            nchannels = int(np.prod(view.shape[3:], initial=1))
            if not (1 <= nchannels <= 4):
                raise ValueError(
                    f"Expected 1-4 texture color channels, got {nchannels}."
                )
            if nchannels == 3:
                self._wgpu_emulate_rgb = True
                if self._force_contiguous:
                    raise ValueError(
                        "When force_contiguous is set, the texture data cannot be rgb, because it requires a padding operation on upload."
                    )
            detected_format = (f"{nchannels}x" + element_format).lstrip("1x")
        elif size is not None and format is not None:
            the_size = size
            the_nbytes = 0
            detected_format = None
        else:
            raise ValueError(
                "Texture must be instantiated with either data or size and format."
            )

        # Check size
        if not all(the_size[i] > 0 for i in range(3)):
            raise ValueError("Texture size cannot be zero.")

        # Store derived props
        self._store.nbytes = the_nbytes
        self._store.size = the_size
        if format is not None:
            self._store.format = str(format)
        elif detected_format:
            self._store.format = detected_format
        else:
            self._store.format = None

        # Get optimal chunk size
        if the_nbytes == 0:  # data is None or empty
            chunk_size = (0, 0, 0)
        elif chunk_size is None:
            chunk_size = calculate_texture_chunk_size(
                the_size,
                bytes_per_element=the_nbytes // int(np.prod(the_size)),
                byte_align=16,
                target_chunk_count=32,
            )
        else:
            if isinstance(chunk_size, int):
                chunk_size = chunk_size, chunk_size, chunk_size
            elif not len(chunk_size) == 3:
                raise ValueError("Chunk size must be int or tuple of 3 ints.")
            chunk_size = tuple(
                min(max(int(chunk_size[i]), 1), the_size[i]) for i in range(3)
            )

        # Init chunks map
        if data is None:
            self._chunks_dirt_flag = 0
            self._chunk_size = (0, 0, 0)
            self._chunk_mask = None
        elif the_nbytes == 0:
            self._chunks_dirt_flag = 0
            self._chunk_size = (0, 0, 0)
            self._chunk_mask = np.ones((0, 0, 0), bool)
        else:
            self._chunks_dirt_flag = 2
            self._chunk_size = chunk_size
            shape = tuple(ceil(the_size[i] / self._chunk_size[i]) for i in (2, 1, 0))
            self._chunk_mask = np.ones(shape, bool)

    @property
    def dim(self):
        """The dimensionality of the texture (1, 2, or 3)."""
        return self._store.dim

    @property
    def data(self):
        """The data for this texture.

        Can be None if the data only exists on the GPU. This object is the same
        that was given to instantiate this object or with ``set_data()``.
        """
        return self._data

    @property
    def view(self):
        """A numpy array view on the data of this texture.

        Can be None if the data only exists on the GPU. This is a view on the
        same memory as ``.data``. It's ``.shape[:3]`` matches ``reversed(size)``.
        """
        return self._view

    @property
    def nbytes(self):
        """Get the number of bytes in the texture."""
        return self._store.nbytes

    @property
    def size(self):
        """The size of the texture as (width, height, depth).
        (always a 3-tuple, regardless of the dimension).
        """
        return self._store.size

    @property
    def format(self):
        """The texture format.

        Usually a pygfx format specifier (e.g. 'u2' for scalar uint16, or '3xf4'
        for RGB float32), but can also be a value from ``wgpu.TextureFormat``.
        """
        return self._store.format

    @property
    def usage(self):
        """Bitmask indicating how the texture can be used in a wgpu pipeline."""
        return self._wgpu_usage

    @property
    def colorspace(self):
        """If this data is used as color, it is interpreted to be in this colorspace.
        Can be "srgb" or "physical". Default "srgb".
        """
        return self._colorspace

    @property
    def generate_mipmaps(self):
        """Whether to automatically generate mipmaps when uploading to the GPU."""
        return self._generate_mipmaps

    def set_data(self, data):
        """Reset the data to a new array.

        This avoids a data-copy compared to doing ``texture.data[:] = new_data``.
        The new data must fit the teture's size and format.
        """
        # Get view
        view = np.asarray(memoryview(data))
        # Do couple of checks
        if self._force_contiguous:
            check_data_is_clean_for_performance("texture", view)
        if view.nbytes != self._view.nbytes:
            raise ValueError("texture.set_data() nbytes does not match.")
        if view.dtype != self._view.dtype:
            raise ValueError("texture.set_data() format does not match.")
        # Make sure the shape is ok.
        reshape_array(view, self.size)
        # Ok
        self._data = data
        self._view = view
        self.update_full()

    def update_full(self):
        """Mark the whole data for upload."""
        self._chunk_mask.fill(True)
        self._chunks_dirt_flag = 2
        Resource._rev += 1
        self._rev = Resource._rev
        self._gfx_mark_for_sync()

    def update_indices(self, indices_x, indices_y, indices_z):
        """Mark specific indices for upload.

        The given arrays represent the indices for x, y, and z, respectively.
        So they must be equal in length, similar to what ``np.where()`` returns.
        They can also be None, to indicate the full range, like ":" does with slicing.
        """
        full_slice = slice(None, None, None)
        div = self._chunk_size
        indices_x = (
            full_slice if indices_x is None else (np.asarray(indices_x) // div[0])
        )
        indices_y = (
            full_slice if indices_y is None else (np.asarray(indices_y) // div[1])
        )
        indices_z = (
            full_slice if indices_z is None else (np.asarray(indices_z) // div[2])
        )
        self._chunk_mask[indices_z, indices_y, indices_x] = True
        self._chunks_dirt_flag = 1
        Resource._rev += 1
        self._rev = Resource._rev
        self._gfx_mark_for_sync()

    def update_range(self, offset, size):
        """Mark a certain range of the data for upload to the GPU.
        The offset and (sub) size should be (width, height, depth)
        tuples. Numpy users beware that an arrays shape is (height, width)!
        """
        full_size = self.size
        # Check input
        assert isinstance(offset, tuple) and len(offset) == 3
        assert isinstance(size, tuple) and len(size) == 3
        if any(s == 0 for s in size):
            return
        elif any(s < 0 for s in size):
            raise ValueError("Update size must not be negative")
        elif any(b < 0 for b in offset):
            raise ValueError("Update offset must not be negative")
        # offset + size being larger than the full size is tolaterated.
        # Get indices
        div = self._chunk_size
        index_a = tuple(floor(offset[i] / div[i]) for i in range(3))
        index_b = tuple(
            ceil(min(full_size[i], offset[i] + size[i]) / div[i]) for i in range(3)
        )
        # Update map
        self._chunk_mask[
            index_a[2] : index_b[2], index_a[1] : index_b[1], index_a[0] : index_b[0]
        ] = True
        self._chunks_dirt_flag = 1
        Resource._rev += 1
        self._rev = Resource._rev
        self._gfx_mark_for_sync()

    def _gfx_get_chunk_descriptions(self):
        """Get a list of (offset, size) tuples, that can be
        used in _gfx_get_chunk_data(). This method also clears
        the chunk dirty statuses.
        """

        if not self._chunks_dirt_flag:
            return []
        elif self._chunks_dirt_flag == 2:
            chunk_descriptions = [((0, 0, 0), self.size)]
        elif np.all(self._chunk_mask):
            chunk_descriptions = [((0, 0, 0), self.size)]
        else:
            # Get merged chunk blocks, using a smart algorithm.
            chunk_blocks = get_merged_blocks_from_mask_3d(self._chunk_mask)

            # Turn into proper descriptions, with chunk indices/counts scaled with the chunk size.
            chunk_descriptions = []
            chunk_size = self._chunk_size
            full_size = self._store["size"]
            for block in chunk_blocks:
                offset = (
                    block.x * chunk_size[0],
                    block.y * chunk_size[1],
                    block.z * chunk_size[2],
                )
                size = (
                    min(block.nx * chunk_size[0], full_size[0] - offset[0]),
                    min(block.ny * chunk_size[1], full_size[1] - offset[1]),
                    min(block.nz * chunk_size[2], full_size[2] - offset[2]),
                )
                chunk_descriptions.append((offset, size))

        # Reset
        self._chunks_dirt_flag = False
        self._chunk_mask.fill(False)

        return chunk_descriptions

    def _gfx_get_chunk_data(self, offset, size, pad_value=0.0):
        """Return subdata as a contiguous array."""
        full_size = self._store["size"]
        emulate_rgb = self._wgpu_emulate_rgb

        # Get chunk
        if offset == (0, 0, 0) and size == full_size and not emulate_rgb:
            chunk = self._view
        else:
            # Calculate slice
            slice_per_dim = []
            for d in reversed(range(3)):
                slice_per_dim.append(slice(offset[d], offset[d] + size[d]))
            slice_per_dim = tuple(slice_per_dim)
            # Slice, pad, make contiguous. Note that the chance of the chunks not being contiguous is pretty big.
            chunk = self._view[slice_per_dim]
            if emulate_rgb:
                if self._force_contiguous:
                    logger.warning(
                        "force_contiguous was set, but still need pixel padding"
                    )
                # Copy into array with one extra channel
                nchannels = chunk.shape[3]  # usually 3 for rgb, padded to make rgba
                padded_shape = chunk.shape[:3] + (nchannels + 1,)
                padded_chunk = np.full(padded_shape, pad_value, dtype=chunk.dtype)
                padded_chunk[:, :, :, :nchannels] = chunk
                chunk = padded_chunk

        # Normalize chunk
        if not is_little_endian(chunk):
            chunk = make_little_endian(chunk)
            if self._force_contiguous:
                logger.warning(
                    "force_contiguous was set, but chunk data is still big endian"
                )
        elif not chunk.flags.c_contiguous:
            # Easily happens when slicing in anything but the last dimension
            chunk = np.ascontiguousarray(chunk)

        return chunk


def size_from_array(data, dim):
    # Check if shape matches dimension
    shape = data.shape

    if len(shape) not in (dim, dim + 1):
        raise ValueError(
            f"Can't map shape {shape} on {dim}D tex. Maybe also specify size?"
        )
    # Determine size based on dim and shape
    if dim == 1:
        return shape[0], 1, 1
    elif dim == 2:
        return shape[1], shape[0], 1
    else:  # dim == 3:
        return shape[2], shape[1], shape[0]


def reshape_array(view, size):
    """Reshape array so it's shape[0:3] match size."""
    expected_shape = tuple(reversed(size))
    if expected_shape != view.shape[:3]:
        # This can fail if the data is not contiguous and strides don't work out.
        view.shape = expected_shape + (-1,)
