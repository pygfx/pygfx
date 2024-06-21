from math import floor, ceil

import numpy as np

from ._base import Resource, get_item_format_from_memoryview


class Texture(Resource):
    """The Texture represents structured 1D, 2D or 3D data on the GPU.

    A texture can be used to represent e.g. image data or colormaps. They can
    also serve as a render target (for the renderer). Supports texture stacks,
    cube textures, and mipmapping.

    Parameters
    ----------
    data : array | None
        The initial data of the texture. It must support the buffer-protocol,
        (e.g. a bytes or numpy array). If None, ``size`` and ``format must be
        provided. The data will be accessible at ``buffer.data``, no copies are
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
    chunksize : None | int
        The chunksize to use for uploading data to the GPU, expressed in bytes.
        When None (default) an optimal chunksize is determined automatically.
    force_contiguous : bool
        When set to true, the texture goes into a stricter mode, forcing set data
        to be c_contiguous. This ensures optimal upload performance for cases when
        the data changes often.
    usage : int | wgpu.TextureUsage
        The wgpu ``usage`` flag for this texture. Optional: typically pygfx can
        derive how the texture is used and apply the appropriate flag. In cases
        where it doesn't this param provides an override. This is a bitmask flag
        (values are OR'd).
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
        chunksize=None,
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

        self._gfx_pending_uploads = []

        # Process data
        if data is not None:
            self._data = data
            self._mem = mem = memoryview(data)
            if self._force_contiguous and not mem.c_contiguous:
                raise ValueError(
                    "Given texture data is not c_contiguous (enforced because force_contiguous is set)."
                )
            the_nbytes = mem.nbytes
            the_size = self._size_from_data(mem, dim, size)
            subformat = get_item_format_from_memoryview(mem)
            if subformat is None:
                raise ValueError(
                    f"Unsupported dtype/format for texture data: {mem.format}"
                )
            shape = mem.shape
            collapsed_size = [x for x in the_size if x > 1]
            if len(shape) == len(collapsed_size) + 1:
                nchannels = shape[-1]
            else:
                if not len(shape) == len(collapsed_size):
                    raise ValueError(
                        "Incompatible data shape for image data, there must be > 1 pixel to draw per channel"
                    )
                nchannels = 1
            if not (1 <= nchannels <= 4):
                raise ValueError(
                    f"Expected 1-4 texture color channels, got {nchannels}."
                )
            format = (f"{nchannels}x" + subformat).lstrip("1x")
        elif size is not None and format is not None:
            the_size = size
            the_nbytes = 0
        else:
            raise ValueError(
                "Texture must be instantiated with either data or size and format."
            )

        # Store derived props
        self._store.nbytes = the_nbytes
        self._store.size = the_size
        if format is not None:
            self._format = str(format)
        else:
            self._format = None

        # Get optimal chunksize
        if chunksize is None:
            chunksize_bytes = max(min(the_nbytes / 16, 2**20), 2**8)
        else:
            chunksize_bytes = max(float(chunksize), 1)

        # # Init chunks map
        # if data is None:
        #     self._chunks_any_dirty = False
        #     self._chunk_itemsize = 0
        #     self._chunk_map = None
        # elif the_nbytes == 0:
        #     self._chunks_any_dirty = False
        #     self._chunk_itemsize = 0
        #     self._chunk_map = np.ones((0,), bool)
        # else:
        #     self._chunks_any_dirty = True
        #     itemsize = the_nbytes // the_nitems
        #     self._chunk_itemsize = ceil(chunksize_bytes / itemsize)
        #     self._chunk_itemsize = max(min(self._chunk_itemsize, the_nitems), 1)
        #     n_chunks = ceil(the_nitems / self._chunk_itemsize)
        #     self._chunk_map = np.ones((n_chunks,), bool)

    @property
    def dim(self):
        """The dimensionality of the texture (1, 2, or 3)."""
        return self._store.dim

    @property
    def data(self):
        """The data for this texture. Can be None if the data only
        exists on the GPU.

        Note: the data is the same reference that was given to
        instantiate this object, but this may change.
        """
        return self._data

    @property
    def mem(self):
        """The data for this buffer as a memoryview. Can be None if
        the data only exists on the GPU.
        """
        return self._mem

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
        return self._format

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

    def update_range(self, offset, size):
        """Mark a certain range of the data for upload to the GPU.
        The offset and (sub) size should be (width, height, depth)
        tuples. Numpy users beware that an arrays shape is (height, width)!
        """
        # Check input
        assert isinstance(offset, tuple) and len(offset) == 3
        assert isinstance(size, tuple) and len(size) == 3
        if any(s == 0 for s in size):
            return
        elif any(s < 0 for s in size):
            raise ValueError("Update size must not be negative")
        elif any(b < 0 for b in offset):
            raise ValueError("Update offset must not be negative")
        elif any(b + s > refsize for b, s, refsize in zip(offset, size, self.size)):
            raise ValueError("Update size out of range")
        # Apply - consider that texture arrays want to be uploaded per-texture
        # todo: avoid duplicates by merging with existing pending uploads
        if self.dim == 1:
            for z in range(size[2]):
                for y in range(size[1]):
                    offset2 = offset[0], y, z
                    size2 = size[0], 1, 1
                    self._gfx_pending_uploads.append((offset2, size2))
        elif self.dim == 2:
            for z in range(size[2]):
                offset2 = offset[0], offset[1], z
                size2 = size[0], size[1], 1
                self._gfx_pending_uploads.append((offset2, size2))
        else:
            self._gfx_pending_uploads.append((offset, size))
        Resource._rev += 1
        self._rev = Resource._rev
        self._gfx_mark_for_sync()

    def _size_from_data(self, data, dim, size):
        # Check if shape matches dimension
        shape = data.shape

        if size:
            # Get version of size with trailing ones stripped
            size2 = size
            size2 = size2[:-1] if size2[-1] == 1 else size2
            size2 = size2[:-1] if size2[-1] == 1 else size2
            rsize = tuple(reversed(size2))
            # Check if size matches shape
            if rsize != shape[: len(rsize)]:
                raise ValueError(f"Given size does not match the data shape.")
            return size
        else:
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

    def _get_subdata(self, offset, size, pixel_padding=None):
        """Return subdata as a contiguous array."""
        # If this is a full range, this is easy (and fast)
        if (
            offset == (0, 0, 0)
            and size == self.size
            and self.mem.c_contiguous
            and pixel_padding is None
        ):
            return self.mem
        # Get a numpy array, because memoryviews do not support nd slicing
        if isinstance(self.data, np.ndarray):
            arr = self.data
        elif not self.mem.c_contiguous:
            raise ValueError(
                "Non-contiguous texture data is only supported for numpy array."
            )
        else:
            arr = np.frombuffer(self.mem, self.mem.format)
        arr = arr.reshape(self.size[2], self.size[1], self.size[0], -1)
        # Slice it
        slices = []
        for d in reversed(range(3)):
            slices.append(slice(offset[d], offset[d] + size[d]))
        sub_arr = arr[tuple(slices)]
        if pixel_padding is not None:
            padding = np.ones(sub_arr.shape[:3] + (1,), dtype=sub_arr.dtype)
            sub_arr = np.concatenate([sub_arr, pixel_padding * padding], -1)
        return memoryview(np.ascontiguousarray(sub_arr))

    def get_view(self, *args, **kwargs):
        raise DeprecationWarning(
            "Texture.get_view() is removed, TextureView is no longer public API: just use plain textures."
        )
