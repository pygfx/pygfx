import numpy as np

from ._base import Resource, get_item_format_from_memoryview


class Texture(Resource):
    """Texture object containing structured 1D, 2D or 3D data.

    Can be used to represent e.g. image data or colormaps. Can also
    serve as a render target (for the renderer). Supports texture
    stacks, cube textures, and mipmapping.

    Parameters:
        data : array, optional
            Array data of any type that supports the buffer-protocol, (e.g. a
            bytes or numpy array). If None, nbytes and nitems must be provided.
            The dtype must be compatible with the rendering backend.
        dim : int
            The dimensionality of the array (1, 2 or 3).
        size : tuple, [3]
            The extent ``(width, height, depth)`` of the array. If None, it is
            derived from `dim` and the shape of the data. The texture can also
            represent a stack of images by setting `dim=2` and `depth > 1`,
            or a cube image by setting `dim=2` and `depth==6`.
        format : str
            A format string describing the texture layout. If None, this is
            derived from the data. This must be a pygfx format
            specifier, e.g. "3xf4", but can also be a format specific to the
            render backend (e.g. from ``wgpu.TextureFormat``).
        colorspace : str
            If this data is used as color, it is interpreted to be in this
            colorspace. Can be "srgb" or "physical". Default "srgb".
        generate_mipmaps : bool
            If True, automatically generates mipmaps when transferring data to
            the GPU.

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
    ):
        super().__init__()
        Resource._rev += 1
        self._rev = Resource._rev
        # The dim specifies the texture dimension
        assert dim in (1, 2, 3)
        self._store.dim = int(dim)
        # The actual data (optional)
        self._data = None
        self._format = None
        self._gfx_pending_uploads = []  # list of (offset, size) tuples

        # Backends-specific attributes for internal use
        self._wgpu_object = None
        self._wgpu_usage = 0
        self._wgpu_mip_level_count = 1

        self._colorspace = (colorspace or "srgb").lower()
        assert self._colorspace in ("srgb", "physical")

        self._generate_mipmaps = bool(generate_mipmaps)

        size = None if size is None else (int(size[0]), int(size[1]), int(size[2]))

        if data is not None:
            self._data = data
            self._mem = mem = memoryview(data)
            self._store.nbytes = mem.nbytes
            self._store.size = self._size_from_data(mem, dim, size)
            subformat = get_item_format_from_memoryview(mem)
            if subformat is None:
                raise ValueError(
                    f"Unsupported dtype/format for texture data: {mem.format}"
                )
            shape = mem.shape
            collapsed_size = [x for x in self.size if x > 1]
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
            self._format = (f"{nchannels}x" + subformat).lstrip("1x")
            self.update_range((0, 0, 0), self.size)
        elif size is not None and format is not None:
            self._store.size = size
            self._store.nbytes = 0
        else:
            raise ValueError(
                "Texture must be instantiated with either data or size and format."
            )

        if format is not None:
            self._format = str(format)

    @property
    def rev(self):
        """An integer that is increased when update_range() is called."""
        return self._rev

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
        """The texture format as a string. Usually a pygfx format specifier
        (e.g. u2 for scalar uint16, or 3xf4 for RGB float32),
        but can also be a overridden to a backend-specific format.
        """
        return self._format

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
        # If this is a full range, this is easy
        if offset == 0 and size == self.nitems and self.mem.contiguous:
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
