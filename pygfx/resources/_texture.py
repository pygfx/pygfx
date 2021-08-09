import numpy as np

from ._buffer import Resource, STRUCT_FORMAT_ALIASES

# todo: what to do about these enums from wgpu. Copy them over?


class Texture(Resource):
    """A base texture wrapper that can be implemented for numpy, ctypes arrays,
    or any other kind of array.

    Parameters:
        data (array, optional): Array data of any type that supports the
            buffer-protocol, (e.g. a bytes or numpy array). If not given
            or None, nbytes and nitems must be provided. The data is
            copied if it's float64 or not contiguous.
        dim (int): The dimensionality of the array (1, 2 or 3).
        usage: The way(s) that the texture will be used. Default "TEXTURE_BINDING",
            set/add "STORAGE_BINDING" if you're using it as a storage texture
            (see wgpu.TextureUsage).
        size (3-tuple): The extent ``(width, height, depth)`` of the array.
            If not given or None, it is derived from dim and the shape of
            the data. By creating a 2D array with ``depth > 1``, a view can
            be created with format 'd2_array' or 'cube'.
        format (enum str): the GPU format of texture. Must be a value from
            wgpu.TextureFormat. By default it is derived from the data. Set when
            data is not given or when you want to overload the derived value.
    """

    def __init__(
        self, data=None, *, dim, usage="TEXTURE_BINDING", size=None, format=None
    ):
        self._rev = 0
        # The dim specifies the texture dimension
        assert dim in (1, 2, 3)
        self._dim = int(dim)
        # The size specifies the size on the GPU (width, height, depth)
        self._size = ()
        self._format = None if format is None else str(format)
        self._nbytes = 0
        # The actual data (optional)
        self._data = None
        self._pending_uploads = []  # list of (offset, size) tuples

        size = None if size is None else (int(size[0]), int(size[1]), int(size[2]))

        if data is not None:
            mem = memoryview(data)
            if mem.format == "d":
                raise ValueError("Float64 data is not supported, use float32 instead.")
            self._data = data
            self._mem = mem
            self._nbytes = mem.nbytes
            self._size = self._size_from_data(mem, dim, size)
            self.update_range((0, 0, 0), self._size)
        elif size is not None and format is not None:
            self._size = size
        else:
            raise ValueError(
                "Texture must be instantiated with either data or size and format."
            )

        # Determine usage
        if isinstance(usage, str):
            usages = usage.upper().replace(",", " ").replace("|", " ").split()
            assert usages
            self._usage = "|".join(usages)
        else:
            raise TypeError("Texture usage must be str.")

    @property
    def rev(self):
        """An integer that is increased when update_range() is called."""
        return self._rev

    def get_view(self, **kwargs):
        """Get a new view on the this texture."""
        return TextureView(self, **kwargs)

    @property
    def dim(self):
        """The dimensionality of the texture (1, 2, or 3)."""
        return self._dim

    @property
    def usage(self):
        """The texture usage flags as a string (compatible with the
        wgpu.TextureUsage enum).
        """
        return self._usage

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
        return self._nbytes

    @property
    def size(self):
        """The size of the texture as (width, height, depth).
        (always a 3-tuple, regardless of the dimension).
        """
        return self._size

    @property
    def format(self):
        """The texture format as a string (compatible with the
        wgpu.TextureFormat enum).
        """
        if self._format is not None:
            return self._format
        elif self.usage == "UNIFORM":
            return None
        elif self.data is not None:
            self._format = format_from_memoryview(self.mem, self.size)
            return self._format
        else:
            raise ValueError("Texture has no data nor format.")

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
                    self._pending_uploads.append((offset2, size2))
        elif self.dim == 2:
            for z in range(size[2]):
                offset2 = offset[0], offset[1], z
                size2 = size[0], size[1], 1
                self._pending_uploads.append((offset2, size2))
        else:
            self._pending_uploads.append((offset, size))
        self._rev += 1

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


def format_from_memoryview(mem, size):
    format = str(mem.format)
    format = STRUCT_FORMAT_ALIASES.get(format, format)
    # Process channels
    shape = mem.shape
    collapsed_size = [x for x in size if x > 1]
    if len(shape) == len(collapsed_size) + 1:
        nchannels = shape[-1]
    else:
        assert len(shape) == len(collapsed_size)
        nchannels = 1
    assert 1 <= nchannels <= 4
    tex_format = [None, "r", "rg", "rgb", "rgba"][nchannels]
    # if tex_format == "rgb":
    #     -> no raise: WGPU does not support rgb, but we handle it in the renderer
    # Process dtype. We select the tex_format that matches the dtype.
    # This means that uint8 values become 0..255 in the shader.
    # todo: not yet entirely sure about this
    # todo: there is no reference to wgpu here, but these are wgpu enums. Is that ok?
    texformatmap = {
        "b": "8snorm",
        "B": "8unorm",
        # "b": "8sint",
        # "B": "8uint",
        "h": "16sint",
        "H": "16uint",
        "i": "32sint",
        "U": "32uint",
        "e": "16float",
        "f": "32float",
    }
    if format in ("d", "float64"):
        raise TypeError("GPU's don't support float64 texture formats.")
    elif format not in texformatmap:
        raise TypeError(
            f"Cannot convert {format!r} to texture format. Maybe specify format?"
        )
    tex_format += texformatmap[format]
    return tex_format


# mipmaps: every texture can have a certain number of mipmap levels. Each
# next level is half the size of the previous level. I think we can design our
# API to target level 0 by default and allow a way to upload data to other levels.
# arrays: a d2_array view can be be created from a d2 texture with dept > 1
# cube: a special kind3 of array texture, with six 2D textures.
# cube_array: I suppose you'd have an array of 6xn textures in this case?


class TextureView(Resource):
    """A view on a texture.

    The view defines the sampling behavior and can specify a selection/different
    view on the texture.

    Passing no more than ``address_mode`` and ``filter`` will create a
    default view on the texture with the given sampling parameters.

    Parameters:
        address_mode (str): How to sample beyond the edges. Use "clamp",
            "mirror" or "repeat". Default "clamp".
            Can also use e.g. "clamp,clamp,repeat" to specify for u, v and w.
        filter (str): Interpolation filter. Use "nearest" or "linear".
            Default "nearest". Can also use e.g. "linear,linear,nearest" to set
            mag, min and mipmap filters.
        format (str): Omit or pass None to use the texture's format.
        view_dim (str): Omit or pass None to use the texture's format. Or e.g.
            get a "2d" slice view from a 3d texture, or e.g. "cube" or "2d-array".
        aspect (str): Omit or pass None to use the default.
        mip_range (range): A range object to specify what mip levels to view.
        layer_range (range): A range object to specify what array layers to view.
    """

    def __init__(
        self,
        texture,
        *,
        address_mode="clamp",
        filter="nearest",
        format=None,
        view_dim=None,
        aspect=None,
        mip_range=None,
        layer_range=None,
    ):
        self._rev = 1
        assert isinstance(texture, Texture)
        self._texture = texture
        # Sampler parameters
        self._address_mode = address_mode
        self._filter = filter
        # Texture view parameters
        self._format = format
        self._view_dim = view_dim
        self._aspect = aspect
        self._mip_range = mip_range or range(1)
        self._layer_range = layer_range or range(1)
        self._is_default_view = all(
            x is None for x in [format, view_dim, aspect, mip_range, layer_range]
        )

    @property
    def rev(self):
        # This is not actually increased anywhere, but it's added for consistency
        return self._rev

    @property
    def texture(self):
        """The Texture object holding the data for this texture view."""
        return self._texture

    @property
    def format(self):
        """The texture format."""
        return self._format or self.texture.format

    @property
    def view_dim(self):
        """The dimensionality of this view, as a string.
        See wgpu.TextureViewDimension.
        """
        return self._view_dim or f"{self.texture.dim}d"

    @property
    def address_mode(self):
        """How to sample beyond the edges. Use "clamp",
        "mirror" or "repeat". Default "clamp".
        """
        return self._address_mode

    @property
    def filter(self):
        """Interpolation filter. Use "nearest" or "linear"."""
        return self._filter
