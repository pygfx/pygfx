import numpy as np

from ._buffer import Resource, STRUCT_FORMAT_ALIASES


class Texture(Resource):
    """Container for textures.

    A base texture wrapper that can be implemented for numpy, ctypes arrays, or
    any other kind of array.

    Parameters:
        data : array, optional
            Array data of any type that supports the buffer-protocol, (e.g. a
            bytes or numpy array). If None, nbytes and nitems must be provided.
            The data is copied if it's not float32 or not contiguous.
        dim : int
            The dimensionality of the array (1, 2 or 3).
        size : tuple, [3]
            The extent ``(width, height, depth)`` of the array. If None, it is
            derived from `dim` and the shape of the data. The texture can also
            represent a stack of images by setting `dim=2` and `depth > 1`. Any
            derived texture views must then have a `view_dim` of  either
            'd2_array' or 'cube'.
        format : str
            A format string describing the texture layout. If None, this is
            automatically set from the data. This must be a pygfx format
            specifier, e.g. "3xf4", but can also be a format specific to the
            render backend if necessary (e.g. from ``wgpu.VertexFormat``).
        colorspace : str
            If this data is used as color, it is interpreted to be in this
            colorspace. Can be "srgb" or "physical".
        generate_mipmaps : bool
            If True, automatically generates mipmaps when transfering data to
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
        self._rev = 0
        # The dim specifies the texture dimension
        assert dim in (1, 2, 3)
        self._store.dim = int(dim)
        # The actual data (optional)
        self._data = None
        self._pending_uploads = []  # list of (offset, size) tuples
        self._mip_level_count = 1

        # Backends-specific attributes for internal use
        self._wgpu_usage = 0

        self._store.format = None if format is None else str(format)

        self._colorspace = (colorspace or "srgb").lower()
        assert self._colorspace in ("srgb", "physical")

        self._generate_mipmaps = bool(generate_mipmaps)

        size = None if size is None else (int(size[0]), int(size[1]), int(size[2]))

        if data is not None:
            mem = memoryview(data)
            if mem.format == "d":
                raise ValueError("Float64 data is not supported, use float32 instead.")
            self._data = data
            self._mem = mem
            self._store.nbytes = mem.nbytes
            self._store.size = self._size_from_data(mem, dim, size)
            self.update_range((0, 0, 0), self.size)
        elif size is not None and format is not None:
            self._store.size = size
            self._store.nbytes = 0
        else:
            raise ValueError(
                "Texture must be instantiated with either data or size and format."
            )

    @property
    def rev(self):
        """An integer that is increased when update_range() is called."""
        return self._rev

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

    def get_view(self, **kwargs):
        """Get a new view on the this texture."""
        return TextureView(self, **kwargs)

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
        but can also be a overriden to a backend-specific format.
        """
        format = self._store.format
        if format is not None:
            return format
        elif self.data is not None:
            self._store["format"] = format_from_memoryview(self.mem, self.size)
            return self._store.format
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
    formatmap = {
        "b": "i1",
        "B": "u1",
        "h": "i2",
        "H": "u2",
        "i": "i4",
        "U": "u4",
        "e": "f2",
        "f": "f4",
    }

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
    if format in ("d", "float64"):
        raise TypeError("GPU's don't support float64 texture formats.")
    elif format not in formatmap:
        raise TypeError(
            f"Cannot convert {format!r} to texture format. Maybe specify format?"
        )
    format = f"{nchannels}x" + formatmap[format]
    return format.lstrip("1x")


# mipmaps: every texture can have a certain number of mipmap levels. Each
# next level is half the size of the previous level. I think we can design our
# API to target level 0 by default and allow a way to upload data to other levels.
# arrays: a d2_array view can be be created from a d2 texture with dept > 1
# cube: a special kind3 of array texture, with six 2D textures.
# cube_array: I suppose you'd have an array of 6xn textures in this case?


class TextureView(Resource):
    """View into a Texture.

    Similar to numpy views, a TextureView is a view into a textures data buffer
    with a (potentially) modified sampling behavior and can specify a
    selection/different view on the texture.

    Passing no more than ``address_mode`` and ``filter`` will create a default
    view on the texture with the given sampling parameters.

    Parameters
    ----------
    texture : Texture
        The texture to view.
    address_mode : str
        How to handle out of bounds access. Use "clamp", "mirror" or "repeat".
        This value can also be set per-channel by providing multiple
        comma-separated values, e.g., "clamp,clamp,repeat".
    filter : str
        Interpolation mode. Possible values are: "nearest" or "linear". This
        value can also be set using three comma-separated values to control mag,
        min and mipmap filters individually, e.g., "linear,linear,nearest".
    format : str
        A format string describing the texture layout. If None, use the same as
        the underlying texture. This must be a pygfx format specifier, e.g.
        "3xf4", but can also be a format specific to the render backend if
        necessary (e.g. from ``wgpu.VertexFormat``).
    view_dim : str
        The dimensionality of the array (1, 2 or 3). If None, use the texture's
        dimension. Or e.g. get a "2d" slice view from a 3d texture, or e.g.
        "cube" or "2d-array".
    aspect : Enum
        The `wgpu.TextureAspect` for this view. Omit or pass None to use the default.
    mip_range : range
        A range object specifying the viewed mip levels.
    layer_range : range
        A range object specifying the array layers to view.

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
        super().__init__()
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
        # The ranges
        if mip_range:
            assert isinstance(mip_range, range)
            assert mip_range.step == 1
            self._mip_range = mip_range
        else:
            self._mip_range = None  # None means all mip levels in the texture are used
        if layer_range:
            assert isinstance(layer_range, range)
            assert layer_range.step == 1
            self._layer_range = layer_range
        else:
            self._layer_range = range(texture.size[2])

        self._is_default_view = all(
            x is None for x in [format, view_dim, aspect, mip_range, layer_range]
        )

    @property
    def rev(self):
        # This is not actually increased anywhere, but it's added for consistency
        return self._rev

    @property
    def colorspace(self):
        """Proxy for the texture's colorspace property."""
        return self._texture.colorspace

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
        """The dimensionality of this view: "1d", "2d" or "3d"."""
        return self._view_dim or f"{self.texture.dim}d"

    @property
    def mip_range(self):
        """The range of mip levels to view, as a range object.
        The step is always 1.
        """
        return self._mip_range or range(
            self.texture._mip_level_count
        )  # "texture._mip_level_count" may be updated in auto mipmap generation

    @property
    def layer_range(self):
        """The range of array layers to view, as a range object.
        The step is always 1.
        """
        return self._layer_range

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
