import ctypes

import numpy as np


# todo: what to do about these enums from wgpu. Copy them over?


class BaseTexture:
    """ A base texture wrapper that can be implemented for numpy, ctypes arrays,
    or any other kind of array.

    Parameters:
        data (ndarray, optional): The array data as an nd array.
        dim (int): The dimensionality of the array (1, 2 or 3).
        usage: The way(s) that the texture will be used. Default "SAMPLED",
            set/add "STORAGE" if you're using it as a storage texture
            (see wgpu.TextureUsage).
        size (3-tuple): The extent ``(width, height, depth)`` of the array.
            If not given or None, it is derived from dim and the shape of
            the data. By creating a 2D array with ``depth > 1``, a view can
            be created with format 'd2_array' or 'cube'.
        format (enum str): the GPU format of texture. If not given or None,
            it is derived from the given data dtype. Otherwise, provide a
            value from wgpu.TextureFormat.
            THIS IS NOT TRUE. we need to decided what happens with uint8
    """

    def __init__(self, data=None, *, dim, usage="SAMPLED", size=None, format=None):
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
            self._data = data
            self._size = self._size_from_data(data, dim, size)
            self._nbytes = self._nbytes_from_data(data)
            self._pending_uploads.append(((0, 0, 0), self._size))
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
            raise TypeError("Buffer usage must be str.")

    @property
    def dirty(self):
        """ Whether the buffer is dirty (needs to be processed by the renderer).
        """
        return bool(self._pending_uploads)

    def get_view(self, **kwargs):
        """ Get a new view on the this texture.
        """
        return TextureView(self, **kwargs)

    @property
    def dim(self):
        """ The dimensionality of the texture (1, 2, or 3).
        """
        return self._dim

    @property
    def usage(self):
        """ The texture usage flags as a string (compatible with the
        wgpu.TextureUsage enum).
        """
        return self._usage

    @property
    def nbytes(self):
        """ Get the number of bytes in the texture.
        """
        return self._nbytes

    @property
    def size(self):
        """ The size of the texture as (width, height, depth).
        (always a 3-tuple, regardless of the dimension).
        """
        return self._size

    @property
    def format(self):
        """ The texture format as a string (compatible with the
        wgpu.TextureFormat enum).
        """
        if self._format is not None:
            return self._format
        elif self.data is not None:
            self._format = self._format_from_data(self.data)
            return self._format
        else:
            raise ValueError("Buffer has no data nor format.")

    @property
    def data(self):
        """ The data that is a view on the data. Can be None if the
        data only exists on the GPU.
        """
        return self._data

    def update_range(self, offset, size):
        """ Mark a certain range of the data for upload to the GPU.
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
        # Merge with current entry?
        if self._pending_uploads:
            cur_offset, cur_size = self._pending_uploads.pop(-1)
            offset = tuple(min(offset[i], cur_offset[i]) for i in range(3))
            size = tuple(max(size[i], cur_size[i]) for i in range(3))
        # Apply
        self._pending_uploads.append((offset, size))

    # To implement in subclasses

    def _nbytes_from_data(self, data):
        raise NotImplementedError()

    def _size_from_data(self, data, dim, size):
        raise NotImplementedError()

    def _format_from_data(self, data):
        raise NotImplementedError()

    def _renderer_copy_data_to_ctypes_object(self, ob, offset, size):
        """ Allows renderer to efficiently copy the data.
        """
        raise NotImplementedError()


class Texture(BaseTexture):  # numpy-based
    """ Object that wraps a (GPU) texture object, optionally providing data
    for it, and optionally *mapping* the data so it's shared. But you can also
    use it as a placeholder for a texture with no representation on the CPU.
    """

    def _nbytes_from_data(self, data):
        return data.nbytes

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

    def _format_from_data(self, data):
        dtype = data.dtype
        # Process channels
        shape = data.shape
        collapsed_size = [x for x in self._size if x > 1]
        if len(shape) == len(collapsed_size) + 1:
            nchannels = shape[-1]
        else:
            assert len(shape) == len(collapsed_size)
            nchannels = 1
        assert 1 <= nchannels <= 4
        format = [None, "r", "rg", "rgb", "rgba"][nchannels]
        if format == "rgb":
            raise ValueError("RGB textures not supported, use RGBA instead")
        # Process dtype. We select the format that matches the dtype.
        # This means that uint8 values become 0..255 in the shader.
        # todo: not yet entirely sure about this
        # todo: there is no reference to wgpu here, but these are wgpu enums. Is that ok?
        formatmap = {
            # "int8": "8snorm",
            # "uint8": "8unorm",
            "int8": "8sint",
            "uint8": "8uint",
            "int16": "16sint",
            "uint16": "16uint",
            "int32": "32sint",
            "uint32": "32uint",
            "float16": "16float",
            "float32": "32float",
        }
        if dtype == np.float64:
            raise TypeError("GPU's don't support float64 texture formats.")
        elif dtype.name not in formatmap:
            raise TypeError(
                f"Cannot convert {dtype} to texture format. Maybe specify format?"
            )
        format += formatmap[dtype.name]
        return format

    def _renderer_copy_data_to_ctypes_object(self, ob, offset, size):
        # todo: double check that we don't make unnecessary copies here
        slices = []
        for d in reversed(range(self.dim)):
            slices.append(slice(offset[d], offset[d] + size[d]))
        subdata = np.ascontiguousarray(self._data[tuple(slices)])
        nbytes = ctypes.sizeof(ob)
        assert nbytes == subdata.nbytes
        ctypes.memmove(
            ctypes.addressof(ob), subdata.ctypes.data, nbytes,
        )


# mipmaps: every texture can have a certain number of mipmap levels. Each
# next level is half the size of the previous level. I think we can design our
# API to target level 0 by default and allow a way to upload data to other levels.
# arrays: a d2_array view can be be created from a d2 texture with dept > 1
# cube: a special kind3 of array texture, with six 2D textures.
# cube_array: I suppose you'd have an array of 6xn textures in this case?


class TextureView:
    """ A view on a texture.

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
        assert isinstance(texture, BaseTexture)
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
    def dirty(self):
        """ Whether this resource needs syncing with the GPU.
        """
        return self._texture.dirty

    @property
    def texture(self):
        """ The Texture object holding the data for this texture view.
        """
        return self._texture

    @property
    def format(self):
        """ The texture format.
        """
        return self._format or self.texture.format

    @property
    def view_dim(self):
        """ The dimensionality of this view, as a string.
        See wgpu.TextureViewDimension.
        """
        return self._view_dim or f"{self.texture.dim}d"

    @property
    def address_mode(self):
        """ How to sample beyond the edges. Use "clamp",
        "mirror" or "repeat". Default "clamp".
        """
        return self._address_mode

    @property
    def filter(self):
        """ Interpolation filter. Use "nearest" or "linear".
        """
        return self._filter
