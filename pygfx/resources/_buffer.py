from ..utils.trackable import Trackable

import numpy as np

STRUCT_FORMAT_ALIASES = {"c": "B", "l": "i", "L": "I"}


class Resource(Trackable):
    pass


class Buffer(Resource):
    """A buffer object represents a piece of memory to the GPU, that can be
    used as index buffer, vertex buffer, uniform buffer, or storage buffer.
    You can provide (and update data for it), or use it as a placeholder
    for a buffer with no representation on the CPU.

    Parameters:
        data (array, optional): Array data of any type that supports the
            buffer-protocol, (e.g. a bytes or numpy array). If not given
            or None, nbytes and nitems must be provided. The data is
            copied if it's float64 or not contiguous.
        nbytes (int): The number of bytes. If data is given, it is derived.
        nitems (int): The number of items. If data is given, it is derived.
        format (str): The format to use when used as a vertex buffer.
            By default this is automatically set from the data. This
            must be a pygfx format specifier, e.g. "3xf4", but can also
            be a format specific to the render backend if necessary
            (e.g. from ``wgpu.VertexFormat``).
    """

    def __init__(
        self,
        data=None,
        *,
        nbytes=None,
        nitems=None,
        format=None,
    ):
        super().__init__()
        self._rev = 0
        # To specify the buffer size
        # The actual data (optional)
        self._data = None
        self._pending_uploads = []  # list of (offset, size) tuples

        # Backends-specific attributes for internal use
        self._wgpu_usage = 0

        # Get nbytes
        if data is not None:
            mem = memoryview(data)
            if mem.format == "d":
                raise ValueError("Float64 data is not supported, use float32 instead.")
            # if not mem.contiguous or mem.format == "d":
            #     format = "f" if mmemformat == "d" else mem.format
            #     x = np.empty(mem.shape, format)
            #     x[:] = mem
            #     mem = memoryview(x)
            self._data = data
            self._mem = mem
            the_nbytes = mem.nbytes
            the_nitems = mem.shape[0] if mem.shape else 1
            self._pending_uploads.append((0, the_nitems))
            if nbytes is not None and nbytes != the_nbytes:
                raise ValueError("Given nbytes does not match size of given data.")
            if nitems is not None and nitems != the_nitems:
                raise ValueError("Given nitems does not match shape of given data.")
        elif nbytes is not None and nitems is not None:
            the_nbytes = int(nbytes)
            the_nitems = int(nitems)
        else:
            raise ValueError(
                "Buffer must be instantiated with either data or nbytes and nitems."
            )

        self._store.nbytes = the_nbytes
        self._store.nitems = the_nitems
        self._store.format = format

        # We can use a subset when used as a vertex buffer
        self._vertex_byte_range = (0, the_nbytes)

    @property
    def rev(self):
        """An integer that is increased when update_range() is called."""
        return self._rev

    @property
    def data(self):
        """The data for this buffer. Can be None if the data only
        exists on the GPU.

        Note: the data is the same reference that was given to instantiate this object,
        but this may change.
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
        """The number of bytes in the buffer."""
        return self._store.nbytes

    @property
    def nitems(self):
        """The number of items in the buffer."""
        return self._store.nitems

    @property
    def format(self):
        """The vertex format. Usually a pygfx format specifier (e.g. u2
        for scalar uint16, or 3xf4 for 3xfloat32), but can also be a
        overriden to a backend-specific format.
        """
        format = self._store.format
        if format is not None:
            return format
        elif self.data is not None:
            self._store["format"] = format_from_memoryview(self.mem)
            return self._store.format
        else:
            raise ValueError("Buffer has no data nor format.")

    @property
    def vertex_byte_range(self):
        """The offset and size, in bytes, when used as a vertex buffer."""
        return self._vertex_byte_range

    @vertex_byte_range.setter
    def vertex_byte_range(self, offset_nbytes):
        offset, nbytes = int(offset_nbytes[0]), int(offset_nbytes[1])
        assert offset >= 0
        assert offset + nbytes <= self.nbytes
        self._vertex_byte_range = offset, nbytes

    def update_range(self, offset=0, size=2**50):
        """Mark a certain range of the data for upload to the GPU. The
        offset and size are expressed in integer number of elements.
        """
        # See ThreeJS BufferAttribute.updateRange
        # Check input
        assert isinstance(offset, int) and isinstance(size, int)
        if size == 0:
            return
        elif size < 0:
            raise ValueError("Update size must not be negative")
        elif offset < 0:
            raise ValueError("Update offset must not be negative")
        elif offset + size > self.nitems:
            size = self.nitems - offset
        # Merge with current entry?
        if self._pending_uploads:
            cur_offset, cur_size = self._pending_uploads.pop(-1)
            end = max(offset + size, cur_offset + cur_size)
            offset = min(offset, cur_offset)
            size = end - offset
        # Limit and apply
        self._pending_uploads.append((offset, size))
        self._rev += 1
        # note: this can be smarter, we have logic for chunking in the morph tool

    def _get_subdata(self, offset, size):
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
            arr = np.frombuffer(self.mem, self.mem.format).reshape(self.mem.shape)
        # Slice it
        sub_arr = arr[offset : offset + size]
        return memoryview(np.ascontiguousarray(sub_arr))


def format_from_memoryview(mem):

    formatmap = {
        "b": "i1",
        "B": "u1",
        "h": "i2",
        "H": "u2",
        "i": "i4",
        "I": "u4",
        "e": "f2",
        "f": "f4",
    }

    shape = mem.shape
    if len(shape) == 1:
        shape = shape + (1,)
    assert len(shape) == 2
    format = str(mem.format)
    format = STRUCT_FORMAT_ALIASES.get(format, format)
    if format in ("d", "float64"):
        raise ValueError("64-bit float is not supported, use 32-bit float instead")
    elif format not in formatmap:
        raise TypeError(
            f"Cannot convert {format!r} to vertex format. Maybe specify format?"
        )
    format = f"{shape[-1]}x" + formatmap[format]
    return format.lstrip("1x")
