import numpy as np

import wgpu

STRUCT_FORMAT_ALIASES = {"c": "B", "l": "i", "L": "I"}


class Resource:
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
        usage (str): The way(s) that the texture will be used. E.g. "INDEX"
            "VERTEX", "UNIFORM". Multiple values can be given separated
            with "|". See wgpu.BufferUsage.
        nbytes (int): The number of bytes. If data is given, it is derived.
        nitems (int): The number of items. If data is given, it is derived.
        format (str): The format to use when used as a vertex buffer.
            Must be a value from wgpu.VertexFormat. By default it is
            derived from the data. Set when data is not given or when
            you want to overload the derived value.
    """

    def __init__(
        self,
        data=None,
        *,
        usage,
        nbytes=None,
        nitems=None,
        format=None,
    ):
        self._rev = 0
        # To specify the buffer size
        self._nbytes = 0
        self._nitems = 1
        self._format = format
        # The actual data (optional)
        self._data = None
        self._pending_uploads = []  # list of (offset, size) tuples

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
            self._nbytes = mem.nbytes
            self._nitems = mem.shape[0] if mem.shape else 1
            self._pending_uploads.append((0, self._nitems))
            if nbytes is not None and nbytes != self._nbytes:
                raise ValueError("Given nbytes does not match size of given data.")
            if nitems is not None and nitems != self._nitems:
                raise ValueError("Given nitems does not match shape of given data.")
        elif nbytes is not None and nitems is not None:
            self._nbytes = int(nbytes)
            self._nitems = int(nitems)
        else:
            raise ValueError(
                "Buffer must be instantiated with either data or nbytes and nitems."
            )

        # Determine usage
        if isinstance(usage, str):
            usages = usage.upper().replace(",", " ").replace("|", " ").split()
            assert usages
            self._usage = "|".join(usages)
        else:
            raise TypeError("Buffer usage must be str.")

        # We can use a subset when used as a vertex buffer
        self._vertex_byte_range = (0, self._nbytes)

    @property
    def rev(self):
        """An integer that is increased when update_range() is called."""
        return self._rev

    @property
    def usage(self):
        """The buffer usage flags (as a string with "|" as a separator)."""
        return self._usage

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
        return self._nbytes

    @property
    def nitems(self):
        """The number of items in the buffer."""
        return self._nitems

    @property
    def format(self):
        """The vertex or index format (depending on the value of usage)."""
        if self._format is not None:
            return self._format
        elif self.data is not None:
            self._format = format_from_memoryview(self.mem, self.usage)
            return self._format
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

    def update_range(self, offset=0, size=2 ** 50):
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
            offset = min(offset, cur_offset)
            size = max(size, cur_size)
        # Limit and apply
        self._pending_uploads.append((offset, size))
        self._rev += 1
        # todo: this can be smarter, we have logic for chunking in the morph tool

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


def format_from_memoryview(mem, usage):

    if "INDEX" in usage:

        format = str(mem.format)
        format = STRUCT_FORMAT_ALIASES.get(format, format)
        mapping = {
            "h": wgpu.IndexFormat.uint16,
            "H": wgpu.IndexFormat.uint16,
            "i": wgpu.IndexFormat.uint32,
            "I": wgpu.IndexFormat.uint32,
        }
        try:
            return mapping[format]
        except KeyError:
            raise TypeError(
                f"Need 16bit or 32bit signed/unsigned int (hHiI) for index data, not '{format}'."
            )

    else:  # if "VERTEX" in self.usage:

        shape = mem.shape
        if len(shape) == 1:
            shape = shape + (1,)
        assert len(shape) == 2
        format = str(mem.format)
        format = STRUCT_FORMAT_ALIASES.get(format, format)
        key = format, shape[-1]
        mapping = {
            ("f", 1): wgpu.VertexFormat.float32,
            ("f", 2): wgpu.VertexFormat.float32x2,
            ("f", 3): wgpu.VertexFormat.float32x3,
            ("f", 4): wgpu.VertexFormat.float32x4,
            #
            ("e", 2): wgpu.VertexFormat.float16x2,
            ("e", 4): wgpu.VertexFormat.float16x4,
            #
            ("b", 2): wgpu.VertexFormat.sint8x2,
            ("b", 4): wgpu.VertexFormat.sint8x4,
            ("B", 2): wgpu.VertexFormat.uint8x2,
            ("B", 4): wgpu.VertexFormat.uint8x4,
            #
            ("h", 2): wgpu.VertexFormat.sint16x2,
            ("h", 4): wgpu.VertexFormat.sint16x4,
            ("H", 2): wgpu.VertexFormat.uint16x2,
            ("H", 4): wgpu.VertexFormat.uint16x4,
            #
            ("i", 1): wgpu.VertexFormat.sint32,
            ("i", 2): wgpu.VertexFormat.sint32x2,
            ("i", 3): wgpu.VertexFormat.sint32x3,
            ("i", 4): wgpu.VertexFormat.sint32x4,
            #
            ("I", 1): wgpu.VertexFormat.uint32,
            ("I", 2): wgpu.VertexFormat.uint32x2,
            ("I", 3): wgpu.VertexFormat.uint32x3,
            ("I", 4): wgpu.VertexFormat.uint32x4,
        }
        try:
            return mapping[key]
        except KeyError:
            if format in ("d", "float64"):
                raise ValueError(
                    "64-bit float is not supported, use 32-bit float instead"
                )
            raise ValueError(f"Invalid format/shape for vertex data: {key}")
