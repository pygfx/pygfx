import ctypes

import wgpu


class BaseBuffer:
    """ A base buffer class that does not make assumptions on the kind
    of array (numpy, ctypes or otherwise).

    Parameters:
        data (array): Array data. Optional. If not given, nbytes and nitems
            must be provided, and format if used as an index or vertex buffer.
        usage (str): The way(s) that the texture will be used. E.g. "INDEX"
            "VERTEX", "UNIFORM". Multiple values can be given separated
            with "|". See wgpu.BufferUsage.
        nbytes (int): The number of bytes. If data is given, it is derived.
        nitems (int): The number of items. If data is given, it is derived.
        vertex_format (str): The format to use when used as a vertex buffer.
            If data is given, it is derived. Must be a value from
            wgpu.VertexFormat.
    """

    def __init__(
        self, data=None, *, usage, nbytes=None, nitems=None, format=None,
    ):
        # To specify the buffer size
        self._nbytes = 0
        self._nitems = 1
        self._format = format
        # The actual data (optional)
        self._data = None
        self._pending_uploads = []  # list of (offset, size) tuples

        # Get nbytes
        if data is not None:
            self._data = data
            self._nbytes = self._nbytes_from_data(data)
            self._nitems = self._nitems_from_data(data, nitems)
            self._pending_uploads.append((0, self._nitems))
            if nbytes is not None:
                if nbytes != self._nbytes:
                    raise ValueError("Given nbytes does not match size of given data.")
            if nitems is not None:
                if nitems != self._nitems:
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
    def dirty(self):
        """ Whether the buffer is dirty (needs to be processed by the renderer).
        """
        return bool(self._pending_uploads)

    @property
    def usage(self):
        """ The buffer usage flags (as a string with "|" as a separator).
        """
        return self._usage

    @property
    def nbytes(self):
        """ The number of bytes in the buffer.
        """
        return self._nbytes

    @property
    def nitems(self):
        """ The number of items in the buffer.
        """
        return self._nitems

    @property
    def format(self):
        """ The vertex or index format (depending on the value of usage).
        """
        if self._format is not None:
            return self._format
        elif self.data is not None:
            self._format = self._format_from_data(self.data)
            return self._format
        else:
            raise ValueError("Buffer has no data nor format.")

    @property
    def vertex_byte_range(self):
        """ The offset and size, in bytes, when used as a vertex buffer.
        """
        return self._vertex_byte_range

    @vertex_byte_range.setter
    def vertex_byte_range(self, offset_nbytes):
        offset, nbytes = int(offset_nbytes[0]), int(offset_nbytes[1])
        assert offset >= 0
        assert offset + nbytes <= self.nbytes
        self._vertex_byte_range = offset, nbytes

    @property
    def data(self):
        """ The data for this buffer as given when instantiated. Can
        be None if the data only exists on the GPU.
        """
        return self._data

    def update_range(self, offset=0, size=2 ** 50):
        """ Mark a certain range of the data for upload to the GPU. The
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
            raise ValueError("Update size out of range")
        # Merge with current entry?
        if self._pending_uploads:
            cur_offset, cur_size = self._pending_uploads.pop(-1)
            offset = min(offset, cur_offset)
            size = max(size, cur_size)
        # Limit and apply
        self._pending_uploads.append((offset, size))
        # todo: this can be smarter, we have logic for chunking in the morph tool

    # To implement in subclasses

    def _nbytes_from_data(self, data):
        raise NotImplementedError()

    def _nitems_from_data(self, data, nitems):
        raise NotImplementedError()

    def _format_from_data(self, data):
        raise NotImplementedError()

    def _renderer_copy_data_to_ctypes_object(self, ob, offset, size):
        """ Allows renderer to efficiently copy the data.
        """
        raise NotImplementedError()


class Buffer(BaseBuffer):  # numpy-based
    """ A class that wraps a (GPU) buffer object, optionally providing
    data for it. You can also use it as a placeholder for a buffer with
    no representation on the CPU.
    """

    def _nbytes_from_data(self, data):
        return data.nbytes

    def _nitems_from_data(self, data, nitems):
        if data.shape:
            return data.shape[0]
        else:
            return 1

    def _format_from_data(self, data):
        if "INDEX" in self.usage:

            key = str(data.dtype)
            mapping = {
                "int16": wgpu.IndexFormat.uint16,
                "uint16": wgpu.IndexFormat.uint16,
                "int32": wgpu.IndexFormat.uint32,
                "uint32": wgpu.IndexFormat.uint32,
            }
            try:
                return mapping[key]
            except KeyError:
                raise TypeError(
                    "Need dtype (u)int16 or (u)int32 for index data, not '{dtype}'."
                )

        else:  # if "VERTEX" in self.usage:

            shape = data.shape
            if len(shape) == 1:
                shape = shape + (1,)
            assert len(shape) == 2
            key = str(data.dtype), shape[-1]
            mapping = {
                ("float32", 1): wgpu.VertexFormat.float,
                ("float32", 2): wgpu.VertexFormat.float2,
                ("float32", 3): wgpu.VertexFormat.float3,
                ("float32", 4): wgpu.VertexFormat.float4,
                #
                ("float16", 2): wgpu.VertexFormat.half2,
                ("float16", 4): wgpu.VertexFormat.half4,
                #
                ("int8", 2): wgpu.VertexFormat.char2,
                ("int8", 4): wgpu.VertexFormat.char4,
                ("uint8", 2): wgpu.VertexFormat.uchar2,
                ("uint8", 4): wgpu.VertexFormat.uchar4,
                #
                ("int16", 2): wgpu.VertexFormat.short2,
                ("int16", 4): wgpu.VertexFormat.short4,
                ("uint16", 2): wgpu.VertexFormat.ushort2,
                ("uint16", 4): wgpu.VertexFormat.ushort4,
                #
                ("int32", 1): wgpu.VertexFormat.int,
                ("int32", 2): wgpu.VertexFormat.int2,
                ("int32", 3): wgpu.VertexFormat.int3,
                ("int32", 4): wgpu.VertexFormat.int4,
                #
                ("uint32", 1): wgpu.VertexFormat.uint,
                ("uint32", 2): wgpu.VertexFormat.uint2,
                ("uint32", 3): wgpu.VertexFormat.uint3,
                ("uint32", 4): wgpu.VertexFormat.uint4,
            }
            try:
                return mapping[key]
            except KeyError:
                if data.dtype == "float64":
                    raise ValueError(
                        "64-bit float is not supported, use 32-bit float instead"
                    )
                raise ValueError(f"Invalid dtype/shape for vertex data: {key}")

    def _renderer_copy_data_to_ctypes_object(self, ob, offset, size):
        nbytes_per_item = self._nbytes // self._nitems
        byte_offset = offset * nbytes_per_item
        byte_size = size * nbytes_per_item
        assert byte_size == ctypes.sizeof(ob)
        ctypes.memmove(
            ctypes.addressof(ob), self.data.ctypes.data + byte_offset, byte_size,
        )
