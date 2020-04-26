import ctypes

import wgpu
import numpy as np


# todo: can this be generic enough, keeping the GPU bits out / optional?

# todo: Support for updating unmapped data. Use something like updateRange to do subBufferUpdate


class BaseBufferWrapper:
    """ A base buffer wrapper that can be implemented for numpy, ctypes arrays,
    or any other kind of array.
    """

    def __init__(self, data=None, *, nbytes=None, usage):
        # To specify the buffer size
        self._nbytes = 0
        self._nitems = 1
        # We can use a view / subset
        # todo: expose this via a BufferView class?
        self._view_range = (0, 2 ** 50)
        # The actual data (optional)
        self._data = None
        self._pending_uploads = []  # list of (offset, size) tuples
        # The native buffer object - created and set by the renderer
        self._gpu_buffer = None

        # Get nbytes
        if data is not None:
            self._data = data
            self._nbytes = self._nbytes_from_data(data)
            self._nitems = self._nitems_from_data(data)
            self._pending_uploads.append((0, self._nbytes))
            if nbytes is not None:
                if nbytes != self._nbytes:
                    raise ValueError("Given nbytes does not match size of given data.")
        elif nbytes is not None:
            self._nbytes = int(nbytes)
        else:
            raise ValueError("Buffer must be instantiated with either data or nbytes.")

        # Determine usage
        if isinstance(usage, str):
            usages = usage.upper().replace(",", " ").replace("|", " ").split()
            assert usages
            self._usage = "|".join(usages)
        else:
            raise TypeError("Buffer usage must be str.")

    @property
    def nbytes(self):
        """ Get the number of bytes in the buffer.
        """
        return self._nbytes

    @property
    def nitems(self):
        """ Get the number of items in the buffer.
        """
        return self._nitems

    @property
    def view_range(self):
        self._view_range

    @property
    def usage(self):
        """ The buffer usage flags (as an int).
        """
        return self._usage

    @property
    def dirty(self):
        """ Whether the buffer is dirty (needs to be processed by the renderer).
        """
        return bool(self._pending_uploads)

    @property
    def strides(self):
        """ Stride info (as a tuple).
        """
        return self._get_strides()

    @property
    def data(self):
        """ The data that is a view on the data. Can be None if the
        data only exists on the GPU.
        Note that this array can be replaced, so get it via this property.
        """
        # todo: maybe this class should not store _data if the data is not mapped?
        return self._data

    @property
    def gpu_buffer(self):
        """ The WGPU buffer object. Can be None if the renderer has not set it (yet).
        """
        return self._gpu_buffer

    # def resize(self):
    #     pass

    def set_view_range(self, start, stop):
        """ Set the view range
        """
        self._view_range = int(start), int(stop)
        # todo: implement this

    # def set_data(self, data):
    #     """ Reset the data.
    #     """
    #     self._data = data
    #     self._nbytes = self._nbytes_from_data(data)
    #     self._pending_data.append((data, 0))
    #     self._dirty = True

    def update_range(self, offset=0, size=2 ** 50):
        """ Mark a certain range of the data for upload to the GPU. The
        offset and size are expressed in integer number of elements.
        """
        # See ThreeJS BufferAttribute.updateRange
        # Check input
        if size == 0:
            return
        elif size < 0:
            raise ValueError("Update size must be positive")
        elif offset < 0:
            raise ValueError("Update size must not be negative")
        elif offset + size > self.nitems:
            raise ValueError("Update size eout of range")
        # Get in bytes
        nbytes_per_item = self._nbytes // self._nitems
        boffset, bsize = nbytes_per_item * int(offset), nbytes_per_item * int(size)
        if self._pending_uploads:
            current = self._pending_uploads.pop(-1)
            boffset, bsize = min(boffset, current[0]), max(bsize, current[1])
        self._pending_uploads.append((boffset, bsize))
        # todo: this can be smarter, we have logic for this in the morph tool

    def _renderer_set_gpu_buffer(self, buffer):
        # This is how the renderer marks the buffer as non-dirty
        self._gpu_buffer = buffer

    # To implement in subclasses

    def _get_strides(self):
        raise NotImplementedError()

    def _nbytes_from_data(self, data):
        raise NotImplementedError()

    def _renderer_copy_data_to_ctypes_object(self, ob, offset=0):
        """ Allows renderer to efficiently copy the data.
        """
        raise NotImplementedError()

    def _renderer_set_data_from_ctypes_object(self, ob):
        """ Allows renderer to replace the data.
        """
        raise NotImplementedError()

    def _renderer_get_data_dtype_str(self):
        """ Return numpy-ish dtype string, e.g. uint8, int16, float32.
        """
        raise NotImplementedError()

    def _renderer_get_vertex_format(self):
        raise NotImplementedError()


class BufferWrapper(BaseBufferWrapper):  # numpy-based
    """ Object that wraps a (GPU) buffer object, optionally providing data
    for it, and optionally *mapping* the data so it's shared. But you can also
    use it as a placeholder for a buffer with no representation on the CPU.
    """

    def _get_strides(self):
        return self.data.strides

    def _nbytes_from_data(self, data):
        return data.nbytes

    def _nitems_from_data(self, data):
        if data.shape:
            return data.shape[0]
        else:
            return 1

    def _renderer_copy_data_to_ctypes_object(self, ob, offset=0):
        nbytes = ctypes.sizeof(ob)
        ctypes.memmove(
            ctypes.addressof(ob), self.data.ctypes.data + offset, nbytes,
        )

    def _renderer_set_data_from_ctypes_object(self, ob):
        new_array = np.asarray(ob)
        new_array.dtype = self._data.dtype
        new_array.shape = self._data.shape
        self._data = new_array

    def _renderer_get_data_dtype_str(self):
        return str(self.data.dtype)

    def _renderer_get_vertex_format(self):
        shape = self.data.shape
        if len(shape) == 1:
            shape = shape + (1,)
        assert len(shape) == 2
        key = str(self.data.dtype), shape[-1]
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
            raise ValueError(f"Invalid dtype/shape for vertex data: {key}")
