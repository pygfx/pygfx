import ctypes

import wgpu
import numpy as np


def array_from_shader_type(spirv_type):
    """ Get a numpy array object from a SpirV type from python-shader.
    """
    return np.asarray(spirv_type())


# todo: can this be generic enough, keeping the GPU bits out / optional?


class BaseBufferWrapper:
    """ A base buffer wrapper that can be implemented for numpy, ctypes arrays,
    or any other kind of array.
    """

    def __init__(self, data=None, nbytes=None, usage=None, mapped=False):
        self._data = data
        if nbytes is not None:
            self._nbytes = int(nbytes)
        elif data is not None:
            self._nbytes = self._nbytes_from_data(data)
        else:
            raise ValueError("Buffer must be instantiated with either data or nbytes.")
        if isinstance(usage, int):
            self._usage = usage
        elif isinstance(usage, str):
            usages = usage.upper().replace(",", " ").replace("|", " ").split()
            assert usages
            self._usage = 0
            for usage in usages:
                self._usage |= getattr(wgpu.BufferUsage, usage)
        else:
            self._usage = wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.VERTEX
            # raise ValueError("BufferWrapper usage must be int or str.")

        self._mapped = bool(mapped)
        self._dirty = True
        self._gpu_buffer = None  # Set by renderer

    @property
    def data(self):
        """ The data that is a view on the data. Can be None if the
        data only exists on the GPU.
        Note that this array can be replaced, so get it via this property.
        """
        # todo: maybe this class should not store _data if the data is not mapped?
        return self._data

    @property
    def nbytes(self):
        """ Get the number of bytes in the buffer.
        """
        return self._nbytes

    @property
    def mapped(self):
        """ Whether the data is mapped. Mapped data can be updated in-place
        to update the data on the GPU.
        """
        return self._mapped

    @property
    def usage(self):
        """ The buffer usage flags (as an int).
        """
        return self._usage

    def set_mapped(self, mapped):
        self._mapped = bool(mapped)
        self._dirty = True

    def set_nbytes(self, n):
        self._nbytes = n
        self._dirty = True

    def set_data(self, data):
        """ Allow user to reset the array data.
        """
        self._data = data
        self._nbytes = self._nbytes_from_data(data)
        self._dirty = True

    def _nbytes_from_data(self, data):
        raise NotImplementedError()

    def _renderer_copy_data_to_ctypes_object(self, ob):
        """ Allows renderer to efficiently copy the data.
        """
        raise NotImplementedError()

    def _renderer_set_data_from_ctypes_object(self, ob):
        """ Allows renderer to replace the data.
        """
        raise NotImplementedError()


class BufferWrapper(BaseBufferWrapper):  # numpy-based
    """ Object that wraps a (GPU) buffer object, optionally providing data
    for it, and optionally *mapping* the data so it's shared. But you can also
    use it as a placeholder for a buffer with no representation on the CPU.
    """

    def _nbytes_from_data(self, data):
        return data.nbytes

    def _renderer_copy_data_to_ctypes_object(self, ob):
        ctypes.memmove(
            ctypes.addressof(ob), self.data.ctypes.data, self.nbytes,
        )

    def _renderer_set_data_from_ctypes_object(self, ob):
        new_array = np.asarray(ob)
        new_array.dtype = self._data.dtype
        new_array.shape = self._data.shape
        self._data = new_array
