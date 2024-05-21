import numpy as np

from ._base import Resource, get_item_format_from_memoryview


class Buffer(Resource):
    """A contiguous piece of GPU memory.

    Buffers can be used as index buffer or storage buffer. They are also used
    for uniform buffers (internally in the pygfx materials). You can provide
    (and update data for it), or use it as a placeholder for a buffer with no
    representation on the CPU.

    Parameters
    ----------
    data : array
        The initial data of the array data. It must support the buffer-protocol,
        (e.g. a bytes or numpy array). If None, nbytes and nitems must be
        provided.
    nbytes : int
        The size of the buffer in bytes. Ignored if ``data`` is used.
    nitems : int
        The number of elements in the buffer. Ignored if ``data`` is used.
    format : None | str | ElementFormat | wgpu.VertexFormat | wgpu.IndexFormat
        A format string describing the buffer layout. This can follow pygfx'
        ``ElementFormat`` e.g. "3xf4", or wgpu's ``VertexFormat``. Optional: if
        None, it is automatically determined from the data.
    usage : int | wgpu.BufferUsage
        The wgpu ``usage`` flag for this buffer. Optional: typically pygfx can
        derive how the buffer is used and apply the appropriate flag. In cases
        where it doesn't this param provides an override. This is a bitmask flag
        (values are OR'd).
    """

    def __init__(self, data=None, *, nbytes=None, nitems=None, format=None, usage=0):
        super().__init__()
        Resource._rev += 1
        self._rev = Resource._rev
        # To specify the buffer size
        # The actual data (optional)
        self._data = None
        detected_format = None
        self._gfx_pending_uploads = []  # list of (offset, size) tuples

        # Attributes for internal use, updated by other parts of pygfx.
        self._wgpu_object = None
        self._wgpu_usage = int(usage)

        # Get nbytes
        if data is not None:
            self._data = data
            self._mem = mem = memoryview(data)
            subformat = get_item_format_from_memoryview(mem)
            if subformat:
                shape = (mem.shape + (1,)) if len(mem.shape) == 1 else mem.shape
                if len(shape) == 2:  # if not, the user does something fancy
                    detected_format = (f"{shape[-1]}x" + subformat).lstrip("1x")
            the_nbytes = mem.nbytes
            the_nitems = mem.shape[0] if mem.shape else 1
            if the_nitems:
                self._gfx_pending_uploads.append((0, the_nitems))
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

        if format is not None:
            self._store.format = str(format)
        elif detected_format:
            self._store.format = detected_format
        else:
            self._store.format = None
        self._store.nbytes = the_nbytes
        self._store.nitems = the_nitems

        self.draw_range = 0, the_nitems

    @property
    def data(self):
        """The data for this buffer. Can be None if the data only
        exists on the GPU.

        Note: the data is the same reference that was given to instantiate this
        object, but this may change.
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
        # Note: many properties are stored on ._store, even if they cannot
        # change. This is done so that whan a buffer is swapped from another, we
        # can track what properties effectively changed. E.g. to determine
        # whether the render_mask changes or a shader recompilation is
        # necessary.
        return self._store.nbytes

    @property
    def nitems(self):
        """The number of items in the buffer."""
        return self._store.nitems

    @property
    def itemsize(self):
        """The number of bytes for a single item."""
        # Note: For regular NxM buffers this can also be calculated from the
        # format, but not when the format is more complex / None, as with
        # uniform buffers (structured arrays).
        nbytes = self._store.nbytes  # deliberately touch
        nitems = self._store.nitems  # deliberately touch
        if nitems > 0:
            return nbytes // nitems
        elif self._data is not None:
            shape = self._mem.shape
            if shape:
                shape = shape[1:]
            nelements_per_item = int(np.prod(shape)) or 1
            return nelements_per_item * self._mem.itemsize
        else:
            raise RuntimeError("Cannot determine Buffer.itemsize")

    @property
    def format(self):
        """The buffer format.

        Usually a pygfx format specifier (e.g. 'u2' for scalar uint16, or '3xf4'
        for 3xfloat32), but can also be a value from ``wgpu.VertexFormat``, or
        None e.g. for uniform buffers.
        """
        return self._store.format

    @property
    def usage(self):
        """Bitmask indicating how the buffer can be used in a wgpu pipeline."""
        return self._wgpu_usage

    @property
    def vertex_byte_range(self):
        raise DeprecationWarning(
            "vertex_byte_range is deprecated, use draw_range instead."
        )

    @vertex_byte_range.setter
    def vertex_byte_range(self, offset_nbytes):
        raise DeprecationWarning(
            "vertex_byte_range is deprecated, use draw_range instead."
        )

    @property
    def draw_range(self):
        """The range to data (origin, size) expressed in items."""
        return self._store.draw_range

    @draw_range.setter
    def draw_range(self, draw_range):
        origin, size = draw_range
        origin, size = int(origin), int(size)
        if not (origin == 0 or 0 < origin < self.nitems):  # note nitems can be 0
            raise ValueError("draw_range origin out of bounds.")
        if not (size >= 0 and origin + size <= self.nitems):
            raise ValueError("draw_range size out of bounds.")
        self._store.draw_range = origin, size
        Resource._rev += 1
        self._rev = Resource._rev

    def update_range(self, offset=0, size=2**50):
        """Mark a certain range of the data for upload to the GPU. The
        offset and size are expressed in integer number of elements.
        """
        # See ThreeJS BufferAttribute.updateRange
        # Check input
        if not isinstance(offset, int) and isinstance(size, int):
            raise TypeError(
                f"`offset` and `size` must be native `int` type, you have passed: "
                f"offset: <{type(offset)}>, size: <{type(size)}>"
            )
        offset, size = int(offset), int(size)  # convert np ints to real ints

        if size == 0:
            return
        elif size < 0:
            raise ValueError("Update size must not be negative")
        elif offset < 0:
            raise ValueError("Update offset must not be negative")
        elif offset + size > self.nitems:
            size = self.nitems - offset
        # Merge with current entry?
        if self._gfx_pending_uploads:
            cur_offset, cur_size = self._gfx_pending_uploads.pop(-1)
            end = max(offset + size, cur_offset + cur_size)
            offset = min(offset, cur_offset)
            size = end - offset
        # Limit and apply
        self._gfx_pending_uploads.append((offset, size))
        Resource._rev += 1
        self._rev = Resource._rev
        self._gfx_mark_for_sync()
        # note: this can be smarter, we have logic for chunking in the morph tool

    def _get_subdata(self, offset, size):
        """Return subdata as a contiguous array."""
        # If this is a full range, this is easy (and fast)
        if offset == 0 and size == self.nitems and self.mem.c_contiguous:
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
