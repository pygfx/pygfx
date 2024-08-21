from math import floor, ceil
import numpy as np

from ._base import Resource
from ._utils import (
    get_element_format_from_numpy_array,
    calculate_buffer_chunk_size,
    get_merged_blocks_from_mask_1d,
    check_data_is_clean_for_performance,
    is_little_endian,
    make_little_endian,
    logger,
)


class Buffer(Resource):
    """The Buffer represents a contiguous piece of GPU memory.

    A buffer can be used as index buffer or storage buffer. They are also used
    for uniform buffers (internally in the pygfx materials). You can provide
    (and update data for it), or use it as a placeholder for a buffer with no
    representation on the CPU.

    Parameters
    ----------
    data : array | None
        The initial data of the buffer. It must support the buffer-protocol,
        (e.g. a bytes or numpy array). If None, ``nbytes`` and ``nitems`` must
        be provided. The data will be accessible at ``buffer.data``, no copies
        are made.
    nbytes : int | None
        The size of the buffer in bytes. If both ``data`` and ``nbytes`` are
        given, the ``data.nbytes`` is checked against ``nbytes``.
    nitems : int | None
        The number of elements in the buffer. If both ``data`` and ``nitems``
        are given, the data is interpreted as having that many items (reshaped
        internally).
    format : None | str | ElementFormat | wgpu.VertexFormat | wgpu.IndexFormat
        A format string describing the buffer layout. This can follow pygfx'
        ``ElementFormat`` e.g. "3xf4", or wgpu's ``VertexFormat``. Optional: if
        None, it is automatically determined from the data.
    chunk_size : None | int
        The chunk size to use for uploading data to the GPU, expressed in items
        counts. When None (default) an optimal chunk size is determined
        automatically.
    force_contiguous : bool
        When set to true, the buffer goes into a stricter mode, forcing set data
        to be c_contiguous. This ensures optimal upload performance for cases
        when the data changes often.
    usage : int | wgpu.BufferUsage
        The wgpu ``usage`` flag for this buffer. Optional: typically pygfx can
        derive how the buffer is used and apply the appropriate flag. In cases
        where it doesn't this param provides an override. This is a bitmask flag
        (values are OR'd).

    Performance tips:

    * If the given data is not c_contiguous, the chunks will need to be copied
      at upload-time, which reduces performance when the data is changed often.
    * Setting ``force_contiguous`` ensures that the set data is contiguous, it
      is recommended to use this when the bufer data is dynamic.
    """

    def __init__(
        self,
        data=None,
        *,
        nbytes=None,
        nitems=None,
        format=None,
        chunk_size=None,
        force_contiguous=False,
        usage=0,
    ):
        super().__init__()
        Resource._rev += 1
        self._rev = Resource._rev

        # Attributes for internal use, updated by other parts of pygfx.
        self._wgpu_object = None
        self._wgpu_usage = int(usage)

        # Init
        self._data = None
        self._view = None
        self._force_contiguous = bool(force_contiguous)

        # Process data
        if data is not None:
            # Store data and view, and do some basic checks.
            # The view is a numpy array, but we go via memoryview to ensure data follows the buffer protocol.
            self._data = data
            self._view = view = np.asarray(memoryview(data))
            if self._force_contiguous:
                check_data_is_clean_for_performance("buffer", view)
            the_nbytes = view.nbytes
            if nbytes is not None and int(nbytes) != the_nbytes:
                raise ValueError("Given nbytes does not match size of given data.")
            # Establish number of items
            if nitems is not None:
                the_nitems = int(nitems)
            elif view.shape:
                the_nitems = view.shape[0]
            else:
                the_nitems = 1  # A scalar, e.g. a uniform struct
            reshape_array(view, the_nitems)
            # Establish format
            detected_format = None
            element_format = get_element_format_from_numpy_array(view)
            if element_format:
                elements_per_item = int(np.prod(view.shape[1:], initial=1))
                detected_format = (f"{elements_per_item}x" + element_format).lstrip(
                    "1x"
                )
        elif nbytes is not None and nitems is not None:
            # No data on the CPU side
            the_nbytes = int(nbytes)
            the_nitems = int(nitems)
            # Check
            if the_nbytes <= 0 or the_nitems <= 0:
                raise ValueError("Buffer size cannot be zero")
            bytes_per_item = the_nbytes // the_nitems
            if bytes_per_item * the_nitems != the_nbytes:
                raise ValueError("The given nbytes is not a multiple of nitems.")
            detected_format = None
        else:
            raise ValueError(
                "Buffer must be instantiated with either data or nbytes and nitems."
            )

        # Check size
        if the_nitems == 0:
            raise ValueError("Buffer size cannot be zero.")

        # Store derived props
        self._store.nbytes = the_nbytes
        self._store.nitems = the_nitems
        if format is not None:
            self._store.format = str(format)
        elif detected_format:
            self._store.format = detected_format
        else:
            self._store.format = None

        # Can now init other properties
        self.draw_range = 0, the_nitems

        # Get optimal chunk size
        if chunk_size is None:
            chunk_size = calculate_buffer_chunk_size(
                the_nitems,
                bytes_per_element=the_nbytes // the_nitems,
                byte_align=16,
                target_chunk_count=32,
            )
        else:
            chunk_size = min(max(int(chunk_size), 1), the_nitems)

        # Init chunks map
        if data is None:
            self._chunks_dirt_flag = 0
            self._chunk_size = 0
            self._chunk_mask = None
        else:
            self._chunks_dirt_flag = 2
            self._chunk_size = chunk_size
            n_chunks = ceil(the_nitems / self._chunk_size)
            self._chunk_mask = np.ones((n_chunks,), bool)

    @property
    def data(self):
        """The data for this buffer.

        Can be None if the data only exists on the GPU. This object is the same
        that was given to instantiate this object or with ``set_data()``.
        """
        return self._data

    @property
    def view(self):
        """A numpy array view on the data of this buffer.

        Can be None if the data only exists on the GPU. This is a view on the
        same memory as ``.data``. The first dimension matches ``nitems``.
        """
        return self._view

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
        elif self._view is not None:
            nelements_per_item = int(np.prod(self._view.shape[1:], initial=1))
            return nelements_per_item * self._view.itemsize
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

    def set_data(self, data):
        """Reset the data to a new array.

        This avoids a data-copy compared to doing ``buffer.data[:] = new_data``.
        The new data must fit the texture's shape and format.
        """
        # Get view
        view = np.asarray(memoryview(data))
        # Do couple of checks
        if self._force_contiguous:
            check_data_is_clean_for_performance("buffer", view)
        if view.nbytes != self._view.nbytes:
            raise ValueError("buffer.set_data() nbytes does not match.")
        if view.dtype != self._view.dtype:
            raise ValueError("buffer.set_data() format does not match.")
        # Make sure the shape is ok. We only care about the first dimension.
        reshape_array(view, self.nitems)
        # Ok
        self._data = data
        self._view = view
        self.update_full()

    def update_full(self):
        """Mark the whole data for upload."""
        self._chunk_mask.fill(True)
        self._chunks_dirt_flag = 2
        Resource._rev += 1
        self._rev = Resource._rev
        self._gfx_mark_for_sync()

    def update_indices(self, indices):
        """Mark specific item indices for upload."""
        indices = np.asarray(indices)
        div = self._chunk_size
        self._chunk_mask[indices // div] = True
        self._chunks_dirt_flag = 1
        Resource._rev += 1
        self._rev = Resource._rev
        self._gfx_mark_for_sync()

    def update_range(self, offset=0, size=None):
        """Mark a certain range of the data for upload to the GPU.

        The offset and size are expressed in integer number of items.
        """
        # See ThreeJS BufferAttribute.updateRange

        nitems = self.nitems
        # Normalize inputs
        offset = int(offset or 0)
        size = int(nitems if size is None else size)
        # Checks
        if size == 0:
            return
        elif size < 0:
            raise ValueError("Update size must not be negative")
        elif offset < 0:
            raise ValueError("Update offset must not be negative")
        # Get indices
        index1 = offset
        index2 = min(nitems, offset + size)
        # Shortcut?
        if index1 == 0 and index2 == nitems:
            return self.update_full()
        # Update map
        div = self._chunk_size
        self._chunk_mask[floor(index1 / div) : ceil(index2 / div)] = True
        self._chunks_dirt_flag = 1
        Resource._rev += 1
        self._rev = Resource._rev
        self._gfx_mark_for_sync()

    def _gfx_get_chunk_descriptions(self):
        """Get a list of (offset, size) tuples, that can be
        used in _gfx_get_chunk_data(). This method also clears
        the chunk dirty statuses.
        """
        if not self._chunks_dirt_flag:
            return []
        elif self._chunks_dirt_flag == 2:
            chunk_descriptions = [(0, self.nitems)]
        elif np.all(self._chunk_mask):
            chunk_descriptions = [(0, self.nitems)]
        else:
            # Get merged chunk blocks, using a smart algorithm.
            chunk_blocks = get_merged_blocks_from_mask_1d(self._chunk_mask)

            # Turn into proper descriptions, with chunk indices/counts scaled with the chunk size.
            chunk_descriptions = []
            chunk_size = self._chunk_size
            nitems = self._store["nitems"]
            for x, nx in chunk_blocks:
                offset = x * chunk_size
                size = min(nx * chunk_size, nitems - offset)
                chunk_descriptions.append((offset, size))

        # Reset
        self._chunks_dirt_flag = 0
        self._chunk_mask.fill(False)
        return chunk_descriptions

    def _gfx_get_chunk_data(self, offset, size):
        """Return subdata as a contiguous array."""

        # Get chunk
        if offset == 0 and size == self.nitems:
            chunk = self._view
        else:
            chunk = self._view[offset : offset + size]

        # Normalize the chunk
        if not is_little_endian(chunk):
            chunk = make_little_endian(chunk)
            if self._force_contiguous:
                logger.warning(
                    "force_contiguous was set, but chunk data is still big endian"
                )
        elif not chunk.flags.c_contiguous:
            if self._force_contiguous:
                logger.warning(
                    "force_contiguous was set, but chunk data is still discontiguous"
                )
            chunk = np.ascontiguousarray(chunk)

        return chunk


def reshape_array(view, n):
    """Reshape array so it's shape[0] is n."""
    if not (view.shape and view.shape[0] == n):
        elements_per_item = -1
        if n == 0:
            elements_per_item = int(np.prod([max(i, 1) for i in view.shape], initial=1))
        # This can fail if the data is not contiguous and strides don't work out.
        view.shape = (n, elements_per_item)
