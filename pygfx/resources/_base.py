import weakref

from ..utils.trackable import Trackable


STRUCT_FORMAT_ALIASES = {"c": "B", "l": "i", "L": "I"}

FORMAT_MAP = {
    "b": "i1",
    "B": "u1",
    "h": "i2",
    "H": "u2",
    "i": "i4",
    "I": "u4",
    "e": "f2",
    "f": "f4",
}


def get_item_format_from_memoryview(mem):
    """Get the per-item format specifier from a memoryview.
    Returns None if the format appears to be a structured array.
    Raises an error if GPU-incompatible dtypes are used (64 bit).
    """

    # Uniform buffers are scalars with a structured dtype.
    # But can also create storage buffers with complex formats.
    if len(mem.format) > 2:
        return None

    # GPUs generally don't support 64-bit buffers or textures.
    # Note: the Python docs say that l and L are 32 bit, but converting
    # a int64 numpy array to a memoryview gives a format of 'l' instead
    # of 'q' on some systems/configs? So we need to check the itemsize.
    if mem.itemsize == 8:
        kind = "64-bit"
        if mem.format in "fd":
            kind = "float64"
        elif mem.format in "ilq":
            kind = "int64"
        elif mem.format in "ILQ":
            kind = "uint64"
        raise ValueError(
            f"A dtype of {kind} is not supported for buffers, use a 32-bit variant instead."
        )

    # Get normalized format
    format = str(mem.format)
    format = STRUCT_FORMAT_ALIASES.get(format, format)

    if format not in FORMAT_MAP:
        raise TypeError(
            f"Cannot convert {format!r} to wgpu format. You should provide data with a different dtype."
        )
    return FORMAT_MAP[format]


class Resource(Trackable):
    """Resource base class."""

    _resource_counts = {}
    _rev = 0

    def __init__(self):
        super().__init__()
        cname = self.__class__.__name__
        Resource._resource_counts[cname] = Resource._resource_counts.get(cname, 0) + 1

    def __del__(self):
        cname = self.__class__.__name__
        Resource._resource_counts[cname] -= 1

    def _gfx_mark_for_sync(self):
        resource_update_registry._gfx_mark_for_sync(self)


class ResourceUpdateRegistry:
    """Singleton registry to keep track of resources that need to be updated."""

    def __init__(self):
        self._syncable = weakref.WeakSet()

    def _gfx_mark_for_sync(self, resource):
        """Register the given resource for synchronization. Only adds the resource
        if it's wgpu-counterpart already exists, and when it has pending uploads.
        """
        # Note: this method is very specific to the wgpu renderer, and
        # its logic to handle updates to buffers and textures. We could
        # create some sort of plugin system so that each renderer can
        # register a registry (and that registry could attach the
        # _wgpu_xx attributes to the buffers/texture) but that feels
        # like overdesigning at this point. Let's track in #272.
        if not isinstance(resource, Resource):
            raise TypeError("Given object is not a Resource")
        if resource._wgpu_object is not None and resource._gfx_pending_uploads:
            self._syncable.add(resource)

    def get_syncable_resources(self, *, flush=False):
        """Get the set of resources that need syncing. If setting flush
        to True, the caller is responsible for syncing the resources.
        """
        syncable = set(self._syncable)
        if flush:
            self._syncable.clear()
        return syncable


resource_update_registry = ResourceUpdateRegistry()
