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


class Resource(Trackable):
    """Resource base class."""

    def __init__(self):
        super().__init__()
        registry._gfx_register(self)

    def _gfx_mark_for_sync(self):
        registry._gfx_mark_for_sync(self)


class ResourceRegistry:
    """Singleton registry for resources."""

    def __init__(self):
        self._all = weakref.WeakSet()
        self._syncable = weakref.WeakSet()

    def _gfx_register(self, resource):
        """Add a resource to the registry."""
        if not isinstance(resource, Resource):
            raise TypeError("Given object is not a Resource")
        self._all.add(resource)

    def _gfx_mark_for_sync(self, resource):
        """Register the given resource for synchonization. Only adds the resource
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

    def get_resource_count(self):
        """Get a dictionary indicating how many buffers and texture are currently alive."""
        counts = {"Buffer": 0, "Texture": 0}
        for r in self._all:
            name = r.__class__.__name__
            counts[name] = counts.get(name, 0) + 1
        return counts

    def get_syncable_resources(self, *, flush=False):
        """Get the set of resources that need syncing. If setting flush
        to True, the caller is responsible for syncing the resources.
        """
        syncable = set(self._syncable)
        if flush:
            self._syncable.clear()
        return syncable


registry = ResourceRegistry()
