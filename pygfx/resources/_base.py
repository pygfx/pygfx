import weakref

from ..utils.trackable import Trackable


class Resource(Trackable):
    """Base class for :class:`~pygfx.resources.Buffer` and :class:`~pygfx.resources.Texture`."""

    _resource_counts = {}  # Just to track the number of buffers and textures alive
    _rev = 0  # integer hash

    def __init__(self):
        super().__init__()
        cname = self.__class__.__name__
        Resource._resource_counts[cname] = Resource._resource_counts.get(cname, 0) + 1

    def __del__(self):
        cname = self.__class__.__name__
        Resource._resource_counts[cname] -= 1

    def _gfx_mark_for_sync(self):
        resource_update_registry._gfx_mark_for_sync(self)

    @property
    def rev(self):
        """The revision number (integer).

        The number changes when ``update_range()`` is called. The number is
        monotonically increasing and globally unique (no two buffers/textures
        have the same rev). This makes that it can be used as hash for the data
        content.
        """
        return self._rev


class ResourceUpdateRegistry:
    """Singleton registry to keep track of resources that need to be updated for the wgpu-backend."""

    def __init__(self):
        self._syncable = weakref.WeakSet()

    def _gfx_mark_for_sync(self, resource):
        """Register the given resource for synchronization. Only adds the resource
        if it's wgpu-counterpart already exists, and when it has pending uploads.
        """
        if not isinstance(resource, Resource):
            raise TypeError("Given object is not a Resource")
        if resource._wgpu_object is not None:
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
