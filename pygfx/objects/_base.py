import random
import weakref
import threading
import enum
from typing import List
import pylinalg as pla

import numpy as np

from ..resources import Buffer
from ..utils import array_from_shadertype
from ..utils.trackable import RootTrackable
from ._events import EventTarget
from ..utils.transform import AffineTransform, ChainedTransform, EmbeddedTransform


class IdProvider:
    """Object for internal use to manage world object id's."""

    def __init__(self):
        self._ids_in_use = set([0])
        self._map = weakref.WeakValueDictionary()
        self._lock = threading.RLock()

    def claim_id(self, wobject):
        """Used by wobjects to claim an id."""
        # We don't simply count up, but keep a pool of ids. This is
        # because an application *could* create and discard objects at
        # a high rate, so we want to be able to re-use these ids.
        #
        # Some numbers:
        # * 4_294_967_296 (2**32) max number for u32
        # * 2_147_483_647 (2**31 -1) max number for i32.
        # *    16_777_216 max integer that can be stored exactly in f32
        # *     4_000_000 max integer that survives being passed as a varying (in my tests)
        # *     1_048_575 is ~1M is 2**20 seems like a good max scene objects.
        # *    67_108_864 is ~50M is 2**26 seems like a good max vertex count.
        #                 which leaves 64-20-26=18 bits for any other picking info.

        # Max allowed id, inclusive
        id_max = 1_048_575  # 2*20-1

        # The max number of ids. This is a bit less to avoid choking
        # when there are few free id's left.
        max_items = 1_000_000

        with self._lock:
            if len(self._ids_in_use) >= max_items:
                raise RuntimeError("Max number of objects reached.")
            id = 0
            while id in self._ids_in_use:
                id = random.randint(1, id_max)
            self._ids_in_use.add(id)
            self._map[id] = wobject

        return id

    def release_id(self, wobject, id):
        """Release an id associated with a wobject."""
        with self._lock:
            self._ids_in_use.discard(id)
            self._map.pop(id, None)

    def get_object_from_id(self, id):
        """Return the wobject associated with an id, or None."""
        return self._map.get(id)


id_provider = IdProvider()


class RenderMask(enum.IntFlag):
    auto = 0
    opaque = 1
    transparent = 2
    all = 3


class WorldObject(EventTarget, RootTrackable):
    """Base class for objects.

    This class represents objects in the world, i.e., the scene graph.Each
    WorldObject has geometry to define it's data, and material to define its
    appearance. The object itself is only responsible for defining object
    hierarchies (parent / children) and its position and orientation in the
    world.

    Parameters
    ----------
    geometry : Geometry
        The data defining the shape of the object.
    material : Material
        The data defining the appearence of the object.
    visible : bool
        Whether the object is visible.
    render_order : int
        The render order (when applicable for the renderer's blend mode).
    render_mask : str
        Determines the render passes that the object is rendered in. It's
        recommended to let the renderer decide, using "auto".
    position : Vector
        The position of the object in the world. Default (0, 0, 0).

    Notes
    -----
    Use :class:`Group` to collect multiple world objects into a single empty
    world object.

    """

    _FORWARD_IS_MINUS_Z = False  # Default is +Z (lights and cameras use -Z)

    # The uniform type describes the structured info for this object, which represents
    # every "propery" that a renderer would need to know in order to visualize it.
    # Put larger items first for alignment, also note that host-sharable structs
    # align at power-of-two only, so e.g. vec3 needs padding.
    # todo: rename uniform to info or something?

    uniform_type = dict(
        world_transform="4x4xf4",
        world_transform_inv="4x4xf4",
        id="i4",
    )

    def __init__(
        self,
        geometry=None,
        material=None,
        *,
        visible=True,
        render_order=0,
        render_mask="auto",
    ):
        super().__init__()
        self._parent: weakref.ReferenceType[WorldObject] = None
        self.children: List[WorldObject] = []

        self.geometry = geometry
        self.material = material

        # Init visibility and render props
        self.visible = visible
        self.render_order = render_order
        self.render_mask = render_mask

        # Compose complete uniform type
        # Note: buffer is a local variable to avoid a circular reference to self
        # that would prevent garbage collection
        buffer = Buffer(array_from_shadertype(self.uniform_type))

        def buffer_callback(transform: AffineTransform):
            buffer.data["world_transform_inv"] = transform.inverse_matrix
            buffer.update_range()

        world_matrix = buffer.data["world_transform"]
        world_matrix[:] = np.eye(4)
        self.world_transform = AffineTransform(
            world_matrix, update_callback=buffer_callback
        )
        self.transform = EmbeddedTransform(self.world_transform)
        self.uniform_buffer = buffer

        # Set id
        self._id = id_provider.claim_id(self)
        self.uniform_buffer.data["id"] = self._id

        self.cast_shadow = False
        self.receive_shadow = False

    def __repr__(self):
        return f"<pygfx.{self.__class__.__name__} at {hex(id(self))}>"

    def __del__(self):
        id_provider.release_id(self, self.id)

    @property
    def id(self):
        """An integer id smaller than 2**31 (read-only)."""
        return self._id

    @property
    def tracker(self):
        return self._root_tracker

    @property
    def visible(self):
        """Wheter is object is rendered or not. Default True."""
        return self._store.visible

    @visible.setter
    def visible(self, visible):
        self._store.visible = bool(visible)

    @property
    def render_order(self):
        """This value allows the default rendering order of scene graph
        objects to be controlled. Default 0. See ``Renderer.sort_objects``
        for details.
        """
        return self._store.render_order

    @render_order.setter
    def render_order(self, value):
        self._store.render_order = float(value)

    @property
    def render_mask(self):
        """Indicates in what render passes to render this object:

        * "auto": try to determine the best approach (default).
        * "opaque": only in the opaque render pass.
        * "transparent": only in the transparent render pass(es).
        * "all": render in both opaque and transparent render passses.

        If "auto" (the default), the renderer attempts to determine
        whether all fragments will be either opaque or all transparent,
        and only apply the needed render passes. If this cannot be
        determined, it falls back to "all".

        Some objects may contain both transparent and opaque fragments,
        and should be rendered in all passes - the object's contribution
        to each pass is determined on a per-fragment basis.

        For clarity, rendering objects in all passes even though they
        are fully opaque/transparent yields correct results and is
        generally the "safest" option. The only cost is performance.
        Rendering transparent fragments in the opaque pass makes them
        invisible. Rendering opaque fragments in the transparent pass
        blends them as if they are transparent with an alpha of 1.
        """
        return self._store.render_mask

    @render_mask.setter
    def render_mask(self, value):
        if value is None:
            self._store.render_mask = RenderMask(0)
        elif isinstance(value, int):
            self._store.render_mask = RenderMask(value)
        elif isinstance(value, str):
            try:
                self._store.render_mask = RenderMask._member_map_[value.lower()]
            except KeyError:
                opts = set(RenderMask._member_names_)
                msg = f"WorldObject.render_mask must be one of {opts} not {value!r}"
                raise ValueError(msg) from None
        else:
            raise TypeError(
                f"WorldObject.render_mask must be int or str, not {type(value)}"
            )

    @property
    def geometry(self):
        """The object's geometry, the data that defines (the shape of) this object."""
        return self._store.geometry

    @geometry.setter
    def geometry(self, geometry):
        self._store.geometry = geometry

    @property
    def material(self):
        """Wheter is object is rendered or not. Default True."""
        return self._store.material

    @material.setter
    def material(self, material):
        self._store.material = material

    @property
    def cast_shadow(self):
        """Whether this object casts shadows, i.e. whether it is rendered into
        a shadow map. Default False."""
        return self._cast_shadow  # does not affect any shaders

    @cast_shadow.setter
    def cast_shadow(self, value):
        self._cast_shadow = bool(value)

    @property
    def receive_shadow(self):
        """Whether this object receives shadows. Default False."""
        return self._store.receive_shadow

    @receive_shadow.setter
    def receive_shadow(self, value):
        self._store.receive_shadow = bool(value)

    @property
    def parent(self) -> "WorldObject":
        if self._parent is None:
            return None
        else:
            return self._parent()

    def add(self, *objects, before=None):
        """Adds object as child of this object. Any number of
        objects may be added. Any current parent on an object passed
        in here will be removed, since an object can have at most one
        parent.
        If ``before`` argument is given, then the items are inserted before the
        given element.

        """

        obj: WorldObject
        for obj in objects:
            if obj.parent is not None:
                obj.parent.remove(obj)

            if before is not None:
                idx = self.children.index(before)
            else:
                idx = len(self.children)

            obj._parent = weakref.ref(self)
            obj.transform.before = self.transform_sequence
            self.children.insert(idx, obj)

    def remove(self, *objects):
        """Removes object as child of this object. Any number of objects may be removed."""

        obj: WorldObject
        for obj in objects:
            obj._parent = None
            obj.transform.before = None
            self.children.remove(obj)

    def clear(self):
        """Removes all children."""
        self.remove(*self.children)

    def traverse(self, callback, skip_invisible=False):
        """Executes the callback on this object and all descendants.

        If ``skip_invisible`` is given and True, objects whose
        ``visible`` property is False - and their children - are
        skipped. Note that modifying the scene graph inside the callback
        is discouraged.
        """

        for child in self.iter(skip_invisible=skip_invisible):
            callback(child)

    def iter(self, filter_fn=None, skip_invisible=False):
        """Create a generator that iterates over this objects and its children.
        If ``filter_fn`` is given, only objects for which it returns ``True``
        are included.
        """
        if skip_invisible and not self.visible:
            return

        if filter_fn is None:
            yield self
        elif filter_fn(self):
            yield self

        for child in self.children:
            yield from child.iter(filter_fn, skip_invisible)

    @property
    def transform_sequence(self) -> ChainedTransform:
        own_sequence = ChainedTransform([self.transform])

        if self.parent is None:
            return own_sequence
        else:
            return self.parent.transform_sequence @ own_sequence

    @property
    def bounding_box(self):
        partial_aabb = np.zeros((len(self.children) + 1, 2, 3), dtype=float)
        for idx, child in enumerate(self.children):
            aabb = child.bounding_box
            trafo = child.transform.matrix
            partial_aabb[idx] = pla.aabb_transform(aabb, trafo)
        partial_aabb[-1] = self.geometry.bounding_box()

        final_aabb = np.zeros((2, 3), dtype=float)
        final_aabb[0] = np.min(partial_aabb[:, 0, :], axis=0)
        final_aabb[1] = np.max(partial_aabb[:, 1, :], axis=0)
        return final_aabb

    @property
    def bounding_sphere(self):
        return pla.aabb_to_sphere(self.bounding_box)

    @property
    def world_bounding_box(self):
        """Updates all parent and children world matrices, and returns
        a single world-space axis-aligned bounding box for this object's
        geometry and all of its children (recursively)."""

        return pla.aabb_transform(self.bounding_box, self.world_transform.matrix)

    @property
    def world_bounding_sphere(self):
        """Returns a world-space bounding sphere by converting an
        axis-aligned bounding box to a sphere.

        See WorldObject.get_world_bounding_box.
        """
        return pla.aabb_to_sphere(self.world_bounding_box)

    def _wgpu_get_pick_info(self, pick_value):
        # In most cases the material handles this.
        return self.material._wgpu_get_pick_info(pick_value)
