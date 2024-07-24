import random
import weakref
import threading
from typing import List, Tuple
import pylinalg as la

import numpy as np

from ..resources import Buffer
from ..utils import array_from_shadertype, logger
from ..utils.trackable import RootTrackable
from ._events import EventTarget
from ..utils.transform import (
    AffineBase,
    AffineTransform,
    RecursiveTransform,
    callback,
)
from ..utils.enums import RenderMask


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
        if id > 0:
            with self._lock:
                self._ids_in_use.discard(id)
                self._map.pop(id, None)

    def get_object_from_id(self, id):
        """Return the wobject associated with an id, or None."""
        return self._map.get(id)


id_provider = IdProvider()


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
        The data defining the appearance of the object.
    visible : bool
        Whether the object is visible.
    render_order : float
        Value that helps controls the order in which objects are rendered.
    render_mask : str | RenderMask
        Determines the render passes that the object is rendered in. It's
        recommended to let the renderer decide, using "auto".
    name : str
        The name of the object.

    Notes
    -----
    Use :class:`Group` to collect multiple world objects into a single empty
    world object.

    See Also
    --------
    pygfx.utils.transform.AffineBase
        Various getters and setters defined on ``obj.local`` and ``obj.world``.
    pygfx.utils.transform.AffineTransform
        The class used to implement ``obj.local``.
    pygfx.utils.transform.RecursiveTransform
        The class used to implement ``obj.world``.

    """

    _FORWARD_IS_MINUS_Z = False  # Default is +Z (lights and cameras use -Z)

    _id = 0

    # The uniform type describes the structured info for this object, which represents
    # every "property" that a renderer would need to know in order to visualize it.
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
        name="",
    ):
        super().__init__()
        self._parent: weakref.ReferenceType[WorldObject] = None

        #: Subtrees of the scene graph that depend on this object.
        self._children: List[WorldObject] = []

        self.geometry = geometry
        self.material = material

        self.name = name

        # Compose complete uniform type
        buffer = Buffer(array_from_shadertype(self.uniform_type), force_contiguous=True)
        buffer.data["world_transform"] = np.eye(4)
        buffer.data["world_transform_inv"] = np.eye(4)

        #: The object's transform expressed in parent space.
        self.local = AffineTransform(is_camera_space=self._FORWARD_IS_MINUS_Z)
        #: The object's transform expressed in world space.
        self.world = RecursiveTransform(
            self.local, is_camera_space=self._FORWARD_IS_MINUS_Z, reference_up=(0, 1, 0)
        )
        self.world.on_update(self._update_uniform_buffers)

        # Set id
        self._id = id_provider.claim_id(self)
        buffer.data["id"] = self._id

        #: The GPU data of this WorldObject.
        self.uniform_buffer = buffer

        # Init visibility and render props
        self.visible = visible
        self.render_order = render_order
        self.render_mask = render_mask
        self.cast_shadow = False
        self.receive_shadow = False

        self.name = name

    @callback
    def _update_uniform_buffers(self, transform: AffineBase):
        self.uniform_buffer.data["world_transform"] = transform.matrix.T
        self.uniform_buffer.data["world_transform_inv"] = transform.inverse_matrix.T
        self.uniform_buffer.update_full()

    def __repr__(self):
        return f"<pygfx.{self.__class__.__name__} {self.name} at {hex(id(self))}>"

    def __del__(self):
        id_provider.release_id(self, self.id)

    @property
    def up(self):
        """
        Relic of old WorldObjects that aliases with the new ``transform.up``
        direction. Prefer `obj.world.reference_up` instead.

        """

        logger.warning(
            "`WorldObject.up` is deprecated. Use `WorldObject.world.reference_up` instead.",
        )

        return self.world.reference_up

    @up.setter
    def up(self, value):
        logger.warning(
            "`WorldObject.up` is deprecated. Use `WorldObject.world.reference_up` instead.",
        )

        self.world.reference_up = np.asarray(value)

    @property
    def id(self):
        """An integer id smaller than 2**31 (read-only)."""
        return self._id

    @property
    def tracker(self):
        return self._root_tracker

    @property
    def visible(self):
        """Whether is object is rendered or not. Default True."""
        return self._store.visible

    @visible.setter
    def visible(self, visible):
        self._store.visible = bool(visible)

    @property
    def render_order(self):
        """A number that helps control the order in which objects are rendered.
        Objects with higher ``render_order`` get rendered later.
        Default 0. Also see ``Renderer.sort_objects``.
        """
        return self._store.render_order

    @render_order.setter
    def render_order(self, value):
        self._store.render_order = float(value)

    @property
    def render_mask(self):
        """Indicates in what render passes to render this object:

        See :obj:`pygfx.utils.enums.RenderMask`:

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
            value = 0
        if isinstance(value, int):
            pass
        elif isinstance(value, str):
            if value not in dir(RenderMask):
                msg = f"WorldObject.render_mask must be one of {dir(RenderMask)} not {value!r}"
                raise ValueError(msg) from None
            value = RenderMask[value]
        else:
            raise TypeError(
                f"WorldObject.render_mask must be int or str, not {type(value)}"
            )
        # Store the value as an int, because this is a flag, but also for backwards compat.
        self._store.render_mask = value

    @property
    def geometry(self):
        """The object's geometry, the data that defines (the shape of) this object."""
        return self._store.geometry

    @geometry.setter
    def geometry(self, geometry):
        self._store.geometry = geometry

    @property
    def material(self):
        """Whether is object is rendered or not. Default True."""
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
        """Object's parent in the scene graph (read-only).
        An object can have at most one parent.
        """
        if self._parent is None:
            return None
        else:
            return self._parent()

    @property
    def children(self) -> Tuple["WorldObject"]:
        """tuple of children of this object. (read-only)"""
        return tuple(self._children)

    def add(self, *objects, before=None, keep_world_matrix=False) -> "WorldObject":
        """Add child objects.

        Any number of objects may be added. Any current parent on an object
        passed in here will be removed, since an object can have at most one
        parent. If ``before`` argument is given, then the items are inserted
        before the given element.

        Parameters
        ----------
        *objects : WorldObject
            The world objects to add as children.
        before : WorldObject
            If not None, insert the objects before this child object.
        keep_world_matrix : bool
            If True, the child will keep it's world transform. It moves in the
            scene graph but will visually remain in the same place. If False,
            the child will keep it's parent transform.

        """

        obj: WorldObject
        for obj in objects:
            if obj.parent is not None:
                obj.parent.remove(obj, keep_world_matrix=keep_world_matrix)

            if before is not None:
                idx = self._children.index(before)
            else:
                idx = len(self._children)

            if keep_world_matrix:
                transform_matrix = obj.world.matrix

            obj._parent = weakref.ref(self)
            obj.world.parent = self.world
            self._children.insert(idx, obj)

            if keep_world_matrix:
                obj.world.matrix = transform_matrix

        return self

    def remove(self, *objects, keep_world_matrix=False):
        """Removes object as child of this object. Any number of objects may be removed."""

        obj: WorldObject
        for obj in objects:
            try:
                self._children.remove(obj)
            except ValueError:
                logger.warning("Attempting to remove object that was not a child.")
                continue
            else:
                obj._reset_parent(keep_world_matrix=keep_world_matrix)

    def clear(self, *, keep_world_matrix=False):
        """Removes all children."""

        for child in self._children:
            child._reset_parent(keep_world_matrix=keep_world_matrix)

        self._children.clear()

    def _reset_parent(self, *, keep_world_matrix=False):
        """Sets the parent to None.

        xref: https://github.com/pygfx/pygfx/pull/482#discussion_r1135670771
        """

        if keep_world_matrix:
            transform_matrix = self.world.matrix

        self._parent = None
        self.world.parent = None

        if keep_world_matrix:
            self.world.matrix = transform_matrix

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

        for child in self._children:
            yield from child.iter(filter_fn, skip_invisible)

    def get_bounding_box(self):
        """Axis-aligned bounding box in parent space.

        Returns
        -------
        aabb : ndarray, [2, 3] or None
            An axis-aligned bounding box, or None when the object does
            not take up a particular space.
        """

        # Collect bounding boxes
        aabbs = []
        for child in self._children:
            aabb = child.get_bounding_box()
            if aabb is not None:
                trafo = child.local.matrix
                aabbs.append(la.aabb_transform(aabb, trafo))
        if self.geometry is not None:
            aabb = self.geometry.get_bounding_box()
            if aabb is not None:
                aabbs.append(aabb)

        # Combine
        if aabbs:
            aabbs = np.stack(aabbs)
            final_aabb = np.zeros((2, 3), dtype=float)
            final_aabb[0] = np.min(aabbs[:, 0, :], axis=0)
            final_aabb[1] = np.max(aabbs[:, 1, :], axis=0)
        else:
            final_aabb = None

        return final_aabb

    def get_bounding_sphere(self):
        """Bounding Sphere in parent space.

        Returns
        -------
        bounding_shere : ndarray, [4] or None
            A sphere (x, y, z, radius), or None when the object does
            not take up a particular space.

        """
        aabb = self.get_bounding_box()
        return None if aabb is None else la.aabb_to_sphere(aabb)

    def get_world_bounding_box(self):
        """Axis aligned bounding box in world space.

        Returns
        -------
        aabb : ndarray, [2, 3] or None
            The transformed axis-aligned bounding box, or None when the
            object does not take up a particular space.

        """
        aabb = self.get_bounding_box()
        return None if aabb is None else la.aabb_transform(aabb, self.world.matrix)

    def get_world_bounding_sphere(self):
        """Bounding Sphere in world space.

        Returns
        -------
        bounding_shere : ndarray, [4] or None
            A sphere (x, y, z, radius), or None when the object does
            not take up a particular space.

        """
        aabb = self.get_world_bounding_box()
        return None if aabb is None else la.aabb_to_sphere(aabb)

    def _wgpu_get_pick_info(self, pick_value):
        # In most cases the material handles this.
        return self.material._wgpu_get_pick_info(pick_value)

    def look_at(self, target) -> None:
        """Orient the object so it looks at the given position.

        This sets the object's rotation such that its ``forward`` direction
        points towards ``target`` (given in world space). This rotation takes
        reference_up into account, i.e., the rotation is chosen in such a way that a
        camera looking ``forward`` follows the rotation of a human head looking
        around without tilting the head sideways.

        """

        self.world.forward = target - self.world.position
