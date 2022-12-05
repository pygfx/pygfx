import random
import weakref
import threading
import enum

import numpy as np

from ..linalg import Vector3, Matrix4, Quaternion
from ..linalg.utils import transform_aabb, aabb_to_sphere
from ..resources import Buffer
from ..utils import array_from_shadertype
from ..utils.trackable import RootTrackable
from ._events import EventTarget


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
    """The base class for objects present in the "world", i.e. the scene graph.

    Each WorldObject has geometry to define it's data, and material to define
    its appearance. The object itself is only responsible for defining object
    hierarchies (parent / children) and its position and orientation in the world.

    This is considered a base class. Use Group to collect multiple world objects
    into a single empty world object.

    Parameters:
        geometry (Geometry): the data defining the shape of the object.
        material (Material): the object defining the appearence of the object.
        visible (bool): whether the object is visible.
        render_order (int): the render order (when applicable for the renderer's blend mode).
        render_mask (str): determines the render passes that the object is rendered in.
           It's recommended to let the renderer decide, using "auto".
        position (Vector): The position of the light source. Default (0, 0, 0).
    """

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

    _v = Vector3()
    _m = Matrix4()
    _q = Quaternion()

    def __init__(
        self,
        geometry=None,
        material=None,
        *,
        visible=True,
        render_order=0,
        render_mask="auto",
        position=None,
    ):
        super().__init__()

        self.geometry = geometry
        self.material = material

        # Init visibility and render props
        self.visible = visible
        self.render_order = render_order
        self.render_mask = render_mask

        # Init parent and children
        self._parent_ref = None
        self._children = []

        position = (0, 0, 0) if position is None else position
        self.position = (
            Vector3(*position) if isinstance(position, (tuple, list)) else position
        )
        self.rotation = Quaternion()
        self.scale = Vector3(1, 1, 1)
        self._transform_hash = ()

        self.up = Vector3(0, 1, 0)

        self._matrix = Matrix4()
        self._matrix_auto_update = True
        self._matrix_world = Matrix4()
        self._matrix_world_dirty = True

        # Compose complete uniform type
        self.uniform_type = {}
        for cls in reversed(self.__class__.mro()):
            self.uniform_type.update(getattr(cls, "uniform_type", {}))
        self.uniform_buffer = Buffer(array_from_shadertype(self.uniform_type))

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
    def parent(self):
        """Object's parent in the scene graph (read-only).
        An object can have at most one parent.
        """
        return self._parent_ref and self._parent_ref()

    @property
    def children(self):
        """The child objects of this wold object (read-only tuple).
        Use ``.add()`` and ``.remove()`` to change this list.
        """
        return tuple(self._children)

    def add(self, *objects, before=None):
        """Adds object as child of this object. Any number of
        objects may be added. Any current parent on an object passed
        in here will be removed, since an object can have at most one
        parent.
        If ``before`` argument is given (and present in children), then
        the items are inserted before the given element.
        """
        idx = len(self._children)
        if before:
            try:
                idx = self._children.index(before)
            except ValueError:
                pass
        for obj in objects:
            assert isinstance(obj, WorldObject)
            # orphan if needed
            if obj._parent_ref is not None:
                obj._parent_ref().remove(obj)
            # attach to scene graph
            obj._parent_ref = weakref.ref(self)
            self._children.insert(idx, obj)
            idx += 1
            # flag world matrix as dirty
            obj._matrix_world_dirty = True
        return self

    def remove(self, *objects):
        """Removes object as child of this object. Any number of objects may be removed.
        If a given object is not a child, it is ignored.
        """
        for obj in objects:
            try:
                self._children.remove(obj)
                obj._parent_ref = None
            except ValueError:
                pass
        return self

    def clear(self):
        """Removes all children."""
        for child in self._children:
            child._parent_ref = None
        self._children.clear()

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

    def update_matrix(self):
        p, r, s = self.position, self.rotation, self.scale
        hash = p.x, p.y, p.z, r.x, r.y, r.z, r.w, s.x, s.y, s.z
        if hash != self._transform_hash:
            self._transform_hash = hash
            self._matrix.compose(self.position, self.rotation, self.scale)
            self._matrix_world_dirty = True

    @property
    def matrix(self):
        """The (settable) transformation matrix."""
        return self._matrix

    @matrix.setter
    def matrix(self, matrix):
        self._matrix.copy(matrix)
        self._matrix.decompose(self.position, self.rotation, self.scale)
        self._matrix_world_dirty = True

    @property
    def matrix_world(self):
        """The world matrix (local matrix composed with any parent matrices)."""
        return self._matrix_world

    @property
    def matrix_auto_update(self):
        """Whether or not the matrix auto-updates."""
        return self._matrix_auto_update

    @matrix_auto_update.setter
    def matrix_auto_update(self, value):
        self._matrix_auto_update = bool(value)

    @property
    def matrix_world_dirty(self):
        """Whether or not the matrix needs updating (readonly)."""
        return self._matrix_world_dirty

    def apply_matrix(self, matrix):
        if self._matrix_auto_update:
            self.update_matrix()
        self._matrix.premultiply(matrix)
        self._matrix.decompose(self.position, self.rotation, self.scale)
        self._matrix_world_dirty = True

    def update_matrix_world(
        self, force=False, update_children=True, update_parents=False
    ):
        if update_parents and self.parent:
            self.parent.update_matrix_world(
                force=force, update_children=False, update_parents=True
            )
        if self._matrix_auto_update:
            self.update_matrix()
        if self._matrix_world_dirty or force:
            if self.parent is None:
                self._matrix_world.copy(self._matrix)
            else:
                self._matrix_world.multiply_matrices(
                    self.parent._matrix_world, self._matrix
                )
            self.uniform_buffer.data[
                "world_transform"
            ].flat = self._matrix_world.elements
            tmp_inv_matrix = Matrix4().get_inverse(self._matrix_world)
            self.uniform_buffer.data[
                "world_transform_inv"
            ].flat = tmp_inv_matrix.elements
            self.uniform_buffer.update_range(0, 1)
            self._matrix_world_dirty = False
            for child in self._children:
                child._matrix_world_dirty = True
        if update_children:
            for child in self._children:
                child.update_matrix_world()

    def look_at(self, target: Vector3):
        self.update_matrix_world(update_parents=True, update_children=False)
        self._v.set_from_matrix_position(self._matrix_world)
        self._m.look_at(self._v, target, self.up)
        self.rotation.set_from_rotation_matrix(self._m)
        if self.parent:
            self._m.extract_rotation(self.parent._matrix_world)
            self._q.set_from_rotation_matrix(self._m)
            self.rotation.premultiply(self._q.inverse())

    def get_world_position(self):
        self.update_matrix_world(update_parents=True, update_children=False)
        self._v.set_from_matrix_position(self._matrix_world)
        return self._v.clone()

    def get_world_bounding_box(self):
        """Updates all parent and children world matrices, and returns
        a single world-space axis-aligned bounding box for this object's
        geometry and all of its children (recursively)."""
        self.update_matrix_world(update_parents=True, update_children=True)
        return self._get_world_bounding_box()

    def _get_world_bounding_box(self):
        """Returns a world-space axis-aligned bounding box for this object's
        geometry and all of its children (recursively)."""
        boxes = []
        if self._store.geometry:
            aabb = self._store.geometry.bounding_box()
            aabb_world = transform_aabb(aabb, self._matrix_world.to_ndarray())
            boxes.append(aabb_world)
        if self._children:
            boxes.extend(
                [
                    b
                    for b in (c.get_world_bounding_box() for c in self._children)
                    if b is not None
                ]
            )
        if len(boxes) == 1:
            return boxes[0]
        if boxes:
            boxes = np.array(boxes)
            return np.array([boxes[:, 0].min(axis=0), boxes[:, 1].max(axis=0)])

    def get_world_bounding_sphere(self):
        """Returns a world-space bounding sphere by converting an
        axis-aligned bounding box to a sphere.

        See WorldObject.get_world_bounding_box.
        """
        aabb = self.get_world_bounding_box()
        if aabb is not None:
            return aabb_to_sphere(aabb)

    def _wgpu_get_pick_info(self, pick_value):
        # In most cases the material handles this.
        return self.material._wgpu_get_pick_info(pick_value)
