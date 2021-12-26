import random
import weakref
import threading

import numpy as np

from ..linalg import Vector3, Matrix4, Quaternion
from ..linalg.utils import transform_aabb, aabb_to_sphere
from ..resources import Resource, Buffer
from ..utils import array_from_shadertype


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


class ResourceContainer:
    """Base class for WorldObject, Geometry and Material."""

    def __init__(self):
        self._resource_parents = weakref.WeakSet()
        self._rev = 0

    @property
    def rev(self):
        """Monotonically increasing integer that gets bumped when any
        of its buffers or textures are set. (Not when updates are made
        to these resources themselves).
        """
        return self._rev

    # NOTE: we could similarly let bumping of a resource's rev bump a
    # data_rev here. But it is not clear whether the (minor?) increase
    # in performance is worth the added complexity.

    def _bump_rev(self):
        """Bump the rev (and that of any "resource parents"), to trigger a pipeline rebuild."""
        self._rev += 1
        for x in self._resource_parents:
            x._rev += 1

    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        if isinstance(value, ResourceContainer):
            value._resource_parents.add(self)
            self._bump_rev()
        elif isinstance(value, Resource):
            self._bump_rev()


class WorldObject(ResourceContainer):
    """The base class for objects present in the "world", i.e. the scene graph.

    Each WorldObject has geometry to define it's data, and material to define
    its appearance. The object itself is only responsible for defining object
    hierarchies (parent / children) and its position and orientation in the world.

    This is considered a base class. Use Group to collect multiple world objects
    into a single empty world object.
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
    ):
        super().__init__()

        self.geometry = geometry
        self.material = material

        # Init visibility and render props
        self.visible = visible
        self.render_order = render_order
        self.render_mask = render_mask

        # Init parent and children
        self._parent = None
        self._children = []

        self.position = Vector3()
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

    def __del__(self):
        id_provider.release_id(self, self.id)

    @property
    def id(self):
        """An integer id smaller than 2**31 (read-only)."""
        return self._id

    @property
    def visible(self):
        """Wheter is object is rendered or not. Default True."""
        return self._visible

    @visible.setter
    def visible(self, visible):
        self._visible = bool(visible)

    @property
    def render_order(self):
        """This value allows the default rendering order of scene graph
        objects to be controlled. Default 0. See ``Renderer.sort_objects``
        for details.
        """
        return self._render_order

    @render_order.setter
    def render_order(self, value):
        self._render_order = float(value)

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
        return self._render_mask

    @render_mask.setter
    def render_mask(self, value):
        value = "auto" if value is None else value
        assert isinstance(value, str), "render_mask should be string"
        value = value.lower()
        options = ("opaque", "transparent", "auto", "all")
        if value not in options:
            raise ValueError(
                f"WorldObject.render_mask must be one of {options} not {value!r}"
            )
        self._render_mask = value
        # Trigger a pipeline redraw, because this info is used in that code path
        self._bump_rev()

    @property
    def geometry(self):
        """The object's geometry, the data that defines (the shape of) this object."""
        return self._geometry

    @geometry.setter
    def geometry(self, geometry):
        self._geometry = geometry

    @property
    def material(self):
        """Wheter is object is rendered or not. Default True."""
        return self._material

    @material.setter
    def material(self, material):
        self._material = material

    @property
    def parent(self):
        """Object's parent in the scene graph (read-only).
        An object can have at most one parent.
        """
        return self._parent

    @property
    def children(self):
        """The child objects of this wold object (read-only tuple).
        Use ``.add()`` and ``.remove()`` to change this list.
        """
        return tuple(self._children)

    def add(self, *objects):
        """Adds object as child of this object. Any number of
        objects may be added. Any current parent on an object passed
        in here will be removed, since an object can have at most one
        parent.
        """
        for obj in objects:
            # orphan if needed
            if obj._parent is not None:
                obj._parent.remove(obj)
            # attach to scene graph
            obj._parent = self
            self._children.append(obj)
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
                obj._parent = None
            except ValueError:
                pass
        return self

    def traverse(self, callback, skip_invisible=False):
        """Executes the callback on this object and all descendants.

        If ``skip_invisible`` is given and True, objects whose
        ``visible`` property is False - and their children - are
        skipped. Note that modifying the scene graph inside the callback
        is discouraged.
        """
        if skip_invisible and not self._visible:
            return
        callback(self)
        for child in self._children:
            child.traverse(callback, skip_invisible)

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
        self.update_matrix_world(update_parents=True, update_children=True)
        return self._get_world_bounding_box()

    def _get_world_bounding_box(self):
        boxes = []
        if self._geometry:
            aabb = self._geometry.bounding_box()
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
        aabb = self.get_world_bounding_box()
        if aabb is not None:
            return aabb_to_sphere(aabb)

    def _wgpu_get_pick_info(self, pick_value):
        # In most cases the material handles this.
        return self.material._wgpu_get_pick_info(pick_value)
