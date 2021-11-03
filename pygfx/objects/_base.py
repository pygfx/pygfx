import gc
import random
import weakref

from ..linalg import Vector3, Matrix4, Quaternion
from ..resources import Resource, Buffer
from ..utils import array_from_shadertype


# Keep track of id's. About the max:
# * 2_147_483_647 (2**31 -1) max number for signed i32.
# *    16_777_216 max integer that can be stored exactly in f32
# *     4_000_000 max integer that survives being passed as a varying (in my tests)
_idmap = weakref.WeakKeyDictionary()
_idmax = 16_777_217  # non-inclusive


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

    def __init__(self, geometry=None, material=None):
        super().__init__()

        self.geometry = geometry
        self.material = material

        # Init visibility
        self._visible = True

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

        # Render order is undocumented feature for now;l it may be removed if we have OIT.
        self.render_order = 0

        # See if we reached max number of objects. If so, try cleanup and try again.
        # Actually, stop earlier to prevent that while loop below to become slow.
        max_items = 0.75 * _idmax
        if len(_idmap) >= max_items:
            gc.collect()
            if len(_idmap) >= max_items:
                raise RuntimeError("Max number of objects reached")
        # Set id
        self._id = random.randint(1, _idmax - 1)
        while self._id in _idmap.values():
            self._id = (self._id + 1) % _idmax
        _idmap[self] = self._id
        self.uniform_buffer.data["id"] = self._id

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
        if skip_invisible and not self.visible:
            return
        callback(self)
        for child in self.children:
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

    def _wgpu_get_pick_info(self, pick_value):
        # In most cases the material handles this.
        return self.material._wgpu_get_pick_info(pick_value)
