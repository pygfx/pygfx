import weakref

from ..linalg import Vector3, Matrix4, Quaternion
from ..datawrappers import Resource


class DataWrapperContainer:
    """ Base class for WorldObject, Geometry and Material.
    """

    def __init__(self):
        self._owr_parents = weakref.WeakSet()
        self._versionflag = 0

    @property
    def versionflag(self):
        """ Monotonically increasing integer that gets bumped when any
        of its buffers or textures are set. (Not when updates are made to these
        resources themselves).
        """
        return self._versionflag

    # todo: we can similarly let bumping of a resource's versionflag bump a resource_version_flag here
    def invalidate(self):
        """ Bump the versionflag (and that of any "parents")
        """
        self._versionflag += 1
        for x in self._owr_parents:
            x._versionflag += 1

    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        if isinstance(value, DataWrapperContainer):
            value._owr_parents.add(self)
            self.invalidate()
        elif isinstance(value, Resource):
            self.invalidate()


class WorldObject(DataWrapperContainer):
    """ The base class for objects present in the "world", i.e. the scene graph.

    Each WorldObject has geometry to define it's data, and material to define
    its apearance. The object itself is only responsible for defining object
    hierarchies (parent / children) and its position and orientation in the world.

    This is considered a base class. Use Group to collect multiple world objects
    into a single empty world object.
    """

    _v = Vector3()
    _m = Matrix4()
    _q = Quaternion()

    def __init__(self):
        super().__init__()
        self.parent = None
        self._children = []

        self.position = Vector3()
        self.rotation = Quaternion()
        self.scale = Vector3(1, 1, 1)
        self._transform_hash = ()

        self.up = Vector3(0, 1, 0)

        self.matrix = Matrix4()
        self.matrix_world = Matrix4()
        self.matrix_world_dirty = True
        self.matrix_world_version = 0

        self.visible = True
        self.render_order = 0

    @property
    def children(self):
        return tuple(self._children)

    def add(self, obj):
        # orphan if needed
        if obj.parent is not None:
            obj.parent.remove(obj)
        # attach to scene graph
        obj.parent = self
        self._children.append(obj)
        # flag world matrix as dirty
        obj.matrix_world_dirty = True
        return self

    def remove(self, obj):
        if obj in self._children:
            obj.parent = None
            self._children.remove(obj)

    def traverse(self, callback):
        callback(self)
        for child in self.children:
            child.traverse(callback)

    def update_matrix(self):
        p, r, s = self.position, self.rotation, self.scale
        hash = p.x, p.y, p.z, r.x, r.y, r.z, r.w, s.x, s.y, s.z
        if hash != self._transform_hash:
            self._transform_hash = hash
            self.matrix.compose(self.position, self.rotation, self.scale)
            self.matrix_world_dirty = True

    def update_matrix_world(
        self, force=False, update_children=True, update_parents=False
    ):
        if update_parents and self.parent:
            self.parent.update_matrix_world(
                force=force, update_children=False, update_parents=True
            )
        self.update_matrix()
        if self.matrix_world_dirty or force:
            if self.parent is None:
                self.matrix_world.copy(self.matrix)
            else:
                self.matrix_world.multiply_matrices(
                    self.parent.matrix_world, self.matrix
                )
            self.matrix_world_dirty = False
            self.matrix_world_version += 1
            for child in self._children:
                child.matrix_world_dirty = True
        if update_children:
            for child in self._children:
                child.update_matrix_world()

    def look_at(self, target: Vector3):
        self.update_matrix_world(update_parents=True, update_children=False)
        self._v.set_from_matrix_position(self.matrix_world)
        self._m.look_at(self._v, target, self.up)
        self.rotation.set_from_rotation_matrix(self._m)
        if self.parent:
            self._m.extract_rotation(self.parent.matrix_world)
            self._q.set_from_rotation_matrix(self._m)
            self.rotation.premultiply(self._q.inverse())
