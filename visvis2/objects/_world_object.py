from ..linalg import Vector3, Matrix4, Quaternion


class WorldObject:
    """ The base class for objects present in the "world", i.e. the scene graph.
    """

    def __init__(self):
        self.parent = None
        self._children = []

        self.position = Vector3()
        self.rotation = Quaternion()
        self.scale = Vector3(1, 1, 1)

        self.matrix = Matrix4()
        self.matrix_world = Matrix4()
        self.matrix_world_dirty = True

    @property
    def children(self):
        return tuple(self._children)

    def add(self, obj):
        # prevent loops in the graph
        parent = self
        while parent is not None:
            if obj is parent:
                raise ValueError("creating circular dependency")
            parent = parent.parent
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
            for child in self._children:
                child.matrix_world_dirty = True
        if update_children:
            for child in self._children:
                child.update_matrix_world()
