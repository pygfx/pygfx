import numpy as np


# Scale, Rotate, Translate (SRT), gebeuren altijd in die volgorde. Dus
# als je object.rotate_x(); object.translate_y() aanroept, is de volgorde
# onbelangrijk -> de (position en quaternion) properties worden aangepast, verder niet.

class WorldObject:
    """ The base class for objects present in the "world", i.e. the scene graph.
    """

    def __init__(self):
        self._children = []
        self._matrix = np.eye(4)
        self._dirty = True

    @property
    def children(self):
        return self._children

    def add_child(self, world_object):
        self._children.append(world_object)
        self._dirty = True

    def remove_child(self, world_object):
        while world_object in self._children:
            self._children.remove(world_object)
        self._dirty = True

    def update_matrix(self):
        # todo: or simply compare the tuples/hash of positon, quat, scale of last update
        # todo: or notify the renderer that we are dirty
        if self._matrix_dirty:
            self._matrix = compose_matrix(self._scale, self._quaternion, self._position)

    def update_matrix_world(self, parent_matrix_world):
        self.update_matrix()
        self._matrix_world = self._matrix * parent_matrix_world
        for child in self._children:
            child.update_matrix_world(self._matrix_world)

    @property
    def matrix(self):
        return self._matrix

    # todo: dit betekend dus dat een child maar 1x in de scene mag zijn
    @property
    def matrix_world(self):
        return self._matrix_world

    @property
    def position(self):
        return self._position

    @property
    def quaternion(self):
        return self._quaternion

    @property
    def euler(self):
        """ Object rotation in Euler angles (xyz order).
        """
        return self._euler

    @property
    def scale(self):
        return self._scale

    def set_rotation_from_matrix():
        pass

    def set_rotation_from_quaternion():
        self._matrix_dirty = True

    def set_rotation(self, ):
        pass

    def rotate_x(self):
        pass

    def rotate_y(self):
        pass

    def rotate_whatever():
        pass

    def translate(self, x, y, z):
        pass
