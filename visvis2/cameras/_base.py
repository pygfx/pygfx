from ..linalg import Matrix4
from ..objects._base import WorldObject


class Camera(WorldObject):
    """ Abstract base camera.
    """

    def __init__(self):
        super().__init__()

        self.matrix_world_inverse = Matrix4()
        self.projection_matrix = Matrix4()
        self.projection_matrix_inverse = Matrix4()

    def set_viewport_size(self, width, height):
        # In logical pixels
        pass

    def update_matrix_world(self, *args, **kwargs):
        super().update_matrix_world(*args, **kwargs)
        self.matrix_world_inverse.get_inverse(self.matrix_world)

    def update_projection_matrix(self):
        raise NotImplementedError()


class NDCCamera(Camera):
    """ A Camera operating in NDC coordinates; it's projection matrix
    is the identity transform (but it's matrix_world can still be set).
    """

    def update_projection_matrix(self):
        eye = 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1
        self.projection_matrix.set(*eye)
        self.projection_matrix_inverse.set(*eye)


class ScreenCoordsCamera(Camera):
    """ A Camera operating in screen coordinates.
    """

    def __init__(self):
        super().__init__()
        self._width = 1
        self._height = 1

    def set_viewport_size(self, width, height):
        self._width = width
        self._height = height

    def update_projection_matrix(self):
        sx, sy, sz = 2 / self._width, 2 / self._height, 1
        dx, dy, dz = -1, -1, 0
        m = sx, 0, 0, dx, 0, sy, 0, dy, 0, 0, sz, dz, 0, 0, 0, 1
        self.projection_matrix.set(*m)
        self.projection_matrix_inverse.get_inverse(self.projection_matrix)
