from ..linalg import Matrix4
from ..objects._world_object import WorldObject


class Camera(WorldObject):
    def __init__(self):
        super().__init__()

        self.matrix_world_inverse = Matrix4()
        self.projection_matrix = Matrix4()
        self.projection_matrix_inverse = Matrix4()
