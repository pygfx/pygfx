from ..linalg import Matrix4
from ..objects._world_object import WorldObject


class Camera(WorldObject):
    def __init__(self):
        super().__init__()

        self.matrixWorldInverse = Matrix4()
        self.projectionMatrix = Matrix4()
        self.projectionMatrixInverse = Matrix4()
