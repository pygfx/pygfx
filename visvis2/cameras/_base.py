from ..linalg import Matrix4
from ..objects._world_object import WorldObject


class Camera(WorldObject):
    def __init__(self):
        super().__init__()
        self.projectionMatrix = Matrix4()


class IdentityCamera(Camera):
    def __init__(self):
        super().__init__()
        self.projectionMatrix.identity()
