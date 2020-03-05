from ._base import WorldObject


class Mesh(WorldObject):
    def __init__(self, geometry, material):
        super().__init__()
        self.geometry = geometry
        self.material = material
