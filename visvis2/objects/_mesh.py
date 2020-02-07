from ._world_object import WorldObject


class Mesh(WorldObject):
    def __init__(self, geometry, material):
        self.geometry = geometry
        self.material = material
