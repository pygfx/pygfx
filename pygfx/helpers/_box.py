import numpy as np

from .. import Geometry, Line, LineThinSegmentMaterial


class BoxHelper(Line):
    """An object visualizing a box."""

    def __init__(self, size=1.0):
        self._size = size

        positions = np.array(
            [
                [0, 0, 0],  # bottom edges
                [1, 0, 0],
                [0, 0, 0],
                [0, 0, 1],
                [1, 0, 1],
                [1, 0, 0],
                [1, 0, 1],
                [0, 0, 1],
                [0, 1, 0],  # top edges
                [1, 1, 0],
                [0, 1, 0],
                [0, 1, 1],
                [1, 1, 1],
                [1, 1, 0],
                [1, 1, 1],
                [0, 1, 1],
                [0, 0, 0],  # side edges
                [0, 1, 0],
                [1, 0, 0],
                [1, 1, 0],
                [0, 0, 1],
                [0, 1, 1],
                [1, 0, 1],
                [1, 1, 1],
            ],
            dtype="f4",
        )
        positions -= 0.5
        positions *= self._size

        geometry = Geometry(positions=positions)
        material = LineThinSegmentMaterial(color=(1, 0, 0, 1))

        super().__init__(geometry, material)

    def set_object_world(self, object):
        aabb = object.get_world_bounding_box()
        self.set_aabb(aabb)

    def set_object_local(self, object):
        if object.geometry:
            aabb = object.geometry.bounding_box()
        else:
            aabb = None
        self.set_aabb(aabb)

    def set_aabb(self, aabb):
        if aabb is None:
            raise ValueError("Object has no geometry")

        diagonal = aabb[1] - aabb[0]
        center = aabb[0] + diagonal * 0.5
        scale = diagonal / self._size

        self.position.set(*center)
        self.scale.set(*scale)
