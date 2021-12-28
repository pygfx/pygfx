import numpy as np

from .. import Geometry, Line, LineThinSegmentMaterial


class BoxHelper(Line):
    """An object visualizing a box."""

    def __init__(self, size=1.0):
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
        positions *= size

        geometry = Geometry(positions=positions)
        material = LineThinSegmentMaterial(color=(1, 0, 0, 1))

        super().__init__(geometry, material)
