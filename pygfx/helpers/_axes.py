import numpy as np

from .. import Geometry, Line, LineSegmentMaterial


class AxesHelper(Line):
    def __init__(self, length=1.0, thickness=6.0):
        positions = np.array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [0, 0, 0],
                [0, 1, 0],
                [0, 0, 0],
                [0, 0, 1],
            ],
            dtype="f4",
        )
        positions *= length

        colors = np.array(
            [
                [1, 0.6, 0, 1],
                [1, 0.6, 0, 1],  # x is orange-ish
                [0.6, 1, 0, 1],
                [0.6, 1, 0, 1],  # y is yellow-ish
                [0, 0.6, 1, 1],
                [0, 0.6, 1, 1],  # z is blue-ish
            ],
            dtype="f4",
        )

        geometry = Geometry(positions=positions, colors=colors)
        material = LineSegmentMaterial(thickness=thickness, vertex_colors=True)

        super().__init__(geometry, material)
