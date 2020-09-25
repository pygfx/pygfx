import numpy as np

from .. import Geometry, Group, Line, LineSegmentMaterial


class AxesHelper(Group):
    def __init__(self, size=1.0, thickness=6.0):
        super().__init__()

        self.size = size

        positions = (
            np.array(
                [
                    [[0, 0, 0], [1, 0, 0]],
                    [[0, 0, 0], [0, 1, 0]],
                    [[0, 0, 0], [0, 0, 1]],
                ],
                dtype=np.float32,
            )
            * self.size
        )

        colors = np.array(
            [
                [1, 0.6, 0, 1],  # x is orange-ish
                [0.6, 1, 0, 1],  # y is yellow-ish
                [0, 0.6, 1, 1],  # z is blue-ish
            ],
            dtype=np.float32,
        )

        for pos, color in zip(positions, colors):
            geometry = Geometry(positions=pos)
            material = LineSegmentMaterial(thickness=thickness, color=color)
            self.add(Line(geometry, material))
        # TODO: draw each axis with a different color using vertex coloring
        # instead of three subnodes
