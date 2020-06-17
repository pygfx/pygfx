import numpy as np

from ..datawrappers import Buffer
from ._base import Geometry


class BoxGeometry(Geometry):
    def __init__(self, width, height, depth):
        super().__init__()

        positions = np.array(
            [
                # top (0, 0, 1)
                [-1, -1, 1, 1],
                [1, -1, 1, 1],
                [1, 1, 1, 1],
                [-1, 1, 1, 1],
                # bottom (0, 0, -1)
                [-1, 1, -1, 1],
                [1, 1, -1, 1],
                [1, -1, -1, 1],
                [-1, -1, -1, 1],
                # right (1, 0, 0)
                [1, -1, -1, 1],
                [1, 1, -1, 1],
                [1, 1, 1, 1],
                [1, -1, 1, 1],
                # left (-1, 0, 0)
                [-1, -1, 1, 1],
                [-1, 1, 1, 1],
                [-1, 1, -1, 1],
                [-1, -1, -1, 1],
                # front (0, 1, 0)
                [1, 1, -1, 1],
                [-1, 1, -1, 1],
                [-1, 1, 1, 1],
                [1, 1, 1, 1],
                # back (0, -1, 0)
                [1, -1, 1, 1],
                [-1, -1, 1, 1],
                [-1, -1, -1, 1],
                [1, -1, -1, 1],
            ],
            dtype="f4",
        )
        positions[:, 0] *= width / 2
        positions[:, 1] *= height / 2
        positions[:, 2] *= depth / 2

        texcoords = np.array(
            [
                # top (0, 0, 1)
                [0, 0],
                [1, 0],
                [1, 1],
                [0, 1],
                # bottom (0, 0, -1)
                [1, 0],
                [0, 0],
                [0, 1],
                [1, 1],
                # right (1, 0, 0)
                [0, 0],
                [1, 0],
                [1, 1],
                [0, 1],
                # left (-1, 0, 0)
                [1, 0],
                [0, 0],
                [0, 1],
                [1, 1],
                # front (0, 1, 0)
                [1, 0],
                [0, 0],
                [0, 1],
                [1, 1],
                # back (0, -1, 0)
                [0, 0],
                [1, 0],
                [1, 1],
                [0, 1],
            ],
            dtype="f4",
        )

        indices = np.array(
            [
                [0, 1, 2, 2, 3, 0],  # top
                [4, 5, 6, 6, 7, 4],  # bottom
                [8, 9, 10, 10, 11, 8],  # right
                [12, 13, 14, 14, 15, 12],  # left
                [16, 17, 18, 18, 19, 16],  # front
                [20, 21, 22, 22, 23, 20],  # back
            ],
            dtype=np.uint32,
        ).flatten()

        self.positions = Buffer(positions, usage="vertex|storage")
        self.texcoords = Buffer(texcoords, usage="vertex|storage")
        self.index = Buffer(indices, usage="index|storage")
