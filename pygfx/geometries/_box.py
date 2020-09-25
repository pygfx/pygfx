import numpy as np

from ..resources import Buffer
from ._base import Geometry


class BoxGeometry(Geometry):
    def __init__(self, width, height, depth):
        super().__init__()

        # Assuming this coordinate frame:
        #
        # z y
        # |/
        #  -- x
        #
        # The first eight vertices, defining the right and left side
        # of the box, are as follows:
        #
        #    5----0
        #   /|   /|
        #  7----2 |
        #  | 4--|-1
        #  |/   |/
        #  6----3
        #
        # The back and front again have the eight corners of the box, but
        # in a different order ... and dito for the top and bottom.

        positions = np.array(
            [
                # right (1, 0, 0)
                [1, 1, 1],
                [1, 1, -1],
                [1, -1, 1],
                [1, -1, -1],
                # left (-10, 0, 0)
                [-1, 1, -1],
                [-1, 1, 1],
                [-1, -1, -1],
                [-1, -1, 1],
                # back (0, 1, 0)
                [-1, 1, -1],
                [1, 1, -1],
                [-1, 1, 1],
                [1, 1, 1],
                # front (0, -1, 0)
                [-1, -1, 1],
                [1, -1, 1],
                [-1, -1, -1],
                [1, -1, -1],
                # top (0, 1, 0)
                [-1, 1, 1],
                [1, 1, 1],
                [-1, -1, 1],
                [1, -1, 1],
                # bottom (0, -1, 0)
                [1, 1, -1],
                [-1, 1, -1],
                [1, -1, -1],
                [-1, -1, -1],
            ],
            dtype="f4",
        )
        positions[:, 0] *= width / 2
        positions[:, 1] *= height / 2
        positions[:, 2] *= depth / 2

        texcoords = np.array(
            [
                # right
                [0, 1],
                [1, 1],
                [0, 0],
                [1, 0],
                # left
                [0, 1],
                [1, 1],
                [0, 0],
                [1, 0],
                # back
                [0, 1],
                [1, 1],
                [0, 0],
                [1, 0],
                # front
                [0, 1],
                [1, 1],
                [0, 0],
                [1, 0],
                # top
                [0, 1],
                [1, 1],
                [0, 0],
                [1, 0],
                # bottom
                [0, 1],
                [1, 1],
                [0, 0],
                [1, 0],
            ],
            dtype="f4",
        )

        indices = np.array(
            [
                [0, 2, 1, 2, 3, 1],
                [4, 6, 5, 6, 7, 5],
                [8, 10, 9, 10, 11, 9],
                [12, 14, 13, 14, 15, 13],
                [16, 18, 17, 18, 19, 17],
                [20, 22, 21, 22, 23, 21],
            ],
            dtype=np.uint32,
        ).flatten()

        self.positions = Buffer(positions, usage="vertex|storage")
        self.texcoords = Buffer(texcoords, usage="vertex|storage")
        self.index = Buffer(indices, usage="index|storage")
        # todo: rename to indices?
