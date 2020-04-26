import numpy as np

from ..datawrappers import BufferWrapper
from ._base import Geometry


class BoxGeometry(Geometry):
    def __init__(self, width, height, depth):
        super().__init__()

        texcoords = np.array(
            [
                [0, 0, 0, 1],
                [0, 0, 1, 1],
                [0, 1, 0, 1],
                [0, 1, 1, 1],
                [1, 0, 0, 1],
                [1, 0, 1, 1],
                [1, 1, 0, 1],
                [1, 1, 1, 1],
            ],
            dtype="f4",
        )

        positions = texcoords.copy()
        positions[:, 0] = (positions[:, 0] - 0.5) * width
        positions[:, 1] = (positions[:, 1] - 0.5) * height
        positions[:, 2] = (positions[:, 2] - 0.5) * depth

        index = np.array(
            [
                0,
                4,
                2,
                4,
                6,
                2,
                0,
                2,
                1,
                1,
                2,
                3,
                5,
                6,
                4,
                5,
                7,
                6,
                2,
                6,
                3,
                6,
                7,
                3,
                1,
                3,
                5,
                5,
                3,
                7,
                4,
                1,
                0,
                4,
                5,
                1,
            ],
            dtype="u2",
        )

        self.positions = BufferWrapper(positions, usage="vertex|storage")
        self.texcoords = BufferWrapper(texcoords, usage="vertex|storage")
        self.index = BufferWrapper(index, usage="index|storage")
