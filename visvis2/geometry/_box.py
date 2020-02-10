import numpy as np

from ._base import Geometry


class BoxGeometry(Geometry):
    def __init__(self, width, height, depth):
        super().__init__()

        half_width = width / 2
        half_height = height / 2
        half_depth = depth / 2
        vertices = np.array(
            [
                -half_width,
                -half_height,
                -half_depth,
                -half_width,
                -half_height,
                half_depth,
                -half_width,
                half_height,
                -half_depth,
                -half_width,
                half_height,
                half_depth,
                half_width,
                -half_height,
                -half_depth,
                half_width,
                -half_height,
                half_depth,
                half_width,
                half_height,
                -half_depth,
                half_width,
                half_height,
                half_depth,
            ],
            dtype="f4",
        )
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
            dtype="i4",
        )
        self.buffers["position"] = vertices
        self.index = index
