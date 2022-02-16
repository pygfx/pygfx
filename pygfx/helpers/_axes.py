import numpy as np

from .. import Geometry, Line, LineSegmentMaterial
from ..utils import Color


class AxesHelper(Line):
    """An object indicating the axes directions.

    Parameters:
        size (float): The length of the lines (default 1).
        thickness (float): the thickness of the lines (default 2 px).
    """

    def __init__(self, size=1.0, thickness=2):
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
        positions *= size

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
        material = LineSegmentMaterial(vertex_colors=True, thickness=thickness, aa=True)

        super().__init__(geometry, material)

    def set_colors(self, x, y, z):
        x, y, z = Color(x), Color(y), Color(z)
        self._geometry.colors.data[0] = x
        self._geometry.colors.data[1] = x
        self._geometry.colors.data[2] = y
        self._geometry.colors.data[3] = y
        self._geometry.colors.data[4] = z
        self._geometry.colors.data[5] = z
        self._geometry.colors.update_range(0, self._geometry.colors.nitems)
