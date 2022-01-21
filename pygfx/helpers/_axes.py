import numpy as np

from .. import Geometry, Line, LineThinSegmentMaterial
from ..utils import Color


class AxesHelper(Line):
    """An object indicating the axes directions."""

    def __init__(self, length=1.0):
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
        material = LineThinSegmentMaterial(vertex_colors=True)

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
