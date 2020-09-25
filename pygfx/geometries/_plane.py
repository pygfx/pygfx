import numpy as np

from ..resources import Buffer
from ._base import Geometry


class PlaneGeometry(Geometry):
    def __init__(self, width=1, height=1, width_segments=1, height_segments=1):
        super().__init__()
        nx, ny = width_segments + 1, height_segments + 1

        x = np.linspace(-width / 2, width / 2, nx, dtype=np.float32)
        y = np.linspace(-height / 2, height / 2, ny, dtype=np.float32)
        xx, yy = np.meshgrid(x, y)
        xx, yy = xx.flatten(), yy.flatten()
        positions = np.column_stack([xx, yy, np.zeros_like(xx)])

        x = np.linspace(0, 1, nx, dtype=np.float32)
        y = np.linspace(0, 1, ny, dtype=np.float32)
        xx, yy = np.meshgrid(x, y)
        texcoords = np.column_stack([xx.flat, yy.flat])

        indices = np.zeros((height_segments, width_segments, 6), np.uint32)

        for y in range(height_segments):
            for x in range(width_segments):
                i = x + y * nx
                indices[y, x, :] = i, i + 1, i + nx, i + nx, i + 1, i + nx + 1

        indices = indices.ravel()

        self.positions = Buffer(positions, usage="vertex|storage")
        self.texcoords = Buffer(texcoords, usage="vertex|storage")
        self.index = Buffer(indices, usage="index|storage")
