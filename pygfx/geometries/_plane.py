import numpy as np

from ..resources import Buffer
from ._base import Geometry


def generate_plane(width, height, width_segments, height_segments):
    nx, ny = width_segments + 1, height_segments + 1

    x = np.linspace(-width / 2, width / 2, nx, dtype=np.float32)
    y = np.linspace(-height / 2, height / 2, ny, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    xx, yy = xx.flatten(), yy.flatten()
    positions = np.column_stack([xx, yy, np.zeros_like(xx)])

    dim = np.array([width, height], dtype=np.float32)
    texcoords = (positions[..., :2] + dim / 2) / dim
    texcoords[..., 1] = 1 - texcoords[..., 1]

    # the amount of vertices
    indices = np.arange(ny * nx, dtype=np.uint32).reshape((ny, nx))
    # for every panel (height_segments, width_segments) there is a quad (2, 3)
    index = np.empty((height_segments, width_segments, 2, 3), dtype=np.uint32)
    # create a grid of initial indices for the panels
    index[:, :, 0, 0] = indices[
        np.arange(height_segments)[:, None], np.arange(width_segments)[None, :]
    ]
    # the remainder of the indices for every panel are relative
    index[:, :, 0, 1] = index[:, :, 0, 0] + 1
    index[:, :, 0, 2] = index[:, :, 0, 0] + nx
    index[:, :, 1, 0] = index[:, :, 0, 0] + nx + 1
    index[:, :, 1, 1] = index[:, :, 1, 0] - 1
    index[:, :, 1, 2] = index[:, :, 1, 0] - nx

    normals = np.tile(np.array([0, 0, 1], dtype=np.float32), (ny * nx, 1))

    return positions, normals, texcoords, index.reshape((-1, 3))


class PlaneGeometry(Geometry):
    """A geometry defining a one-dimensional plane."""

    def __init__(self, width=1, height=1, width_segments=1, height_segments=1):
        super().__init__()

        positions, normals, texcoords, index = generate_plane(
            width, height, width_segments, height_segments
        )

        self.positions = Buffer(positions)
        self.normals = Buffer(normals)
        self.texcoords = Buffer(texcoords)
        self.index = Buffer(index.reshape((-1, 3)))
