import numpy as np

from ..resources import Buffer
from ._base import Geometry


def generate_sphere(
    radius,
    width_segments,
    height_segments,
    phi_start,
    phi_length,
    theta_start,
    theta_length,
):
    # create grid of spherical coordinates
    nx = width_segments + 1
    phi_end = phi_start + phi_length
    phi = np.linspace(phi_start, phi_end, num=nx, dtype="f4")

    ny = height_segments + 1
    theta_end = theta_start + theta_length
    theta = np.linspace(theta_start, theta_end, num=ny, dtype="f4")

    # grid has shape (ny, nx)
    phi_grid, theta_grid = np.meshgrid(phi, theta)

    # convert to cartesian coordinates
    theta_grid_sin = np.sin(theta_grid)
    xx = np.cos(phi_grid) * theta_grid_sin * -1
    yy = np.cos(theta_grid)
    zz = np.sin(phi_grid) * theta_grid_sin

    # construct normals and positions
    normals = np.stack([xx, yy, zz], axis=-1)
    positions = normals * radius

    # construct texture coordinates
    # u maps 0..1 to phi_start..phi_start+phi_length
    # v maps 1..0 to theta_start..theta_start+theta_length
    uu = (phi_grid - phi_start) / phi_length
    vv = 1 - ((theta_grid - theta_start) / theta_length)
    texcoords = np.stack([uu, vv], axis=-1)

    # the face indices
    # assign an index to every vertex on the grid
    indices = np.arange(nx * ny, dtype="u4").reshape((ny, nx))
    # for every panel (height_segments, width_segments) there is a quad (2, 3)
    index = np.empty((height_segments, width_segments, 2, 3), dtype="u4")
    # create a grid of initial indices for the panels
    index[:, :, 0, 0] = indices[
        np.arange(height_segments)[:, None], np.arange(width_segments)[None, :]
    ]
    # the remainder of the indices for every panel are relative
    index[:, :, 0, 1] = index[:, :, 0, 0] + nx
    index[:, :, 0, 2] = index[:, :, 0, 0] + 1
    index[:, :, 1, 0] = index[:, :, 0, 0] + nx + 1
    index[:, :, 1, 1] = index[:, :, 1, 0] - nx
    index[:, :, 1, 2] = index[:, :, 1, 0] - 1

    return (
        positions.reshape((-1, 3)),
        normals.reshape((-1, 3)),
        texcoords.reshape((-1, 2)),
        index.flatten(),
    )


class SphereGeometry(Geometry):
    """A geometry defining a Sphere."""

    def __init__(
        self,
        radius=1,
        width_segments=32,
        height_segments=16,
        phi_start=0,
        phi_length=np.pi * 2,
        theta_start=0,
        theta_length=np.pi,
    ):
        super().__init__()

        vertices, normals, texcoords, indices = generate_sphere(
            radius=radius,
            width_segments=width_segments,
            height_segments=height_segments,
            phi_start=phi_start,
            phi_length=phi_length,
            theta_start=theta_start,
            theta_length=theta_length,
        )

        self.positions = Buffer(vertices, usage="vertex|storage")
        self.normals = Buffer(normals, usage="vertex|storage")
        self.texcoords = Buffer(texcoords, usage="vertex|storage")
        self.index = Buffer(indices, usage="index|storage")
