import numpy as np

from ._base import Geometry


def sphere_geometry(
    radius=1,
    width_segments=32,
    height_segments=16,
    phi_start=0,
    phi_length=np.pi * 2,
    theta_start=0,
    theta_length=np.pi,
):
    """Generate a sphere.

    Creates a sphere that has its center in the local origin. The sphere is
    constructed by creating a grid of longituidnal and latitudinal lines and
    placing vetices at the intersections. The area between 4 vertices is then
    approximated by a rectangular-shaped face.

    Optionally, the geometry can be limited to a segment of the full sphere
    described by a longituidnal and latitudinal arc.

    Parameters
    ----------
    radius : float
        The radius of the sphere. Vertices are placed at this distance around
        the local origin.
    width_segments : int
        The number of (evenly-spaced) longitudinal lines to draw within the
        selected sphere segment.
    height_segments : int
        The number of (evenly-spaced) latitudinal lines to draw within the
        selected sphere segment.
    phi_start : float
        The angle (in rad) at which to start the longitudinal lines. Zero means
        the first line is drawn in the xz-plane.
    phi_length : float
        The central angle (in rad) of the sphere segments latitudinal segment.
    theta_start : float
        The angle (in rad) at which to start the latitudinal lines. Zero means
        the first line is drawn at the "tip" of the sphere, i.e., at ``(0, 0,
        radius)``.
    theta_length : float
        The central angle (in rad) of the sphere segment's longitudinal segment.

    Returns
    -------
    sphere : Geometry
        A geometry object that represents the requested sphere or sphere segment.
        Mathematically, it is an open orientable manifold.

    """

    # create grid of spherical coordinates
    nx = width_segments + 1
    phi_end = phi_start + phi_length
    phi = np.linspace(phi_start, phi_end, num=nx, dtype=np.float32)

    ny = height_segments + 1
    theta_end = theta_start + theta_length
    theta = np.linspace(theta_start, theta_end, num=ny, dtype=np.float32)

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
    idx = np.arange(nx * ny, dtype=np.uint32).reshape((ny, nx))
    # for every panel (height_segments, width_segments) there is a quad (2, 3)
    indices = np.empty((height_segments, width_segments, 2, 3), dtype=np.uint32)
    # create a grid of initial idx for the panels
    indices[:, :, 0, 0] = idx[
        np.arange(height_segments)[:, None], np.arange(width_segments)[None, :]
    ]
    # the remainder of the indices for every panel are relative
    indices[:, :, 0, 1] = indices[:, :, 0, 0] + nx
    indices[:, :, 0, 2] = indices[:, :, 0, 0] + 1
    indices[:, :, 1, 0] = indices[:, :, 0, 0] + nx + 1
    indices[:, :, 1, 1] = indices[:, :, 1, 0] - nx
    indices[:, :, 1, 2] = indices[:, :, 1, 0] - 1

    return Geometry(
        indices=indices.reshape((-1, 3)),
        positions=positions.reshape((-1, 3)),
        normals=normals.reshape((-1, 3)),
        texcoords=texcoords.reshape((-1, 2)),
    )
