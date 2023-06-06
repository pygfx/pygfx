import numpy as np

from ._base import Geometry
from .utils import merge


def generate_torso(
    radius_bottom,
    radius_top,
    height,
    radial_segments,
    height_segments,
    theta_start,
    theta_length,
):
    # compute POSITIONS assuming x-y horizontal plane and z up axis

    # radius for each vertex ring from bottom to top
    n_rings = height_segments + 1
    radii = np.linspace(radius_bottom, radius_top, num=n_rings, dtype=np.float32)

    # height for each vertex ring from bottom to top
    half_height = height / 2
    heights = np.linspace(-half_height, half_height, num=n_rings, dtype=np.float32)

    # to enable texture mapping to fully wrap around the cylinder,
    # we can't close the geometry and need a degenerate vertex
    n_vertices = radial_segments + 1

    # xy coordinates on unit circle for a single vertex ring
    theta = np.linspace(
        theta_start, theta_start + theta_length, num=n_vertices, dtype=np.float32
    )
    ring_xy = np.column_stack([np.cos(theta), np.sin(theta)])

    # put all the rings together
    positions = np.empty((n_rings, n_vertices, 3), dtype=np.float32)
    positions[..., :2] = ring_xy[None, ...] * radii[:, None, None]
    positions[..., 2] = heights[:, None]

    # the NORMALS are the same for every ring, so compute for only one ring
    # and then repeat
    slope = (radius_bottom - radius_top) / height
    ring_normals = np.empty(positions.shape[1:], dtype=np.float32)
    ring_normals[..., :2] = ring_xy
    ring_normals[..., 2] = slope
    ring_normals /= np.linalg.norm(ring_normals, axis=-1)[:, None]
    normals = np.empty_like(positions)
    normals[:] = ring_normals[None, ...]

    # the TEXTURE COORDS
    # u maps 0..1 to theta_start..theta_start+theta_length
    # v maps 0..1 to -height/2..height/2
    ring_u = (theta - theta_start) / theta_length
    ring_v = (heights / height) + 0.5
    texcoords = np.empty((n_rings, n_vertices, 2), dtype=np.float32)
    texcoords[..., 0] = ring_u[None, :]
    texcoords[..., 1] = ring_v[:, None]

    # the face INDEX
    # the amount of vertices
    indices = np.arange(n_rings * n_vertices, dtype=np.uint32).reshape(
        (n_rings, n_vertices)
    )
    # for every panel (height_segments, radial_segments) there is a quad (2, 3)
    index = np.empty((height_segments, radial_segments, 2, 3), dtype=np.uint32)
    # create a grid of initial indices for the panels
    index[:, :, 0, 0] = indices[
        np.arange(height_segments)[:, None], np.arange(radial_segments)[None, :]
    ]
    # the remainder of the indices for every panel are relative
    index[:, :, 0, 1] = index[:, :, 0, 0] + 1
    index[:, :, 0, 2] = index[:, :, 0, 0] + n_vertices
    index[:, :, 1, 0] = index[:, :, 0, 0] + n_vertices + 1
    index[:, :, 1, 1] = index[:, :, 1, 0] - 1
    index[:, :, 1, 2] = index[:, :, 1, 0] - n_vertices

    return (
        positions.reshape((-1, 3)),
        normals.reshape((-1, 3)),
        texcoords.reshape((-1, 2)),
        index.flatten(),
    )


def generate_cap(radius, height, radial_segments, theta_start, theta_length, up=True):
    # compute POSITIONS assuming x-y horizontal plane and z up axis

    # to enable texture mapping to fully wrap around the cylinder,
    # we can't close the geometry and need a degenerate vertex
    n_vertices = radial_segments + 1

    # xy coordinates on unit circle for vertex ring
    theta = np.linspace(
        theta_start, theta_start + theta_length, num=n_vertices, dtype=np.float32
    )
    ring_xy = np.column_stack([np.cos(theta), np.sin(theta)])

    # put the vertices together, inserting a center vertex at the start
    positions = np.empty((1 + n_vertices, 3), dtype=np.float32)
    positions[0, :2] = [0.0, 0.0]
    positions[1:, :2] = ring_xy * radius
    positions[..., 2] = height

    # the NORMALS
    normals = np.zeros_like(positions, dtype=np.float32)
    sign = int(up) * 2.0 - 1.0
    normals[..., 2] = sign

    # the TEXTURE COORDS
    # uv etches out a circle from the [0..1, 0..1] range
    # direction is reversed for up=False
    texcoords = np.empty((1 + n_vertices, 2), dtype=np.float32)
    texcoords[0] = [0.5, 0.5]
    texcoords[1:, 0] = ring_xy[:, 0] * 0.5 + 0.5
    texcoords[1:, 1] = ring_xy[:, 1] * 0.5 * sign + 0.5

    # the face INDEX
    indices = np.arange(n_vertices) + 1
    # for every radial segment there is a triangle (3)
    index = np.empty((radial_segments, 3), dtype=np.uint32)
    # create a grid of initial indices for the panels
    index[:, 0] = indices[np.arange(radial_segments)]
    # the remainder of the indices for every panel are relative
    index[:, 1 + int(up)] = n_vertices
    index[:, 2 - int(up)] = index[:, 0] + 1

    return (
        positions,
        normals,
        texcoords,
        index.flatten(),
    )


def cylinder_geometry(
    radius_bottom=1.0,
    radius_top=1.0,
    height=1.0,
    radial_segments=8,
    height_segments=1,
    theta_start=0.0,
    theta_length=np.pi * 2,
    open_ended=False,
):
    """Generate a cylinder or a cylinder segment.

    This function generates a cylinder or a cylinder segment. The cylinder's
    axis runs along the local z-axis, and its midpoint is located at the local
    origin. The cylinder's faces (top and bottom cap) are approximated by
    regular N-sided polygons, with corners on a circle of the given radius.

    Optionally, the cylinder's faces may be replaced with polygon approximations
    of two circle segments. In this case, a cylinder segment will be created,
    and each segment's arc is constructed from N equal-length line segments
    closed by a line along the segments cord.

    Parameters
    ----------
    radius_bottom : float
        The radius at the bottom of the cyliner.
    radius_top : float
        The radius at the top of the cyliner.
    height : float
        The height of the cylinder.
    radial_segments : int
        The number of segments to use when approximating the circle/arc.
    height_segments : int
        The number of evenly spaced segments into which the mantle should be
        split.
    theta_start : float
        The angle (in rad) at which to start the circle segment. Zero points
        into the direction of the local x-axis.
    theta_length : float
        The arc's central angle (in rad). Defaults to a full circle.
    open_ended : bool
        If True, the cylinder's faces are not added and the resulting geometry
        only contains the mantle.

    Returns
    -------
    cylinder : Geometry
        A geometry object representing a cylinder.
        Mathematically, it consists of a set of open orientable manifolds.

    """

    assert radial_segments > 0
    assert height_segments > 0
    assert theta_length != 0.0

    mesh = generate_torso(
        radius_bottom,
        radius_top,
        height,
        radial_segments,
        height_segments,
        theta_start,
        theta_length,
    )

    if not open_ended:
        groups = [mesh]
        if radius_bottom > 0:
            bottom_cap = generate_cap(
                radius_bottom,
                -height / 2,
                radial_segments,
                theta_start,
                theta_length,
                up=False,
            )
            groups.append(bottom_cap)
        if radius_top > 0:
            top_cap = generate_cap(
                radius_top,
                height / 2,
                radial_segments,
                theta_start,
                theta_length,
                up=True,
            )
            groups.append(top_cap)
        mesh = merge(groups)

    positions, normals, texcoords, indices = mesh
    return Geometry(
        indices=indices.reshape((-1, 3)),
        positions=positions,
        normals=normals,
        texcoords=texcoords,
    )


def cone_geometry(
    radius=1.0,
    height=1.0,
    radial_segments=8,
    height_segments=1,
    theta_start=0.0,
    theta_length=np.pi * 2,
    open_ended=False,
):
    """Generate a cone or a cone segment.

    This function generates a cone or a cone segment. The cone's
    axis runs along the local z-axis, and its midpoint is located at the local
    origin.

    The function is a thin wrapper around
    :func:`pygfx.geometries.cylinder_geometry` with ``radius_top = 0.0`` and
    slightly renamed arguments. For details, see the wrapped function.

    Parameters
    ----------
    radius : float
        The radius of the cone's bottom face.
    height : float
        The height of the cone.
    radial_segments : int
        The number of segments to use when approximating the circle/arc.
    height_segments : int
        The number of evenly spaced segments into which the mantle should be
        split.
    theta_start : float
        The angle (in rad) at which to start the circle segment. Zero points
        into the direction of the local x-axis.
    theta_length : float
        The arc's central angle (in rad). Defaults to a full circle.
    open_ended : bool
        If True, the cone's faces are not added and the resulting geometry
        only contains the mantle.

    Returns
    -------
    cylinder : Geometry
        A geometry object representing a cylinder.
        Mathematically, it consists of a set of open orientable manifolds.

    See Also
    --------
    pygfx.cylinder_geometry

    """

    return cylinder_geometry(
        radius_bottom=radius,
        radius_top=0.0,
        height=height,
        radial_segments=radial_segments,
        height_segments=height_segments,
        theta_start=theta_start,
        theta_length=theta_length,
        open_ended=open_ended,
    )
