import numpy as np

from ._base import Geometry


# Note that we keep this function separate, because its used by other geometry-generating functions.


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


def plane_geometry(width=1, height=1, width_segments=1, height_segments=1):
    """Generate a plane.

    Creates a flat (2D) rectangle in the local xy-plane that has its center at
    local origin. The plane may be subdivided into segments along the x- or
    y-axis respectively.

    Parameters
    ----------
    width : float
        The plane's width measured along the x-axis.
    height : float
        The plane's width measured along the y-axis.
    width_segments : int
        The number of evenly spaced segments along the x-axis into which the
        plane should be divided.
    height_segments : int
        The number of evenly spaced segments along the y-axis into which the
        plane should be divided.

    Returns
    -------
    plane : Geometry
        A geometry object representing the requested plane.
        Mathematically, it is an open orientable manifold.

    """

    positions, normals, texcoords, indices = generate_plane(
        width, height, width_segments, height_segments
    )

    return Geometry(
        indices=indices.reshape((-1, 3)),
        positions=positions,
        normals=normals,
        texcoords=texcoords,
    )


def mobius_strip_geometry(radius=1, width=0.5, strip_segments=64, stitch=True):
    """Generate a Möbius strip.

    The Möbios strip is a surface that can be formed by attaching the
    ends of a strip of paper together with a half-twist.

    Parameters
    ----------
    radius : float
        The radius of the circe along which the strip is positioned.
    width : float
        The width of the strip.
    strip_segments : int
        The number of evenly spaced segments along the strip.
    stitch : bool
        Whether to stitch the ends together to form a closed loop. If False (default)
        the strip's ends meet, but are not actually attached to one another.

    Returns
    -------
    strip : Geometry
        A geometry object representing the Möbius strip.
        Mathematically, it is an open manifold, that is only orientable
        if ``stitch`` is False.

    """

    # Check/convert inputs
    n_strip = int(strip_segments)
    radius = float(radius)
    width = float(width)
    stitch = bool(stitch)

    n_verts = n_strip + (0 if stitch else 1)

    # Create base vectors
    t = np.linspace(0, np.pi * 2, n_verts, endpoint=not stitch, dtype=np.float32)
    u = np.linspace(-width / 2, width / 2, 2, endpoint=True, dtype=np.float32)
    u, t = np.meshgrid(u, t)

    # The math
    x = radius * np.cos(t) + u * np.sin(0.5 * t)
    z = radius * np.sin(t)
    y = u * np.cos(0.5 * t)

    # Define connectivity
    indices = np.array([0, 1, 2, 2, 1, 3], np.uint32)
    indices = np.tile(indices, (n_strip, 1, 1))
    indices += np.arange(0, n_strip * 2, 2, dtype=np.uint32).reshape(n_strip, 1, 1)

    # Stitch the ends
    if stitch:
        indices[-1, :, (2, 3, 5)] = np.array((1, 1, 0)).reshape(3, 1)

    return Geometry(
        indices=indices.reshape((-1, 3)),
        positions=np.column_stack([x.flat, y.flat, z.flat]),
    )
