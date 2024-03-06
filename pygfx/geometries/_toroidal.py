import numpy as np

from ._base import Geometry


def klein_bottle_geometry(scale=1.0, stitch=False):
    """Generate a Klein bottle.

    The Klein bottle is a surface for which the inside and outside
    are the same, similar to a Möbius strip. In fact, a Klein bottle
    can be constructed by glueing together two Möbius strips.

    Parameters
    ----------
    scale : float
        The scale of the bottle.
    stitch : bool
        Whether to stitch the ends together to produce a closed
        manifold. If True, a mathematically correct Klein bottle is
        produced. If False (default) an approximation is produced where
        the ends of the bottle meet, but are not actually connected.

    Returns
    -------
    klein_bottle : Geometry
        A geometry object representing the requested klein bottle.
        Mathematically, it is either an "orientable open manifold" or
        a "non-orientable closed manifold", depending on the ``stitch``
        parameter.

    """

    # This is an interesting object for mathematicians. For us it's
    # interesting because we can test whether our lighting etc. deals
    # correctly with objects for which the "inside" must also be shown.

    # The number of vertices is nxn
    n = 40

    # Get 2D surface in 3D space
    u = np.linspace(0, 2 * np.pi, n, endpoint=True, dtype=np.float32)
    v = np.linspace(0, 2 * np.pi, n, endpoint=False, dtype=np.float32)
    ux, vx = np.meshgrid(u, v)
    x, y, z = klein_bottle_surface(ux, vx)

    # Scaled to a unit cube, then scale to width / height / depth
    # x = (x + 1.66559) * (0.0437597 * width)
    # y = (y - 2.04939) * (0.0277017 * height)
    # z = (z + 0.00000) * (0.0833333 * depth)
    # Scaled to fit inside a unit cube, but maintain original proportions
    x = (x + 1.66559) * (0.0833333 * scale)
    y = (y - 2.04939) * (0.0833333 * scale)
    z = (z + 0.00000) * (0.0833333 * scale)

    # Put into an Nx4 array
    positions = np.empty((x.size, 3), np.float32)
    positions[:, 0] = x.flat
    positions[:, 1] = y.flat
    positions[:, 2] = z.flat

    # Texcoords are easy
    texcoords = np.column_stack([ux.flat, vx.flat]).astype(np.float32, copy=False)
    texcoords *= 1 / (2 * np.pi)

    # Map indices
    # Two triangles onto the "top-left" rectangle (six vertices)
    indices = np.array([0, 1, n + 1, n + 1, n, 0], np.uint32)
    # Replicate to all rectangles, add offsets
    indices = np.tile(indices, (n, n - 1, 1))
    gx, gy = np.meshgrid(
        np.arange(indices.shape[1], dtype=np.uint32),
        n * np.arange(indices.shape[0], dtype=np.uint32),
    )
    indices += (gx + gy).reshape(indices.shape[:2] + (1,))

    # Stitch the faces at the tube's edge.
    indices[-1, :, 2:5] -= n * n

    if stitch:
        # We can do the same for the tube's ends, and make this a closed
        # manifold! Note that in doing that, we also make it
        # not-orientable - the orientation of the faces (i.e. the
        # winding) switches where we apply the stitch. Also note that
        # in the current implementation we're left with n unused
        # vertices.

        # In the code below, i1 are one end of the tube, and i2 the
        # other. Since the tube is inside-out, the matching pairs are
        # on opposing sides.
        for i in range(n):
            i2 = n - 1 + i * n
            i1 = (n // 2 - i) * n
            i1 = i1 if i1 >= 0 else i1 + positions.shape[0]
            indices[indices == i2] = i1

    # Create buffers for this geometry
    indices = indices.reshape((-1, 3))
    return Geometry(indices=indices, positions=positions, texcoords=texcoords)


def klein_bottle_surface(u, v):
    """
    http://paulbourke.net/geometry/toroidal/

        A German topologist named Klein
        Thought the Möbius Loop was divine
        Said he, "If you glue
        The edges of two
        You get a weird bottle like mine.
    """
    half = (0 <= u) & (u < np.pi)
    r = 4 * (1 - np.cos(u) / 2)
    x = 6 * np.cos(u) * (1 + np.sin(u)) + r * np.cos(v + np.pi)
    x[half] = (6 * np.cos(u) * (1 + np.sin(u)) + r * np.cos(u) * np.cos(v))[half]
    y = 16 * np.sin(u)
    y[half] = (16 * np.sin(u) + r * np.sin(u) * np.cos(v))[half]
    z = r * np.sin(v)
    return x, y, z


def torus_knot_geometry(
    scale=1.0, tube=0.4, tubular_segments=64, radial_segments=8, p=2, q=3, stitch=False
):
    """Generate a torus knot.

    Create geometry representing a torus knot, the particular shape of which is
    defined by a pair of coprime integers, p and q. If p and q are not coprime,
    the result will be a torus link.

    Parameters
    ----------
    scale : float
        The scale of the torus, default 1.
    tube : float
        The radius of the tube. Default 0.4.
    tubular_segments : int
        default is 64.
    radial_segments : int
        default is 8.
    p : int
        How many times the geometry winds around its axis of
        rotational symmetry. Default 2.
    q : int
        How many times the geometry winds around a circle in
        the interior of the torus. Default 3.
    stitch : bool
        Whether to stitch the ends together to produce a closed
        manifold. Default False. If False, the mesh is basically a
        curved surface with the edges meeting to make it visually
        closed, which works better for texturing. Set to True for a
        mathematically closed object.

    Returns
    -------
    torus : Geometry
        A geometry object representing the requested torus.
        Mathematically, it is an open orientable manifold, which can
        be closed with the ``stitch`` parameter.

    """

    # If we do not stitch, the two ends meet (i.e. duplicate vertices)
    # to make the mesh look closed (while mathematically it is not).

    if stitch:
        tubular_verts = tubular_segments
        radial_verts = radial_segments
    else:
        tubular_verts = tubular_segments + 1
        radial_verts = radial_segments + 1

    # Define base factors
    u = np.linspace(
        0, p * 2 * np.pi, tubular_verts, endpoint=not stitch, dtype=np.float32
    )
    v = np.linspace(0, 2 * np.pi, radial_verts, endpoint=not stitch, dtype=np.float32)

    # Get positions along the torus' center, and a tiny step further
    pos1 = torus_knot_surface(u, p, q, scale)
    pos2 = torus_knot_surface(u + 0.01, p, q, scale)

    # Two vectors along the torus' centerline
    vec1 = np.ndarray.astype(pos1 - pos2, np.float32, copy=False)
    vec2 = np.ndarray.astype(pos1 + pos2, np.float32, copy=False)

    # Two vectors orthoginal to the torus' centerline
    vec3 = np.cross(vec1, vec2)
    vec4 = np.cross(vec3, vec1)
    # Normalize
    vec3 /= ((vec3[:, 0] ** 2 + vec3[:, 1] ** 2 + vec3[:, 2] ** 2) ** 0.5).reshape(
        -1, 1
    )
    vec4 /= ((vec4[:, 0] ** 2 + vec4[:, 1] ** 2 + vec4[:, 2] ** 2) ** 0.5).reshape(
        -1, 1
    )
    # Define positions relative to the centerline
    cx = -tube * np.cos(v)
    cy = +tube * np.sin(v)
    # Prepare shapes, so we can do numpy broadcast
    pos = pos1.reshape(-1, 1, 3)
    cx.shape = 1, -1, 1
    cy.shape = 1, -1, 1
    vec3.shape = -1, 1, 3
    vec4.shape = -1, 1, 3
    # Broadcast!
    positions = pos + cx * vec4 + cy * vec3
    normals = positions - pos
    positions.shape = -1, 3
    normals.shape = -1, 3
    normals *= 1 / np.linalg.norm(normals, axis=1).reshape(-1, 1)

    # Create texcords
    # ty, tx = np.meshgrid(u / u[-1], v / v[-1])
    ty, tx = np.meshgrid(v / v[-1], u / u[-1])
    texcoords = np.column_stack((tx.flat, ty.flat))

    # Create indices
    # Two triangles onto the "top-left" rectangle (six vertices)
    if stitch:
        base_triangle = [radial_verts - 1, 0, radial_verts, radial_verts, 0, 1]
    else:
        base_triangle = [radial_verts, 0, radial_verts + 1, radial_verts + 1, 0, 1]
    base_triangle = np.array(base_triangle, np.uint32)

    # Replicate to all rectangles, add offsets
    indices = np.tile(base_triangle, (tubular_segments, radial_segments, 1))
    gx, gy = np.meshgrid(
        np.arange(indices.shape[1], dtype=np.uint32),
        radial_verts * np.arange(indices.shape[0], dtype=np.uint32),
    )
    indices += (gx + gy).reshape(indices.shape[:2] + (1,))

    # Correct the faces at the tube's ends
    if stitch:
        indices[indices >= len(positions)] -= len(positions)

    indices = indices.reshape((-1, 3))
    # indices = np.fliplr(indices)  # Use this to change winding between CW and CCW

    return Geometry(
        indices=indices, positions=positions, normals=normals, texcoords=texcoords
    )


def torus_knot_surface(u, p, q, radius):
    """Taken from ThreeJS, but vectorized."""
    cu = np.cos(u)
    su = np.sin(u)
    qu_over_p = q / p * u
    cs = np.cos(qu_over_p)
    x = (2 + cs) * cu * (0.5 * radius)
    y = (2 + cs) * su * (0.5 * radius)
    z = np.sin(qu_over_p) * (0.5 * radius)
    return np.column_stack((x, y, z))
