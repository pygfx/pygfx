import numpy as np

from ._base import Geometry


def tetrahedron_geometry(radius=1.0, subdivisions=0):
    """Generate a tetrahedron.

    Creates a tetrahedron that is centered around local origin. Its vertices
    lie on the surface of a sphere of the given radius. Its faces are optionally
    subdivided if subdivisions>0.

    Parameters
    ----------
    radius : int
        The radius of a sphere that has the vertices on its surface.
    subdivisions: int
        The amount of times each face will be subdivided, where 0
        (the default) means no subdivision.

    Returns
    -------
    tetrahedron : Geometry
        A geometry object representing the desired tetrahedron.
        Mathematically, it consists of a set of open orientable manifolds.

    """
    positions = np.array(
        [
            [1, 1, 1],
            [-1, -1, 1],
            [-1, 1, -1],
            [1, -1, -1],
        ],
        dtype=np.float32,
    )

    indices = np.array(
        [
            [2, 0, 1],
            [0, 2, 3],
            [1, 0, 3],
            [2, 1, 3],
        ],
        dtype=np.int32,
    )

    return polyhedron_geometry(positions, indices, radius, subdivisions)


def octahedron_geometry(radius=1.0, subdivisions=0):
    """Generate a octahedron.

    Creates an octahedron that is centered around the local origin. It has its
    vertices lie on the surface of a sphere of the given radius. Its faces are
    optionally subdivided if subdivisions>0.

    Parameters
    ----------
    radius : int
        The radius of a sphere that has the vertices on its surface.
    subdivisions: int
        The amount of times each face will be subdivided, where 0
        (the default) means no subdivision.

    Returns
    -------
    octahedron : Geometry
        A geometry object representing the desired octahedron.
        Mathematically, it consists of a set of open orientable manifolds.

    """
    positions = np.array(
        [
            [1, 0, 0],
            [-1, 0, 0],
            [0, 1, 0],
            [0, -1, 0],
            [0, 0, 1],
            [0, 0, -1],
        ],
        dtype=np.float32,
    )

    indices = np.array(
        [
            [0, 4, 2],
            [0, 3, 4],
            [0, 5, 3],
            [0, 2, 5],
            [1, 5, 2],
            [1, 3, 5],
            [1, 4, 3],
            [1, 2, 4],
        ],
        dtype=np.int32,
    )

    return polyhedron_geometry(positions, indices, radius, subdivisions)


def icosahedron_geometry(radius=1.0, subdivisions=0):
    """Generate a icosahedron.

    Creates an icosahedron that is centered around the local origin. It has its
    vertices lie on the surface of a sphere of the given radius. Its faces are
    optionally subdivided if subdivisions>0.

    Parameters
    ----------
    radius : int
        The radius of a sphere that has the vertices on its surface.
    subdivisions: int
        The amount of times each face will be subdivided, where 0
        (the default) means no subdivision.

    Returns
    -------
    icosahedron : Geometry
        A geometry object representing the desired icosahedron.
        Mathematically, it consists of a set of open orientable manifolds.

    """
    t = (1 + np.sqrt(5)) / 2

    positions = np.array(
        [
            [-1, t, 0],
            [1, t, 0],
            [-1, -t, 0],
            [1, -t, 0],
            [0, -1, t],
            [0, 1, t],
            [0, -1, -t],
            [0, 1, -t],
            [t, 0, -1],
            [t, 0, 1],
            [-t, 0, -1],
            [-t, 0, 1],
        ],
        dtype=np.float32,
    )

    indices = np.array(
        [
            [0, 5, 11],
            [0, 1, 5],
            [0, 7, 1],
            [0, 10, 7],
            [0, 11, 10],
            [1, 9, 5],
            [5, 4, 11],
            [11, 2, 10],
            [10, 6, 7],
            [7, 8, 1],
            [3, 4, 9],
            [3, 2, 4],
            [3, 6, 2],
            [3, 8, 6],
            [3, 9, 8],
            [4, 5, 9],
            [2, 11, 4],
            [6, 10, 2],
            [8, 7, 6],
            [9, 1, 8],
        ],
        dtype=np.int32,
    )

    return polyhedron_geometry(positions, indices, radius, subdivisions)


def dodecahedron_geometry(radius=1.0, subdivisions=0):
    """Generate a dodecahedron.

    Creates an dodecahedron that is centered around the local origin. It has its
    vertices lie on the surface of a sphere of the given radius. Its faces are
    optionally subdivided if subdivisions>0.

    Parameters
    ----------
    radius : int
        The radius of a sphere that has the vertices on its surface.
    subdivisions: int
        The amount of times each face will be subdivided, where 0
        (the default) means no subdivision.

    Returns
    -------
    dodecahedron : Geometry
        A geometry object representing the desired dodecahedron.
        Mathematically, it consists of a set of open orientable manifolds.

    """
    t = (1 + np.sqrt(5)) / 2
    r = 1 / t

    positions = np.array(
        [
            [-1, -1, -1],
            [-1, -1, 1],
            [-1, 1, -1],
            [-1, 1, 1],
            [1, -1, -1],
            [1, -1, 1],
            [1, 1, -1],
            [1, 1, 1],
            [0, -r, -t],
            [0, -r, t],
            [0, r, -t],
            [0, r, t],
            [-r, -t, 0],
            [-r, t, 0],
            [r, -t, 0],
            [r, t, 0],
            [-t, 0, -r],
            [t, 0, -r],
            [-t, 0, r],
            [t, 0, r],
        ],
        dtype=np.float32,
    )

    indices = np.array(
        [
            [3, 7, 11],
            [3, 15, 7],
            [3, 13, 15],
            [7, 17, 19],
            [7, 6, 17],
            [7, 15, 6],
            [17, 8, 4],
            [17, 10, 8],
            [17, 6, 10],
            [8, 16, 0],
            [8, 2, 16],
            [8, 10, 2],
            [0, 1, 12],
            [0, 18, 1],
            [0, 16, 18],
            [6, 2, 10],
            [6, 13, 2],
            [6, 15, 13],
            [2, 18, 16],
            [2, 3, 18],
            [2, 13, 3],
            [18, 9, 1],
            [18, 11, 9],
            [18, 3, 11],
            [4, 12, 14],
            [4, 0, 12],
            [4, 8, 0],
            [11, 5, 9],
            [11, 19, 5],
            [11, 7, 19],
            [19, 14, 5],
            [19, 4, 14],
            [19, 17, 4],
            [1, 14, 12],
            [1, 5, 14],
            [1, 9, 5],
        ],
        dtype=np.int32,
    )

    return polyhedron_geometry(positions, indices, radius, subdivisions)


def polyhedron_geometry(
    positions: np.ndarray, indices: np.ndarray, radius: float, subdivisions: int
):
    """Generate a Polyhedron.

    Procedurally generate geometry representing a polyhedron, centered around
    the origin. The vertices lie on the surface of a sphere of the given radius.
    The faces are optionally subdivided if subdivisions>0.

    Parameters
    ----------
    positions : ndarray
        The vertices used to initialize polyhedron generation.
    indices : ndarray
        The face index used to initialize polyhedron generation.
    radius : float
        The vertices lie on the surface of a sphere with this radius.
    subdivisions : int
        The amount of times the faces will be subdivided, where 0 (the
        default) means no subdivision.

    Returns
    -------
    polyhedron : Geometry
        A geometry object representing the requested polyhedron.
        Mathematically, it consists of a set of open orientable manifolds.
    """
    # subdivide faces
    faces = positions[indices]
    a = faces[..., 0, :]
    ab = faces[..., 1, :] - a  # face winding convention
    ac = faces[..., 2, :] - a

    # assume we're creating a grid for vectorization purposes
    # and throw out half at the end...
    n = subdivisions + 2
    # set up the grid to have origin in the top right
    # so we can use numpy's upper triangle indexing
    i = np.linspace(1.0, 0.0, n, dtype=np.float32)
    j = np.linspace(0.0, 1.0, n, dtype=np.float32)
    alpha, beta = np.meshgrid(i, j)
    # grid shape is (n_faces, 3, n, n)
    grid = a[..., None, None] + alpha * ab[..., None, None] + beta * ac[..., None, None]

    # zero out all the elements in the lower corner of the grid
    # in other words those positions which lie outside of the original face
    grid = np.triu(grid)

    # normalize the positions so they lie on the surface a sphere
    # grid shape is (n_faces, n, n, 3)
    grid = np.moveaxis(grid, 1, 3)
    # keeping nan's so we can filter by that later
    with np.errstate(invalid="ignore"):
        grid /= np.linalg.norm(grid, axis=-1)[..., None]

    # create an index array for a single face's plane of quads
    indices = np.arange(n**2, dtype=np.uint32).reshape((n, n))
    index = np.empty((n - 1, n - 1, 2, 3), dtype=np.uint32)
    index[:, :, 0, 0] = indices[np.arange(n - 1)[:, None], np.arange(n - 1)[None, :]]
    # the remainder of the indices for every panel are relative
    # note that they are constructed such that the diagonal aligns
    # with that of np.triu; otherwise we get nan vertices in all
    # faces that include the diagonal
    index[:, :, 0, 1] = index[:, :, 0, 0] + 1
    index[:, :, 0, 2] = index[:, :, 0, 0] + n + 1
    index[:, :, 1, 0] = index[:, :, 0, 0] + n + 1
    index[:, :, 1, 1] = index[:, :, 1, 0] - 1
    index[:, :, 1, 2] = index[:, :, 1, 0] - n - 1

    # now index the grid
    grid = grid.reshape(grid.shape[0], -1, grid.shape[-1])
    # we get (n_faces, n-1, n-1, 2, 3 vertices, xyz)
    faces = grid[:, index]
    # and we reshape to get our new list of triangles
    faces = faces.reshape(-1, 3, 3)
    # filter out any triangles that contain nan elements
    nan_index = np.any(np.isnan(faces), axis=(-1, -2))
    faces = faces[~nan_index]
    positions = faces.reshape(-1, 3)

    # now that we have our subdivided faces with duplicated vertices
    # we can compute normals
    a = faces[..., 0, :]
    ab = faces[..., 1, :] - a  # face winding convention
    ac = faces[..., 2, :] - a

    ab = np.ndarray.astype(ab, np.float32, copy=False)
    ac = np.ndarray.astype(ac, np.float32, copy=False)
    face_normals = np.cross(ab, ac)

    face_normals /= np.linalg.norm(face_normals, axis=-1)[..., None]
    normals = np.broadcast_to(face_normals[..., None, :], faces.shape)

    # and our texcoords via spherical coordinates
    # NOTE: if the initial faces straddle the boundary between 2pi back to 0
    # you will get a weird seam. that can be handled but we don't do so for now
    x, y, z = np.moveaxis(faces, -1, 0)
    uu = np.arctan2(y, x) / (np.pi * 2)
    vv = np.arccos(z) / np.pi
    texcoords = np.stack([uu, vv], axis=-1)

    # after all this, scale the points to the radius
    # we don't do this earlier so we can rely on the assumption that vectors
    # are normalized there
    faces *= radius

    # technically if meshmaterial didn't require an index buffer we could leave this out
    indices = np.arange(positions.shape[0], dtype=np.int32).reshape(-1, 3)

    return Geometry(
        indices=indices,
        positions=positions,
        normals=normals.reshape(-1, 3),
        texcoords=texcoords.reshape(-1, 2),
    )
