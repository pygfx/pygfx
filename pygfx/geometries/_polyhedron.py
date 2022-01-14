import numpy as np

from ._base import Geometry


def octahedron_geometry(radius=1.0, subdivisions=0):
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
            [0, 2, 4],
            [0, 4, 3],
            [0, 3, 5],
            [0, 5, 2],
            [1, 2, 5],
            [1, 5, 3],
            [1, 3, 4],
            [1, 4, 2],
        ],
        dtype=np.int32,
    )

    return polyhedron_geometry(positions, indices, radius, subdivisions)


def icosahedron_geometry(radius=1.0, subdivisions=0):
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
            [0, 11, 5],
            [0, 5, 1],
            [0, 1, 7],
            [0, 7, 10],
            [0, 10, 11],
            [1, 5, 9],
            [5, 11, 4],
            [11, 10, 2],
            [10, 7, 6],
            [7, 1, 8],
            [3, 9, 4],
            [3, 4, 2],
            [3, 2, 6],
            [3, 6, 8],
            [3, 8, 9],
            [4, 9, 5],
            [2, 4, 11],
            [6, 2, 10],
            [8, 6, 7],
            [9, 8, 1],
        ],
        dtype=np.int32,
    )

    return polyhedron_geometry(positions, indices, radius, subdivisions)


def polyhedron_geometry(
    positions: np.ndarray, indices: np.ndarray, radius: float, subdivisions: int
):
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

    # grid shape is (n_faces, n, n, 3), moveaxis returns a view so create a copy for
    # optimal memory layout
    grid = np.moveaxis(grid, 1, 3)

    # normalize the positions so they lie on the surface a sphere
    # keeping nan's so we can filter by that later
    with np.errstate(invalid="ignore"):
        grid /= np.linalg.norm(grid, axis=-1)[..., None]

    # create an index array for a single face's plane of quads
    indices = np.arange(n ** 2, dtype=np.uint32).reshape((n, n))
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

    # now that we have our subdivided faces with duplicated vertices
    # we can compute normals
    a = faces[..., 0, :]
    ab = faces[..., 1, :] - a  # face winding convention
    ac = faces[..., 2, :] - a
    face_normals = np.cross(ab, ac)
    face_normals /= np.linalg.norm(face_normals, axis=-1)[..., None]
    normals = np.broadcast_to(face_normals[..., None, :], faces.shape)

    # and our texcoords
    x, y, z = np.moveaxis(faces, -1, 0)
    uu = np.arctan2(y, x) / (np.pi * 2)
    vv = np.arccos(z) / np.pi
    texcoords = np.stack([uu, vv], axis=-1)

    # after all this, scale the points to the radius
    faces *= radius

    return Geometry(
        positions=faces.reshape(-1, 3),
        normals=normals.reshape(-1, 3),
        texcoords=texcoords.reshape(-1, 2),
    )
