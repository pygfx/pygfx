import numpy as np
import pylinalg as la

from ._base import Geometry
from ._plane import generate_plane
from .utils import merge


def box_geometry(
    width=1,
    height=1,
    depth=1,
    width_segments=1,
    height_segments=1,
    depth_segments=1,
):
    """Generate a box (rectangular cuboid).

    Creates a box (a rectangular cuboid) of the given size that is centered
    around the local frame's origin. Faces may be subdivided by specifying the
    number of segments along each axis. This will result in all faces parallel to
    the given axis to be evenly divided into the requested number of segments.

    Parameters
    ----------
    width : int
        Size along the x-axis.
    height : int
        Size along the y-axis.
    depth : int
        Size along the z-axis.
    width_segments : int
        Number of segments along x-axis.
    height_segments : int
        Number of segments along y-axis.
    depth_segments : int
        Number of segments along z-axis.

    Returns
    -------
    box : Geometry
        A geometry object containing the requested box shape.
        Mathematically, it consists of a set of open orientable manifolds.

    """

    cube_dim = np.array([width, height, depth], dtype=np.float32)
    cube_seg = np.array(
        [width_segments, height_segments, depth_segments], dtype=np.uint32
    )
    cube_normal_up = np.array(
        [
            [[1, 0, 0], [0, 1, 0]],  # right
            [[-1, 0, 0], [0, 1, 0]],  # left
            [[0, 1, 0], [1, 0, 0]],  # top
            [[0, -1, 0], [-1, 0, 0]],  # bottom
            [[0, 0, 1], [0, 1, 0]],  # front (this matches the default plane)
            [[0, 0, -1], [0, 1, 0]],  # back
        ],
        dtype=np.float32,
    )

    plane_csys = np.array(
        [
            *cube_normal_up[4],
            np.cross(
                np.ndarray.astype(cube_normal_up[4, 0], np.float32, copy=False),
                np.ndarray.astype(cube_normal_up[4, 1], np.float32, copy=False),
            ),
        ]
    )

    planes = []
    for normal, up in cube_normal_up:
        normal = np.ndarray.astype(normal, np.float32, copy=False)
        up = np.ndarray.astype(up, np.float32, copy=False)

        plane_idx = np.flatnonzero(normal == 0)
        (
            plane_positions,
            plane_normals,
            plane_texcoords,
            plane_index,
        ) = generate_plane(*cube_dim[plane_idx], *cube_seg[plane_idx])

        affine = np.identity(4, dtype=np.float32)

        sign_idx = np.flatnonzero(normal != 0)[0]
        affine[:-1, -1] = (cube_dim[sign_idx] / 2) * normal

        swap_axes = np.dot(
            np.array([normal, up, np.cross(normal, up)]).T,
            plane_csys,
        )

        if normal[2] == 0:
            theta = np.pi / 2
            rotate90 = np.array(
                [
                    [np.cos(theta), -np.sin(theta), 0],
                    [np.sin(theta), np.cos(theta), 0],
                    [0, 0, 1],
                ]
            )
            affine[:-1, :-1] = np.dot(
                swap_axes,
                rotate90,
            )
        else:
            affine[:-1, :-1] = swap_axes

        plane_positions = la.vec_transform(plane_positions, affine)

        affine[:-1, -1] = 0
        plane_normals = la.vec_transform(plane_normals, affine)

        planes.append((plane_positions, plane_normals, plane_texcoords, plane_index))

    positions, normals, texcoords, indices = merge(planes)
    positions = positions.astype(np.float32)
    normals = normals.astype(np.float32)

    return Geometry(
        indices=indices, positions=positions, normals=normals, texcoords=texcoords
    )
