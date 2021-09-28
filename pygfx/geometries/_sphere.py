import numpy as np

from ..resources import Buffer
from ._base import Geometry
from ..linalg import Vector3


DTYPE = "f4"


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
    radii = np.linspace(radius_bottom, radius_top, num=n_rings, dtype=DTYPE)

    # height for each vertex ring from bottom to top
    half_height = height / 2
    heights = np.linspace(-half_height, half_height, num=n_rings, dtype=DTYPE)

    # to enable texture mapping to fully wrap around the cylinder,
    # we can't close the geometry and need a degenerate vertex
    n_vertices = radial_segments + 1

    # xy coordinates on unit circle for a single vertex ring
    theta = np.linspace(
        theta_start, theta_start + theta_length, num=n_vertices, dtype=DTYPE
    )
    ring_xy = np.column_stack([np.cos(theta), np.sin(theta)])

    # put all the rings together
    positions = np.empty((n_rings, n_vertices, 3), dtype=DTYPE)
    positions[..., :2] = ring_xy[None, ...] * radii[:, None, None]
    positions[..., 2] = heights[:, None]

    # the NORMALS are the same for every ring, so compute for only one ring
    # and then repeat
    slope = (radius_bottom - radius_top) / height
    ring_normals = np.empty(positions.shape[1:], dtype=DTYPE)
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
    texcoords = np.empty((n_rings, n_vertices, 2), dtype=DTYPE)
    texcoords[..., 0] = ring_u[None, :]
    texcoords[..., 1] = ring_v[:, None]

    # the face INDEX
    # the amount of vertices
    indices = np.arange(n_rings * n_vertices).reshape((n_rings, n_vertices))
    # for every panel (height_segments, radial_segments) there is a quad (2, 3)
    index = np.empty((height_segments, radial_segments, 2, 3), dtype="u4")
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
        theta_start, theta_start + theta_length, num=n_vertices, dtype=DTYPE
    )
    ring_xy = np.column_stack([np.cos(theta), np.sin(theta)])

    # put the vertices together, inserting a center vertex at the start
    positions = np.empty((1 + n_vertices, 3), dtype=DTYPE)
    positions[0, :2] = [0.0, 0.0]
    positions[1:, :2] = ring_xy * radius
    positions[..., 2] = height

    # the NORMALS
    normals = np.zeros_like(positions, dtype=DTYPE)
    sign = int(up) * 2.0 - 1.0
    normals[..., 2] = sign

    # the TEXTURE COORDS
    # uv etches out a circle from the [0..1, 0..1] range
    # direction is reversed for up=False
    texcoords = np.empty((1 + n_vertices, 2), dtype=DTYPE)
    texcoords[0] = [0.5, 0.5]
    texcoords[1:, 0] = ring_xy[:, 0] * 0.5 + 0.5
    texcoords[1:, 1] = ring_xy[:, 1] * 0.5 * sign + 0.5

    # the face INDEX
    indices = np.arange(n_vertices) + 1
    # for every radial segment there is a triangle (3)
    index = np.empty((radial_segments, 3), dtype="u4")
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


def merge(groups):
    positions = np.concatenate([g[0] for g in groups])
    normals = np.concatenate([g[1] for g in groups])
    texcoords = np.concatenate([g[2] for g in groups])
    index = np.concatenate([g[3] for g in groups])
    i = 0
    j = 0
    for g in groups:
        index[i:] += j
        # advance cursor to start of next group index
        i += len(g[3])
        j = len(g[0])
    return positions, normals, texcoords, index


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

        assert width_segments >= 3
        assert height_segments >= 2
        theta_end = np.min([theta_start + theta_length, np.pi])

        vertices = []
        normals = []
        uvs = []
        grid = []
        idx = 0
        for i in range(height_segments + 1):
            vertices_row = []
            v = i / height_segments
            u_offset = 0
            if i == 0 and theta_start == 0:
                u_offset = 0.5 / width_segments
            elif i == height_segments and theta_end == np.pi:
                u_offset = -0.5 / width_segments
            for j in range(width_segments + 1):
                u = j / width_segments
                x = -radius * np.cos(phi_start + u * phi_length) * np.sin(theta_start + v * theta_length)
                y = radius * np.cos(theta_start + v * theta_length)
                z = radius * np.sin(phi_start + u * phi_length) * np.sin(theta_start + v * theta_length)
                vertices.extend([x, y, z])
                normals.extend(Vector3(*vertices[-3:]).normalize().to_array())
                uvs.extend([u + u_offset, 1 - v])
                vertices_row.append(idx)
                idx += 1
            grid.append(vertices_row)
        
        indices = []
        for i in range(height_segments):
            for j in range(width_segments):
                a = grid[i][j + 1]
                b = grid[i][j]
                c = grid[i + 1][j]
                d = grid[i + 1][j + 1]
                if i != 0 or theta_start > 0:
                    indices.extend([a, b, d])
                if i != (height_segments - 1) or theta_end < np.pi:
                    indices.extend([b, c, d])

        self.positions = Buffer(np.array(vertices, dtype='f4'), usage="vertex|storage")
        self.normals = Buffer(np.array(normals, dtype='f4'), usage="vertex|storage")
        self.texcoords = Buffer(np.array(uvs, dtype='f4'), usage="vertex|storage")
        self.index = Buffer(np.array(indices, dtype='i4'), usage="index|storage")
