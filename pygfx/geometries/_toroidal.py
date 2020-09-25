import numpy as np

from ..resources import Buffer
from ._base import Geometry


class KleinBottleGeometry(Geometry):
    """The Klein bottle is a surface for which the inside and outside
    are the same. A bit like a Möbius strip. In fact, a Klein bottle
    can be constructed by glueing together two Möbius strips.

    More technically: the Klein bottle is an example of a non-orientable
    surface; it is a two-dimensional manifold against which a system
    for determining a normal vector cannot be consistently defined.
    """

    # This is an interesting object for mathematicians. For us it's
    # interesting because we can test whether our lighting etc. deals
    # correctly with objects for which the "inside" must also be shown.

    def __init__(self, scale=1.0):
        super().__init__()

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
        indices = np.array([0, 1, n + 1, n + 1, n, 0], np.int32)
        # Replicate to all rectangles, add offsets
        indices = np.tile(indices, (n, n - 1, 1))
        gx, gy = np.meshgrid(
            np.arange(indices.shape[1]), n * np.arange(indices.shape[0])
        )
        indices += (gx + gy).reshape(indices.shape[:2] + (1,))
        # Stitch the ends together over one axis. We can't stitch the other ends
        # together, since that's where the normals flip from "inside" to "outside".
        indices[-1, :, 2:5] -= n * n
        indices = indices.reshape(-1)

        # Create buffers for this geometry
        self.positions = Buffer(positions, usage="vertex|storage")
        self.index = Buffer(indices, usage="index|storage")
        self.texcoords = Buffer(texcoords, usage="vertex|storage")


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


class TorusKnotGeometry(Geometry):
    """Creates a torus knot, the particular shape of which is defined
    by a pair of coprime integers, p and q. If p and q are not coprime,
    the result will be a torus link.

    Arguments:
        scale (float): the scale of the torus, default 1.
        tube (float): the radius of the tube. Default 0.4.
        tubular_segments (int): default is 64.
        radial_segments (int): default is 8.
        p (int): how many times the geometry winds around its axis of
            rotational symmetry. Default 2.
        q (int): how many times the geometry winds around a circle in
            the interior of the torus. Default 3.

    """

    def __init__(
        self, scale=1.0, tube=0.4, tubular_segments=64, radial_segments=8, p=2, q=3
    ):
        super().__init__()

        # If we stitch, we have no duplicate vertices, but stitch the
        # ends together with the indices, resulting in a fully closed object.
        # However, texturing works better without such stitching.
        stitch = False

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
        v = np.linspace(
            0, 2 * np.pi, radial_verts, endpoint=not stitch, dtype=np.float32
        )

        # Get positions along the torus' center, and a tiny step further
        pos1 = torus_knot_surface(u, p, q, scale)
        pos2 = torus_knot_surface(u + 0.01, p, q, scale)
        # Two vectors along the torus' centerline
        vec1 = pos1 - pos2
        vec2 = pos1 + pos2
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

        # Create texcords
        # ty, tx = np.meshgrid(u / u[-1], v / v[-1])
        ty, tx = np.meshgrid(v / v[-1], u / u[-1])
        texcoords = np.column_stack((tx.flat, ty.flat))

        # Create indices
        # Two triangles onto the "top-left" rectangle (six vertices)
        indices = np.array(
            [0, radial_verts, radial_verts + 1, radial_verts + 1, 1, 0],
            np.int32,
        )
        # Replicate to all rectangles, add offsets
        indices = np.tile(indices, (tubular_segments, radial_segments, 1))
        gx, gy = np.meshgrid(
            np.arange(indices.shape[1]), radial_verts * np.arange(indices.shape[0])
        )
        indices += (gx + gy).reshape(indices.shape[:2] + (1,))
        # Stitch the ends together over both axii.
        if stitch:
            indices[-1, :, 1:4] -= radial_verts * tubular_verts
            indices[:, -1, 2:5] -= radial_verts
        indices = indices.reshape(-1)

        # Create buffers for this geometry
        self.positions = Buffer(positions, usage="vertex|storage")
        self.normals = Buffer(normals, usage="vertex|storage")
        self.index = Buffer(indices, usage="index|storage")
        self.texcoords = Buffer(texcoords, usage="vertex|storage")


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
