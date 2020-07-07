import numpy as np

from ..datawrappers import Buffer
from ._base import Geometry


class KleinBottleGeometry(Geometry):
    """ The Klein bottle is a surface for which the inside and outside
    are the same. A bit like a Möbius strip. In fact, a Klein bottle
    can be constructed by glueing together two Möbius strips.

    More technically: the Klein bottle is an example of a non-orientable
    surface; it is a two-dimensional manifold against which a system
    for determining a normal vector cannot be consistently defined.
    """

    # This is an interesting object for mathematicians. For us it's
    # interesting because we can test whether our lighting etc. deals
    # correctly with objects for which the "inside" must also be shown.

    def __init__(self, width, height, depth):
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
        x = (x + 1.66559) * (0.0277017 * width)
        y = (y - 2.04939) * (0.0277017 * height)
        z = (z + 0.00000) * (0.0277017 * depth)

        # Put into an Nx4 array
        positions = np.empty((x.size, 4), np.float32)
        positions[:, 0] = x.flat
        positions[:, 1] = y.flat
        positions[:, 2] = z.flat
        positions[:, 3] = 1

        # Texcoords are easy
        texcoords = np.concatenate([ux.flat, vx.flat]).astype(np.float32, copy=False)

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
