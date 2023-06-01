import numpy as np

from pygfx.geometries import (
    dodecahedron_geometry,
    octahedron_geometry,
    icosahedron_geometry,
    tetrahedron_geometry,
)


def test_polyhedrons():
    for nr_faces, factory in [
        [4, tetrahedron_geometry],
        [8, octahedron_geometry],
        [20, icosahedron_geometry],
        [36, dodecahedron_geometry],
    ]:
        g = factory()
        assert hasattr(g, "indices")
        assert hasattr(g, "positions")
        assert hasattr(g, "normals")
        assert hasattr(g, "texcoords")
        assert g.positions.data.dtype == np.float32
        assert g.normals.data.dtype == np.float32
        assert g.texcoords.data.dtype == np.float32
        assert g.indices.data.dtype == np.int32
        assert g.positions.data.shape == (nr_faces * 3, 3)
        assert g.positions.data.shape == g.normals.data.shape
        assert g.positions.data.shape[:-1] + (2,) == g.texcoords.data.shape
        assert g.indices.data.size == g.positions.data.shape[0]
        assert g.indices.data.size == len(set(g.indices.data.flat))
        assert np.allclose(np.linalg.norm(g.positions.data, axis=-1), 1)


def test_octahedron_radius():
    for radius in np.arange(5) + 1:
        g = octahedron_geometry(radius=radius)

        assert np.allclose(np.linalg.norm(g.positions.data, axis=-1), radius)


def test_octahedron_subdivision():
    for subdivisions in range(5):
        g = octahedron_geometry(subdivisions=subdivisions)
        assert g.positions.data.shape[0] == (subdivisions + 1) ** 2 * 8 * 3
