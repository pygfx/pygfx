import numpy as np

from pygfx.geometries import octahedron_geometry, icosahedron_geometry


def test_octahedron():
    g = octahedron_geometry()
    assert hasattr(g, "positions")
    assert hasattr(g, "normals")
    assert hasattr(g, "texcoords")
    assert g.positions.data.shape[-1] == 3
    assert g.positions.data.shape == g.normals.data.shape
    assert g.positions.data.shape[:-1] + (2,) == g.texcoords.data.shape

    # octahedron with vertices duplicated for each face
    assert g.positions.data.shape[0] == 8 * 3
    assert np.allclose(np.linalg.norm(g.positions.data, axis=-1), 1)


def test_octahedron_radius():
    radius = 3
    g = octahedron_geometry(radius=radius)

    assert np.allclose(np.linalg.norm(g.positions.data, axis=-1), radius)


def test_octahedron_subdivision():
    for subdivisions in range(5):
        g = octahedron_geometry(subdivisions=subdivisions)
        assert g.positions.data.shape[0] == (subdivisions + 1) ** 2 * 8 * 3


def test_icosahedron():
    g = icosahedron_geometry()
    assert hasattr(g, "positions")
    assert hasattr(g, "normals")
    assert hasattr(g, "texcoords")
    assert g.positions.data.shape[-1] == 3
    assert g.positions.data.shape == g.normals.data.shape
    assert g.positions.data.shape[:-1] + (2,) == g.texcoords.data.shape

    # icosahedron with vertices duplicated for each face
    assert g.positions.data.shape[0] == 20 * 3
    assert np.allclose(np.linalg.norm(g.positions.data, axis=-1), 1)
