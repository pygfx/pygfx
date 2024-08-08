import pygfx as gfx
import numpy as np
import pylinalg as la


def test_bounding_box():
    pos = np.array([(0, 0, 0), (1, 1, 1), (3, 3, 3)], np.float32)
    geo = gfx.Geometry(positions=pos)
    assert geo.get_bounding_box().tolist() == [[0, 0, 0], [3, 3, 3]]

    pos = np.array([(0, 1, 3), (3, 0, 1), (1, 3, 0)], np.float32)
    geo = gfx.Geometry(positions=pos)
    assert geo.get_bounding_box().tolist() == [[0, 0, 0], [3, 3, 3]]

    pos = np.array([(1, 1, 1), (1, 1, 1), (1, 1, 1)], np.float32)
    geo = gfx.Geometry(positions=pos)
    assert geo.get_bounding_box().tolist() == [[1, 1, 1], [1, 1, 1]]

    pos = np.array([(0, 1, 2), (0, 1, 2), (0, 1, 2)], np.float32)
    geo = gfx.Geometry(positions=pos)
    assert geo.get_bounding_box().tolist() == [[0, 1, 2], [0, 1, 2]]

    pos = np.array([(0, 0, np.nan), (1, 1, 1), (2, 2, 2)], np.float32)
    geo = gfx.Geometry(positions=pos)
    assert geo.get_bounding_box().tolist() == [[1, 1, 1], [2, 2, 2]]

    pos = np.array([(0, np.inf, 0), (1, 1, 1), (2, 2, 2)], np.float32)
    geo = gfx.Geometry(positions=pos)
    assert geo.get_bounding_box().tolist() == [[1, 1, 1], [2, 2, 2]]

    pos = np.array([(-np.inf, 0, 0), (1, 1, 1), (2, 2, 2)], np.float32)
    geo = gfx.Geometry(positions=pos)
    assert geo.get_bounding_box().tolist() == [[1, 1, 1], [2, 2, 2]]

    # Empty buffer is not allowed
    # pos = np.zeros((0, 3), np.float32)
    # geo = gfx.Geometry(positions=pos)
    # assert geo.get_bounding_box() is None

    pos = np.array([(-np.inf, 0, 0), (1, np.nan, 1)], np.float32)
    geo = gfx.Geometry(positions=pos)
    assert geo.get_bounding_box() is None


def test_bounding_sphere():
    pos = np.array([(0, 0, -8), (1, 1, 1), (2, 2, 10)], np.float32)
    geo = gfx.Geometry(positions=pos)
    bsphere = geo.get_bounding_sphere()
    bsphere_via_aabb = la.aabb_to_sphere(geo.get_bounding_box())

    assert np.allclose(bsphere, [1, 1, 1, 12.7279])
    assert np.allclose(bsphere_via_aabb, [1, 1, 1, 9.1104])
