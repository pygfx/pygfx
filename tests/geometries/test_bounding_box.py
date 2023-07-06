import pygfx as gfx
import numpy as np


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

    pos = np.zeros((0, 3), np.float32)
    geo = gfx.Geometry(positions=pos)
    assert geo.get_bounding_box() is None

    pos = np.array([(-np.inf, 0, 0), (1, np.nan, 1)], np.float32)
    geo = gfx.Geometry(positions=pos)
    assert geo.get_bounding_box() is None
