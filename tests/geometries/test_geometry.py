import numpy as np
import pytest

import pygfx as gfx


def test_different_data_types():
    # Numpy array, let's do float16 while we're at it
    a = np.zeros((10, 3), np.float16)
    g = gfx.Geometry(foo=a)
    assert g.foo.format == "3xf2"
    assert g.foo.data is a

    # Memoryview
    a = memoryview(np.zeros((10, 2), np.int16))
    g = gfx.Geometry(foo=a)
    assert g.foo.format == "2xi2"
    assert g.foo.data is a

    # Bytes
    a = b"0000000000000000"
    g = gfx.Geometry(foo=a)
    assert g.foo.format == "u1"
    assert g.foo.data is a

    # Lists
    a = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    g = gfx.Geometry(indices=a)
    assert g.indices.format == "3xi4"  # <-- indices must be int
    assert isinstance(g.indices.data, np.ndarray)
    g = gfx.Geometry(positions=a)
    assert g.positions.format == "3xf4"  # <-- positions are usually f32
    assert isinstance(g.positions.data, np.ndarray)
    g = gfx.Geometry(foo=a)
    assert g.foo.format == "3xf4"  # <-- assumed that f32 is best
    assert isinstance(g.foo.data, np.ndarray)


def test_check_positions():
    # ok
    a = np.zeros((10, 3), np.float32)
    g = gfx.Geometry(positions=a)
    assert g.positions.data is a

    a = np.zeros((10, 4), np.float32)
    with pytest.raises(ValueError):
        gfx.Geometry(positions=a)

    a = np.zeros((10, 2), np.float32)
    with pytest.raises(ValueError):
        gfx.Geometry(positions=a)


def test_check_common_dict_attributes():
    g = gfx.Geometry(positions=[[0, 1, 0], [1, 0, 1]])
    assert "positions" in g.keys()
    assert len(g.keys()) == len(g.values())
