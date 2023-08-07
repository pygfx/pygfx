import numpy as np
import pygfx as gfx
import pytest


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
