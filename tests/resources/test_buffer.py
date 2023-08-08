import numpy as np
import pygfx as gfx
import pytest


def test_empty_data():
    b = gfx.Buffer(np.zeros((0, 3), np.float32))
    assert b.draw_range == (0, 0)
    assert b.itemsize == 3 * 4

    b = gfx.Buffer(np.zeros((0,), np.float32))
    assert b.draw_range == (0, 0)
    assert b.itemsize == 4

    b = gfx.Buffer(np.zeros((0, 4, 5), np.float32))
    assert b.draw_range == (0, 0)
    assert b.itemsize == 20 * 4


def test_nonempty_data():
    b = gfx.Buffer(np.zeros((2, 3), np.float32))
    assert b.draw_range == (0, 2)
    assert b.itemsize == 3 * 4

    b = gfx.Buffer(np.zeros((2,), np.float32))
    assert b.draw_range == (0, 2)
    assert b.itemsize == 4

    b = gfx.Buffer(np.zeros((2, 4, 5), np.float32))
    assert b.draw_range == (0, 2)
    assert b.itemsize == 20 * 4


def test_different_data_types():
    # Numpy array, let's do float16 while we're at it
    a = np.zeros((10, 3), np.float16)
    b = gfx.Buffer(a)
    assert b.format == "3xf2"

    # Memoryview
    a = memoryview(np.zeros((10, 2), np.int16))
    b = gfx.Buffer(a)
    assert b.format == "2xi2"

    # Bytes
    a = b"0000000000000000"
    b = gfx.Buffer(a)
    assert b.format == "u1"

    # You can specify the format ...
    b = gfx.Buffer(a, format="2xf4")
    assert b.format == "2xf4"
    # ... but some of the props will be wrong, so probably a bad idea
    assert b.nitems == 16
    assert b.itemsize == 1

    # Lists not supported
    with pytest.raises(TypeError):
        gfx.Buffer([1, 2, 3, 4, 5])


def test_unsupported_dtypes():
    a = np.zeros((10, 3), np.float64)
    with pytest.raises(ValueError):
        gfx.Buffer(a)

    a = np.zeros((10, 3), np.int64)
    with pytest.raises(ValueError):
        gfx.Buffer(a)

    a = np.zeros((10, 3), np.uint64)
    with pytest.raises(ValueError):
        gfx.Buffer(a)


def test_special_data():
    # E.g. uniform buffers, storage buffers with structured dtype, multidimensional storage data.

    a = np.zeros((10, 3, 2), np.float16)
    b = gfx.Buffer(a)
    assert b.format is None

    a = np.zeros((), dtype=[("a", "<i4"), ("b", "<f4")])
    b = gfx.Buffer(a)
    assert b.format is None
