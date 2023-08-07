import numpy as np
import pygfx as gfx
import pytest


def test_different_data_types():
    # Numpy array, let's do float16 while we're at it
    a = np.zeros((10, 10), np.float16)
    b = gfx.Texture(a, dim=2)
    assert b.format == "f2"

    # Memoryview
    a = memoryview(np.zeros((10, 10), np.int16))
    b = gfx.Texture(a, dim=2)
    assert b.format == "i2"

    # Lists not supported
    with pytest.raises(TypeError):
        gfx.Texture([1, 2, 3, 4, 5], dim=1)


def test_unsupported_dtypes():
    a = np.zeros((10, 10), np.float64)
    with pytest.raises(ValueError):
        gfx.Texture(a, dim=2)

    a = np.zeros((10, 10), np.int64)
    with pytest.raises(ValueError):
        gfx.Texture(a, dim=2)

    a = np.zeros((10, 10), np.uint64)
    with pytest.raises(ValueError):
        gfx.Texture(a, dim=2)


def test_supported_shapes():
    # 1D
    for i in range(4):
        if i == 0:
            a = np.zeros((10,), np.float32)
        else:
            a = np.zeros((10, i), np.float32)
        t = gfx.Texture(a, dim=1)
        assert t.size == (10, 1, 1)

    # 2D
    for i in range(4):
        if i == 0:
            a = np.zeros((10, 10), np.float32)
        else:
            a = np.zeros((10, 10, i), np.float32)
        t = gfx.Texture(a, dim=2)
        assert t.size == (10, 10, 1)

    # 3D
    for i in range(4):
        if i == 0:
            a = np.zeros((10, 10, 10), np.float32)
        else:
            a = np.zeros((10, 10, 10, i), np.float32)
        t = gfx.Texture(a, dim=3)
        assert t.size == (10, 10, 10)

    # Stack of 1D images
    a = np.zeros((10, 10), np.float32)
    t = gfx.Texture(a, dim=1, size=(10, 10, 1))

    # Stack of 2D images
    a = np.zeros((10, 10, 10), np.float32)
    t = gfx.Texture(a, dim=2, size=(10, 10, 10))


def test_unsupported_shapes():
    # Always fail on empty array
    a = np.zeros((), np.float32)
    for dim in (1, 2, 3):
        with pytest.raises(ValueError):
            gfx.Texture(a, dim=dim)

    # Always fail on 4D array
    a = np.zeros((5, 5, 5, 5), np.float32)
    for dim in (1, 2, 3):
        with pytest.raises(ValueError):
            gfx.Texture(a, dim=dim)

    # 1D
    for i in [0, 5, 6]:
        a = np.zeros((10, i), np.float32)
        with pytest.raises(ValueError):
            gfx.Texture(a, dim=1)

    # 2D
    a = np.zeros((10,), np.float32)
    with pytest.raises(ValueError):
        gfx.Texture(a, dim=3)
    for i in [0, 5, 6]:
        a = np.zeros((10, 10, i), np.float32)
        with pytest.raises(ValueError):
            gfx.Texture(a, dim=2)

    # 3D
    a = np.zeros((10,), np.float32)
    with pytest.raises(ValueError):
        gfx.Texture(a, dim=3)
    a = np.zeros((10, 10), np.float32)
    with pytest.raises(ValueError):
        gfx.Texture(a, dim=3)
    for i in [0, 5, 6]:
        a = np.zeros((10, 10, 10, i), np.float32)
        with pytest.raises(ValueError):
            gfx.Texture(a, dim=3)
