import numpy as np
import pygfx as gfx


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
