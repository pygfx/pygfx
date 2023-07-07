import numpy as np
import pygfx as gfx


def test_empty_data():
    b = gfx.Buffer(np.zeros((0, 3), np.float32))
    assert b.view == (0, 0)
    assert b.itemsize == 3 * 4
