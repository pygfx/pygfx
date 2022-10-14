import pygfx as gfx
import numpy as np

from pytest import raises


def test_contiguous():

    # Contiguous array is fine
    a = np.arange(100, dtype=np.float32)
    gfx.Buffer(a)

    # Non-contiguous not
    b = a[::2]
    with raises(ValueError):
        gfx.Buffer(b)


if __name__ == "__main__":
    test_contiguous()
