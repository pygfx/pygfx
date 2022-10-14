import pygfx as gfx
import numpy as np

from pytest import raises


def test_contiguous():
    # Contiguous array is fine
    a = np.zeros((100, 100), np.float32)
    gfx.Texture(a, dim=2)

    # Non-contiguous not
    b = a[::2, ::2]
    with raises(ValueError):
        gfx.Texture(b, dim=2)


if __name__ == "__main__":
    test_contiguous()
