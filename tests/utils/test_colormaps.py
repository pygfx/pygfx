import numpy as np
import pygfx as gfx
from pygfx.utils import cm
import pytest


def test_create_colormap_basic():
    tm1 = cm.create_colormap([(0, 0, 0), (1, 0.2, 0.3)])
    tm2 = cm.create_colormap([(0, 0, 0, 1), (1, 0.2, 0.3, 1)])

    assert isinstance(tm1, gfx.TextureMap)
    assert isinstance(tm2, gfx.TextureMap)
    assert np.all(tm1.texture.data == tm2.texture.data)

    data = tm1.texture.data
    assert data.shape == (256, 4)

    assert tuple(data[0]) == (0.0, 0.0, 0.0, 1.0)
    # assert tuple(data[-1]) == (1, 0.2, 0.3, 1)  # 0.2f4 != 0.2f8 ?
    assert np.allclose(data[-1], (1, 0.2, 0.3, 1))

    # In the middle
    assert np.allclose(data[127], (0.5, 0.1, 0.15, 1), atol=0.002)
    assert np.allclose(data[128], (0.5, 0.1, 0.15, 1), atol=0.002)

    # But it does increase
    assert np.all(data[128, :3] > data[127, :3])

    # Fails

    with pytest.raises(ValueError):
        cm.create_colormap([(0, 0), (1, 1)])

    with pytest.raises(ValueError):
        cm.create_colormap([(0, 0, 0, 0, 0), (1, 1, 1, 1, 1)])


def test_create_colormap_size():
    # Create large colormap
    tm1 = cm.create_colormap([(0, 0, 0), (1, 0.2, 0.3)], 1024)
    assert isinstance(tm1, gfx.TextureMap)
    assert tm1.texture.data.shape == (1024, 4)

    # Use large data with n=256, reduces
    tm2 = cm.create_colormap(tm1.texture.data, 256)
    assert isinstance(tm2, gfx.TextureMap)
    assert tm2.texture.data.shape == (256, 4)

    # Use same size
    tm3 = cm.create_colormap(tm2.texture.data, 256)
    assert isinstance(tm3, gfx.TextureMap)
    assert tm3.texture.data.shape == (256, 4)

    # Check that copy is made
    assert tm3.texture.data is not tm2.texture.data


def test_create_colormap_dict():
    tm = cm.create_colormap(
        {
            "g": [(0, 0), (1, 1)],
            "b": [(0.5, 0), (1, 1)],
        },
        101,
    )

    assert isinstance(tm, gfx.TextureMap)
    data = tm.texture.data
    assert data.shape == (101, 4)

    assert tuple(data[0]) == (0, 0, 0, 1)
    assert tuple(data[25]) == (0, 0.25, 0, 1)
    assert tuple(data[50]) == (0, 0.5, 0, 1)
    assert tuple(data[75]) == (0, 0.75, 0.5, 1)
    assert tuple(data[100]) == (0, 1, 1, 1)


def test_colormap_reuse():
    # This fails if we don't implement the __getattr__ in cm.py correctly
    tm1 = cm.viridis
    tm2 = cm.viridis
    assert tm1 is tm2


if __name__ == "__main__":
    test_create_colormap_basic()
    test_create_colormap_size()
    test_create_colormap_dict()
    test_colormap_reuse()
