import numpy as np
import pygfx as gfx
import pytest


def test_different_data_types():
    # Numpy array, let's do float16 while we're at it
    a = np.zeros((10, 10), np.float16)
    tex = gfx.Texture(a, dim=2)
    assert tex.format == "f2"

    # Memoryview
    a = memoryview(np.zeros((10, 10), np.int16))
    tex = gfx.Texture(a, dim=2)
    assert tex.format == "i2"

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


def test_set_data():

    a = np.zeros((10, 3), np.float32)
    b = np.ones((10, 3), np.float32)
    c = np.ones((20, 3), np.float32)[::2]  # not contiguous

    # Create texture with a
    tex = gfx.Texture(a, dim=2, force_contiguous=True)
    assert tex.data is a
    assert len(tex._gfx_get_chunk_descriptions()) == 1

    # Set data to b
    tex.set_data(b)
    assert tex.data is b
    assert len(tex._gfx_get_chunk_descriptions()) == 1

    # Set back to a
    tex.set_data(a)
    assert tex.data is a
    assert len(tex._gfx_get_chunk_descriptions()) == 1

    # Identity is *not* checked.
    tex.set_data(a)
    assert tex.data is a
    assert len(tex._gfx_get_chunk_descriptions()) == 1

    # Check behavior with force_contiguous set
    with pytest.raises(ValueError) as err:
        tex.set_data(c)
    assert err.match("not c_contiguous")


def test_chunk_size_small():

    # 1 x uint8

    # Small textures have just one chunk
    a = np.zeros((4, 4, 4), np.uint8)
    tex = gfx.Texture(a, dim=3)
    assert tex._chunk_size == (4, 4, 4)
    assert tex._chunk_mask.size == 1

    # Goes up to 256 bytes
    a = np.zeros((16, 16, 1), np.uint8)
    tex = gfx.Texture(a, dim=2)
    assert tex._chunk_size == (16, 16, 1)
    assert tex._chunk_mask.size == 1

    # Beyond 256 bytes, chunking kicks in
    a = np.zeros((257, 1), np.uint8)
    tex = gfx.Texture(a, dim=1)
    assert tex._chunk_size == (144, 1, 1)
    assert tex._chunk_mask.size == 2
    #
    a = np.zeros((17, 16, 1), np.uint8)
    tex = gfx.Texture(a, dim=2)
    assert tex._chunk_size == (16, 9, 1)
    assert tex._chunk_mask.size == 2
    #
    a = np.zeros((7, 7, 7), np.uint8)
    tex = gfx.Texture(a, dim=3)
    assert tex._chunk_size == (7, 4, 7)
    assert tex._chunk_mask.size == 2

    # --- float32 -> itemsize is 4 bytes

    # Small textures have just one chunk
    a = np.zeros((4, 4, 4), np.float32)
    tex = gfx.Texture(a, dim=3)
    assert tex._chunk_size == (4, 4, 4)
    assert tex._chunk_mask.size == 1

    # Goes up to 256 bytes
    a = np.zeros((8, 8, 1), np.float32)
    tex = gfx.Texture(a, dim=2)
    assert tex._chunk_size == (8, 8, 1)
    assert tex._chunk_mask.size == 1

    # Beyond 256 bytes, chunking kicks in
    a = np.zeros((65, 1), np.float32)
    tex = gfx.Texture(a, dim=1)
    assert tex._chunk_size == (36, 1, 1)
    assert tex._chunk_mask.size == 2
    #
    a = np.zeros((9, 8, 1), np.float32)
    tex = gfx.Texture(a, dim=2)
    assert tex._chunk_size == (8, 5, 1)
    assert tex._chunk_mask.size == 2
    #
    a = np.zeros((5, 5, 5), np.float32)
    tex = gfx.Texture(a, dim=3)
    assert tex._chunk_size == (4, 3, 5)
    assert tex._chunk_mask.size == 4


def test_chunk_size_large():

    # Caps to 16MB

    a = np.zeros((10, 1024, 1024), np.uint8)
    tex = gfx.Texture(a, dim=3)
    assert tex._chunk_mask.size == 25

    a = np.zeros((100, 1024, 1024), np.uint8)
    tex = gfx.Texture(a, dim=3)
    assert tex._chunk_mask.size == 25

    a = np.zeros((1000, 1024, 1024), np.uint8)
    tex = gfx.Texture(a, dim=3)
    assert tex._chunk_mask.size == 64

    a = np.zeros((2, 1024, 1024), np.float32)
    tex = gfx.Texture(a, dim=3)
    assert tex._chunk_mask.size == 25

    a = np.zeros((25, 1024, 1024), np.float32)
    tex = gfx.Texture(a, dim=3)
    assert tex._chunk_mask.size == 25

    a = np.zeros((250, 1024, 1024), np.float32)
    tex = gfx.Texture(a, dim=3)
    assert tex._chunk_mask.size == 72


def test_custom_chunk_size():
    a = np.zeros((16, 4), np.float32)
    tex = gfx.Texture(a, dim=2, chunk_size=1)
    assert tex._chunk_size == (1, 1, 1)
    assert tex._chunk_mask.size == 16 * 4

    a = np.zeros((26, 4), np.float32)
    tex = gfx.Texture(a, dim=2, chunk_size=(2, 26, 1))
    assert tex._chunk_size == (2, 26, 1)
    assert tex._chunk_mask.size == 2

    # Custom chunk size caps to 1 byte / element
    a = np.zeros((200,), np.uint8)
    tex = gfx.Texture(a, dim=1, chunk_size=-1)
    assert tex._chunk_size == (1, 1, 1)
    assert tex._chunk_mask.size == 200


def test_contiguous():

    im1 = np.zeros((100, 100), np.float32)[10:-10, 10:-10]
    im2 = np.ascontiguousarray(im1)

    # This works, because at upload time the data is copied if necessary
    tex = gfx.Texture(im1, dim=2)
    mem = tex._gfx_get_chunk_data((0, 0, 0), im1.shape + (1,))
    assert mem.c_contiguous

    # Dito when the data is a memoryview (did not work in an earlier version).
    # For textures with non-contiguous data this takes a performance hit due to an extra data copy.
    tex = gfx.Texture(memoryview(im1), dim=2)
    mem = tex._gfx_get_chunk_data((0, 0, 0), im1.shape + (1,))
    assert mem.c_contiguous

    # This works, and avoids the aforementioned copy
    tex = gfx.Texture(im2, dim=2)
    mem = tex._gfx_get_chunk_data((0, 0, 0), im2.shape + (1,))
    assert mem.c_contiguous
    assert mem is tex.mem


if __name__ == "__main__":
    test_set_data()
    test_chunk_size_small()
    test_chunk_size_large()
    test_custom_chunk_size()
    test_contiguous()
