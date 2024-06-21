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


def test_chunk_size_small():

    # 1 x uint8

    # Small buffers have just one chunk
    a = np.zeros((10, 1), np.uint8)
    b = gfx.Buffer(a)
    assert b._chunk_itemsize == 10
    assert b._chunk_map.size == 1

    # Goes up to 256 bytes
    a = np.zeros((256, 1), np.uint8)
    b = gfx.Buffer(a)
    assert b._chunk_itemsize == 256
    assert b._chunk_map.size == 1

    # Beyond 256 bytes, chunking kicks in
    a = np.zeros((257, 1), np.uint8)
    b = gfx.Buffer(a)
    assert b._chunk_itemsize == 256
    assert b._chunk_map.size == 2

    # Happy scaling to 16 pieces
    a = np.zeros((16*256, 1), np.uint8)
    b = gfx.Buffer(a)
    assert b._chunk_itemsize == 256
    assert b._chunk_map.size == 16

    # Stays about 16 pieces, but changes itemsize
    a = np.zeros((16*256+1, 1), np.uint8)
    b = gfx.Buffer(a)
    assert b._chunk_itemsize == 257
    assert b._chunk_map.size == 16

    # --- 4 x float -> itemsize is 16 bytes

    # Small buffers have just one chunk
    a = np.zeros((10, 4), np.float32)
    b = gfx.Buffer(a)
    assert b._chunk_itemsize == 10
    assert b._chunk_map.size == 1

    # Goes up to 256 bytes
    a = np.zeros((16, 4), np.float32)
    b = gfx.Buffer(a)
    assert b._chunk_itemsize == 16
    assert b._chunk_map.size == 1

    # Beyond 256 bytes, chunking kicks in
    a = np.zeros((17, 4), np.float32)
    b = gfx.Buffer(a)
    assert b._chunk_itemsize == 16
    assert b._chunk_map.size == 2

    # Happy scaling to 16 pieces
    a = np.zeros((256, 4), np.float32)
    b = gfx.Buffer(a)
    assert b._chunk_itemsize == 16
    assert b._chunk_map.size == 16

    # Stays about 16 pieces, but changes itemsize
    a = np.zeros((257, 4), np.float32)
    b = gfx.Buffer(a)
    assert b._chunk_itemsize == 17
    assert b._chunk_map.size == 16


def test_custom_chunk_size():
    # Custom chunk size can get it lower (one row is 16 bytes)
    a = np.zeros((16, 4), np.float32)
    b = gfx.Buffer(a, chunksize=16)
    assert b._chunk_itemsize == 1
    assert b._chunk_map.size == 16

    # Custom chunk size rounds to item size
    a = np.zeros((16, 4), np.float32)
    b = gfx.Buffer(a, chunksize=10)
    assert b._chunk_itemsize == 1
    assert b._chunk_map.size == 16

    # Custom chunk size rounds to item size
    a = np.zeros((26, 4), np.float32)
    b = gfx.Buffer(a, chunksize=1000)
    assert b._chunk_itemsize == 26
    assert b._chunk_map.size == 1

    # Custom chunk size caps to 1 byte
    a = np.zeros((200, 1), np.uint8)
    b = gfx.Buffer(a, chunksize=-1)
    assert b._chunk_itemsize == 1
    assert b._chunk_map.size == 200

    # Custom chunk size caps to 1 element
    a = np.zeros((200, 4), np.float32)
    b = gfx.Buffer(a, chunksize=-1)
    assert b._chunk_itemsize == 1
    assert b._chunk_map.size == 200




def test_contiguous():
    xs = np.linspace(0, 20 * np.pi, 500)
    ys = np.sin(xs) * 10
    zs = np.zeros(xs.size)

    # This works, because at upload time the data is copied if necessary
    positions1 = np.vstack([xs, ys, zs]).astype(np.float32).T
    buf = gfx.Buffer(positions1)
    mem = buf._gfx_get_chunk_data(0, buf.nitems)
    assert mem.c_contiguous

    # This work, and avoids the aforementioned copy
    positions2 = np.ascontiguousarray(positions1)
    buf = gfx.Buffer(positions2)
    mem = buf._gfx_get_chunk_data(0, buf.nitems)
    assert mem.c_contiguous
    assert mem is buf.mem


if __name__ == "__main__":
    test_chunk_size_small()
    test_custom_chunk_size()
    test_contiguous()
