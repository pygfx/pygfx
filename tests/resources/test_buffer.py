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


def test_set_data():

    a = np.zeros((10, 3, 2), np.float16)
    b = np.ones((10, 3, 2), np.float16)
    c = np.ones((20, 3, 2), np.float16)[::2]  # not contiguous

    # Create buffer with a
    buf = gfx.Buffer(a, force_contiguous=True)
    assert buf.data is a
    assert buf._gfx_get_chunk_descriptions()

    # Set data to b
    buf.set_data(b)
    assert buf.data is b
    assert buf._gfx_get_chunk_descriptions()

    # Set back to a
    buf.set_data(a)
    assert buf.data is a
    assert buf._gfx_get_chunk_descriptions()

    # Identity is *not* checked.
    buf.set_data(a)
    assert buf.data is a
    assert buf._gfx_get_chunk_descriptions()

    # Check behavior with force_contiguous set
    with pytest.raises(ValueError) as err:
        buf.set_data(c)
    assert err.match("not c_contiguous")


def test_chunk_size_small():

    # 1 x uint8

    # Small buffers have just one chunk
    a = np.zeros((10, 1), np.uint8)
    b = gfx.Buffer(a)
    assert b._chunk_size == 10
    assert b._chunk_map.size == 1

    # Goes up to 256 bytes
    a = np.zeros((256, 1), np.uint8)
    b = gfx.Buffer(a)
    assert b._chunk_size == 256
    assert b._chunk_map.size == 1

    # Beyond 256 bytes, chunking kicks in
    a = np.zeros((257, 1), np.uint8)
    b = gfx.Buffer(a)
    assert b._chunk_size == 256
    assert b._chunk_map.size == 2

    # Happy scaling to 16 pieces
    a = np.zeros((16 * 256, 1), np.uint8)
    b = gfx.Buffer(a)
    assert b._chunk_size == 256
    assert b._chunk_map.size == 16

    # Stays about 16 pieces, but changes itemsize
    a = np.zeros((16 * 256 + 1, 1), np.uint8)
    b = gfx.Buffer(a)
    assert b._chunk_size == 257
    assert b._chunk_map.size == 16

    # --- 4 x float -> itemsize is 16 bytes

    # Small buffers have just one chunk
    a = np.zeros((10, 4), np.float32)
    b = gfx.Buffer(a)
    assert b._chunk_size == 10
    assert b._chunk_map.size == 1

    # Goes up to 256 bytes
    a = np.zeros((16, 4), np.float32)
    b = gfx.Buffer(a)
    assert b._chunk_size == 16
    assert b._chunk_map.size == 1

    # Beyond 256 bytes, chunking kicks in
    a = np.zeros((17, 4), np.float32)
    b = gfx.Buffer(a)
    assert b._chunk_size == 16
    assert b._chunk_map.size == 2

    # Happy scaling to 16 pieces
    a = np.zeros((256, 4), np.float32)
    b = gfx.Buffer(a)
    assert b._chunk_size == 16
    assert b._chunk_map.size == 16

    # Stays about 16 pieces, but changes itemsize
    a = np.zeros((257, 4), np.float32)
    b = gfx.Buffer(a)
    assert b._chunk_size == 17
    assert b._chunk_map.size == 16


def test_chunk_size_large():

    # Caps to 1MB

    a = np.zeros((50 * 2**20, 1), np.uint8)
    b = gfx.Buffer(a)
    assert b._chunk_size == 2**20

    a = np.zeros((200 * 2**20, 1), np.uint8)
    b = gfx.Buffer(a)
    assert b._chunk_size == 2**20

    a = np.zeros((50 * 65536, 4), np.float32)
    b = gfx.Buffer(a)
    assert b._chunk_size == 2**16  # 2**16 * 16 == 2**20

    a = np.zeros((200 * 65536, 4), np.float32)
    b = gfx.Buffer(a)
    assert b._chunk_size == 2**16


def test_custom_chunk_size():
    # Custom chunk size can get it lower (one row is 16 bytes)
    a = np.zeros((16, 4), np.float32)
    b = gfx.Buffer(a, chunksize=16)
    assert b._chunk_size == 1
    assert b._chunk_map.size == 16

    # Custom chunk size rounds to item size
    a = np.zeros((16, 4), np.float32)
    b = gfx.Buffer(a, chunksize=10)
    assert b._chunk_size == 1
    assert b._chunk_map.size == 16

    # Custom chunk size rounds to item size
    a = np.zeros((26, 4), np.float32)
    b = gfx.Buffer(a, chunksize=1000)
    assert b._chunk_size == 26
    assert b._chunk_map.size == 1

    # Custom chunk size caps to 1 byte
    a = np.zeros((200, 1), np.uint8)
    b = gfx.Buffer(a, chunksize=-1)
    assert b._chunk_size == 1
    assert b._chunk_map.size == 200

    # Custom chunk size caps to 1 element
    a = np.zeros((200, 4), np.float32)
    b = gfx.Buffer(a, chunksize=-1)
    assert b._chunk_size == 1
    assert b._chunk_map.size == 200


def test_contiguous():
    xs = np.linspace(0, 20 * np.pi, 500)
    ys = np.sin(xs) * 10
    zs = np.zeros(xs.size)

    positions1 = np.vstack([xs, ys, zs]).astype(np.float32).T
    positions2 = np.ascontiguousarray(positions1)
    assert not positions1.flags.c_contiguous
    assert positions2.flags.c_contiguous

    # This works, because at upload time the data is copied if necessary
    buf = gfx.Buffer(positions1)
    mem = buf._gfx_get_chunk_data(0, buf.nitems)
    assert mem.c_contiguous

    # Dito when the data is a memoryview (did not work in an earlier version)
    buf = gfx.Buffer(memoryview(positions1))
    mem = buf._gfx_get_chunk_data(0, buf.nitems)
    assert mem.c_contiguous

    # But prevented when force_contiguous is set
    with pytest.raises(ValueError) as err:
        buf = gfx.Buffer(positions1, force_contiguous=True)
    assert err.match("not c_contiguous")

    # This works, and avoids the aforementioned copy
    buf = gfx.Buffer(positions2)
    mem = buf._gfx_get_chunk_data(0, buf.nitems)
    assert mem.c_contiguous
    assert mem is buf.mem


if __name__ == "__main__":
    test_set_data()
    test_chunk_size_small()
    test_chunk_size_large()
    test_custom_chunk_size()
    test_contiguous()
