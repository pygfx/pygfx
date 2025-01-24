import numpy as np
import pygfx as gfx
import pytest


def test_empty_data():
    with pytest.raises(ValueError):
        gfx.Buffer(np.zeros((0, 3), np.float32))

    with pytest.raises(ValueError):
        gfx.Buffer(np.zeros((0,), np.float32))

    with pytest.raises(ValueError):
        gfx.Buffer(np.zeros((0, 4, 5), np.float32))


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
    assert b.format == "6xf2"

    a = np.zeros((), dtype=[("a", "<i4"), ("b", "<f4")])
    b = gfx.Buffer(a)
    assert b.format is None


def test_custom_nbytes():
    # It's only really used as a check, no other implications
    a = np.zeros((10, 4), np.float32)
    buf = gfx.Buffer(a, nbytes=160)
    assert buf.nitems == 10
    assert buf.nbytes == 160

    # When the check fails
    with pytest.raises(ValueError):
        gfx.Buffer(a, nbytes=80)


def test_custom_nitems():
    # Can be just a check
    a = np.zeros((10, 4), np.float32)
    buf = gfx.Buffer(a, nitems=10)
    assert buf.view.shape == (10, 4)

    # The view's shape first dim is nitems, but otherwise follows data.shape
    a = np.zeros((10,), np.float32)
    buf = gfx.Buffer(a, nitems=10)
    assert buf.view.shape == (10,)

    # Reshaping, part 1
    a = np.zeros((40,), np.float32)
    buf = gfx.Buffer(a, nitems=10)
    assert buf.view.shape == (10, 4)

    # Reshaping, part 2
    a = np.zeros((10, 4), np.float32)
    buf = gfx.Buffer(a, nitems=40)
    assert buf.view.shape == (40, 1)

    # Reshaping, part 3
    a = np.zeros((10, 4), np.float32)
    buf = gfx.Buffer(a, nitems=20)
    assert buf.view.shape == (20, 2)

    # --

    # Sanity check
    assert buf.data is a
    assert a.shape == (10, 4)

    # Check set_data
    a = np.zeros((40,), np.float32)
    buf.set_data(a)
    assert buf.view.shape == (20, 2)
    assert buf.data is a
    assert a.shape == (40,)


def test_set_data():
    a = np.zeros((10, 3, 2), np.float16)
    b = np.ones((10, 3, 2), np.float16)
    c = np.ones((20, 3, 2), np.float16)[::2]  # not contiguous

    # Create buffer with a
    buf = gfx.Buffer(a, force_contiguous=True)
    assert buf.data is a
    assert len(buf._gfx_get_chunk_descriptions()) == 1

    # Set data to b
    buf.set_data(b)
    assert buf.data is b
    assert len(buf._gfx_get_chunk_descriptions()) == 1

    # Set back to a
    buf.set_data(a)
    assert buf.data is a
    assert len(buf._gfx_get_chunk_descriptions()) == 1

    # Identity is *not* checked.
    buf.set_data(a)
    assert buf.data is a
    assert len(buf._gfx_get_chunk_descriptions()) == 1

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
    assert b._chunk_mask.size == 1

    # Goes up to 256 bytes
    a = np.zeros((256, 1), np.uint8)
    b = gfx.Buffer(a)
    assert b._chunk_size == 256
    assert b._chunk_mask.size == 1

    # Beyond 256 bytes, chunking kicks in
    a = np.zeros((257, 1), np.uint8)
    b = gfx.Buffer(a)
    assert b._chunk_size == 144
    assert b._chunk_mask.size == 2

    # Happy scaling to 16 pieces
    a = np.zeros((16 * 256, 1), np.uint8)
    b = gfx.Buffer(a)
    assert b._chunk_size == 256
    assert b._chunk_mask.size == 16

    # Stays about 16 pieces, but snaps to 16 byte alignment
    a = np.zeros((16 * 256 + 1, 1), np.uint8)
    b = gfx.Buffer(a)
    assert b._chunk_size == 256
    assert b._chunk_mask.size == 17

    # --- 4 x float -> itemsize is 16 bytes

    # Small buffers have just one chunk
    a = np.zeros((10, 4), np.float32)
    b = gfx.Buffer(a)
    assert b._chunk_size == 10
    assert b._chunk_mask.size == 1

    # Goes up to 256 bytes
    a = np.zeros((16, 4), np.float32)
    b = gfx.Buffer(a)
    assert b._chunk_size == 16
    assert b._chunk_mask.size == 1

    # Beyond 256 bytes, chunking kicks in
    a = np.zeros((17, 4), np.float32)
    b = gfx.Buffer(a)
    assert b._chunk_size == 9
    assert b._chunk_mask.size == 2

    # Happy scaling to 16 pieces
    a = np.zeros((256, 4), np.float32)
    b = gfx.Buffer(a)
    assert b._chunk_size == 16
    assert b._chunk_mask.size == 16

    # Stays about 16 pieces, but changes itemsize
    a = np.zeros((257, 4), np.float32)
    b = gfx.Buffer(a)
    assert b._chunk_size == 16
    assert b._chunk_mask.size == 17


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
    a = np.zeros((16, 4), np.float32)
    b = gfx.Buffer(a, chunk_size=1)
    assert b._chunk_size == 1
    assert b._chunk_mask.size == 16

    a = np.zeros((26, 4), np.float32)
    b = gfx.Buffer(a, chunk_size=26)
    assert b._chunk_size == 26
    assert b._chunk_mask.size == 1

    # Custom chunk size caps to 1 byte
    a = np.zeros((200, 1), np.uint8)
    b = gfx.Buffer(a, chunk_size=-1)
    assert b._chunk_size == 1
    assert b._chunk_mask.size == 200

    # Custom chunk size caps to 1 element
    a = np.zeros((200, 4), np.float32)
    b = gfx.Buffer(a, chunk_size=-1)
    assert b._chunk_size == 1
    assert b._chunk_mask.size == 200


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
    chunk = buf._gfx_get_chunk_data(0, buf.nitems)
    assert chunk.flags.c_contiguous

    # Dito when the data is a memoryview (did not work in an earlier version)
    buf = gfx.Buffer(memoryview(positions1))
    chunk = buf._gfx_get_chunk_data(0, buf.nitems)
    assert chunk.flags.c_contiguous

    # But prevented when force_contiguous is set
    with pytest.raises(ValueError) as err:
        buf = gfx.Buffer(positions1, force_contiguous=True)
    assert err.match("not c_contiguous")

    # This works, and avoids the aforementioned copy
    buf = gfx.Buffer(positions2)
    chunk = buf._gfx_get_chunk_data(0, buf.nitems)
    assert chunk.flags.c_contiguous
    assert chunk is buf.view


def test_endianness():
    data1 = np.random.uniform(size=(10,)).astype("<f4")
    data2 = data1.astype(">f4")

    buf1 = gfx.Buffer(data1)
    buf2 = gfx.Buffer(data2)

    chunk1 = buf1._gfx_get_chunk_data(0, 10)
    chunk2 = buf2._gfx_get_chunk_data(0, 10)

    assert data2.dtype.byteorder == ">"
    assert chunk2.dtype.byteorder == "<"
    assert np.all(chunk1 == chunk2)

    # Little endian support is for convenience, but forbidden when performance matters
    gfx.Buffer(data1, force_contiguous=True)
    with pytest.raises(ValueError):
        gfx.Buffer(data2, force_contiguous=True)


# %% Upload validity tests


def upload_validity_checker(func):
    def wrapper():
        for contiguous, nchannels, dtype in [
            (True, 1, np.uint8),
            (True, 1, np.float32),
            (True, 3, np.float32),
            (True, 4, np.float32),
            (False, 1, np.uint8),
            (False, 3, np.float32),
        ]:
            if contiguous:
                data = np.zeros((1000, nchannels), dtype)
            else:
                data = np.zeros((1000, nchannels + 1), dtype)[:, :nchannels]

            synced_data = data.copy()

            buf = gfx.Buffer(data)
            buf._gfx_get_chunk_descriptions()  # flush

            # Appy changes
            func(buf)

            # Do what the pygfx internals would do to sync to the gpu
            for offset, size in buf._gfx_get_chunk_descriptions():
                chunk = buf._gfx_get_chunk_data(offset, size)
                synced_data[offset : offset + size] = chunk

            # Check
            assert np.all(buf.data == synced_data)

    wrapper.__name__ = func.__name__
    return wrapper


@upload_validity_checker
def test_upload_validity_full1(buf):
    new_data = buf.data.copy()
    new_data.fill(1)
    buf.set_data(new_data)  # efficient, provided you already have the new_data
    buf.update_full()


@upload_validity_checker
def test_upload_validity_full2(buf):
    buf.data[:] = 1
    buf.update_full()


@upload_validity_checker
def test_upload_validity_every_2(buf):
    step = 2
    buf.data[::step] = 1
    indices = np.asarray(range(0, buf.nitems, step))
    buf.update_indices(indices)


@upload_validity_checker
def test_upload_validity_every_3(buf):
    step = 3
    buf.data[::step] = 1
    indices = np.asarray(range(0, buf.nitems, step))
    buf.update_indices(indices)


@upload_validity_checker
def test_upload_validity_every_9(buf):
    step = 9
    buf.data[::step] = 1
    indices = np.asarray(range(0, buf.nitems, step))
    buf.update_indices(indices)


@upload_validity_checker
def test_upload_validity_every_22(buf):
    step = 22
    buf.data[::step] = 1
    indices = np.asarray(range(0, buf.nitems, step))
    buf.update_indices(indices)


@upload_validity_checker
def test_upload_validity_every_77(buf):
    step = 77
    buf.data[::step] = 1
    indices = np.asarray(range(0, buf.nitems, step))
    buf.update_indices(indices)


@upload_validity_checker
def test_upload_validity_every_101(buf):
    step = 101
    buf.data[::step] = 1
    indices = np.asarray(range(0, buf.nitems, step))
    buf.update_indices(indices)


@upload_validity_checker
def test_upload_validity_single_1(buf):
    i = np.random.randint(0, buf.nitems)
    buf.data[i] = 1
    buf.update_indices([i])


@upload_validity_checker
def test_upload_validity_single_20(buf):
    indices = []
    for i in range(20):
        i = np.random.randint(0, buf.nitems)
        buf.data[i] = 1
        indices.append(i)
    buf.update_indices(indices)


@upload_validity_checker
def test_upload_validity_range_1(buf):
    i = np.random.randint(0, buf.nitems)
    n = np.random.randint(0, buf.nitems // 2)
    buf.data[i : i + n] = 1
    buf.update_range(i, n)


@upload_validity_checker
def test_upload_validity_range_10(buf):
    for i in range(10):
        i = np.random.randint(0, buf.nitems)
        n = np.random.randint(0, buf.nitems // 8)
        buf.data[i : i + n] = 1
        buf.update_range(i, n)


if __name__ == "__main__":
    test_empty_data()
    test_special_data()
    test_custom_nbytes()
    test_custom_nitems()
    test_set_data()
    test_chunk_size_small()
    test_chunk_size_large()
    test_custom_chunk_size()
    test_contiguous()
    test_endianness()

    test_upload_validity_full1()
    test_upload_validity_full2()
    test_upload_validity_every_2()
    test_upload_validity_every_3()
    test_upload_validity_every_9()
    test_upload_validity_every_22()
    test_upload_validity_every_77()
    test_upload_validity_every_101()
    test_upload_validity_single_1()
    test_upload_validity_single_20()
    test_upload_validity_range_1()
    test_upload_validity_range_10()
