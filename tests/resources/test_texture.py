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


def test_simple_shapes():
    # 1D
    for i in range(4):
        if i == 0:
            a = np.zeros((10,), np.float32)
        else:
            a = np.zeros((10, i), np.float32)
        t = gfx.Texture(a, dim=1)
        assert t.size == (10, 1, 1)
        assert t.size == tuple(reversed(t.view.shape[:3]))

    # 2D
    for i in range(4):
        if i == 0:
            a = np.zeros((10, 10), np.float32)
        else:
            a = np.zeros((10, 10, i), np.float32)
        t = gfx.Texture(a, dim=2)
        assert t.size == (10, 10, 1)
        assert t.size == tuple(reversed(t.view.shape[:3]))

    # 3D
    for i in range(4):
        if i == 0:
            a = np.zeros((10, 10, 10), np.float32)
        else:
            a = np.zeros((10, 10, 10, i), np.float32)
        t = gfx.Texture(a, dim=3)
        assert t.size == (10, 10, 10)
        assert t.size == tuple(reversed(t.view.shape[:3]))


def test_ambiguous_shapes():
    # ----

    # A 2D scalar image, 1D rgba, or stack of 1D grayscale?
    a = np.zeros((3, 4), np.float32)

    # The default for dim==1 is the rgba kind
    t = gfx.Texture(a, dim=1)
    assert t.size == (3, 1, 1)
    assert t.format == "4xf4"
    # Explicit 1D rgba via size
    t = gfx.Texture(a, dim=1, size=(3, 1, 1))
    assert t.size == (3, 1, 1)
    assert t.format == "4xf4"
    # Explicit 1D stack via size
    t = gfx.Texture(a, dim=1, size=(4, 1, 3))
    assert t.size == (4, 1, 3)
    assert t.format == "f4"

    # For dim==2 there is not much ambiguity
    t = gfx.Texture(a, dim=2)
    assert t.size == (4, 3, 1)
    assert t.format == "f4"

    # Can even make it 3D
    t = gfx.Texture(a, dim=3, size=(4, 3, 1))
    assert t.size == (4, 3, 1)
    assert t.size == tuple(reversed(t.view.shape[:3]))
    assert t.format == "f4"
    # Can even make it 3D, but different
    t = gfx.Texture(a, dim=3, size=(4, 1, 3))
    assert t.size == (4, 1, 3)
    assert t.size == tuple(reversed(t.view.shape[:3]))
    assert t.format == "f4"

    # ----

    # An rgba image, a stack of grayscale images, or 3D?
    a = np.zeros((2, 3, 4), np.float32)

    # Default is rgba
    t = gfx.Texture(a, dim=2)
    assert t.size == (3, 2, 1)
    assert t.size == tuple(reversed(t.view.shape[:3]))
    assert t.format == "4xf4"
    # Make rgba explicit
    t = gfx.Texture(a, dim=2, size=(3, 2, 1))
    assert t.size == (3, 2, 1)
    assert t.size == tuple(reversed(t.view.shape[:3]))
    assert t.format == "4xf4"
    # Make it a stack
    t = gfx.Texture(a, dim=2, size=(4, 3, 2))
    assert t.size == (4, 3, 2)
    assert t.size == tuple(reversed(t.view.shape[:3]))
    assert t.format == "f4"

    # Can make it 3D
    t = gfx.Texture(a, dim=3)
    assert t.size == (4, 3, 2)
    assert t.size == tuple(reversed(t.view.shape[:3]))
    assert t.format == "f4"


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

    # Though can correct it via size!
    a = np.zeros((5, 5, 5, 5), np.float32)
    gfx.Texture(a, dim=1, size=(625, 1, 1))
    gfx.Texture(a, dim=2, size=(25, 25, 1))
    gfx.Texture(a, dim=3, size=(5, 5, 25))

    # But given size must match
    a = np.zeros((5, 5, 5, 5), np.float32)
    with pytest.raises(ValueError):
        gfx.Texture(a, dim=2, size=(5, 5, 1))
    a = np.zeros((5, 6), np.float32)
    with pytest.raises(ValueError):
        gfx.Texture(a, dim=2, size=(5, 5, 1))
    with pytest.raises(ValueError):
        gfx.Texture(a, dim=2, size=(6, 6, 1))
    # This is weird, but it does work
    gfx.Texture(a, dim=2, size=(5, 6, 1))

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
    a = np.zeros((12, 12, 1), np.uint8)
    tex = gfx.Texture(a, dim=2)
    assert tex._chunk_size == (12, 12, 1)
    assert tex._chunk_mask.size == 1

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
    assert tex._chunk_size == (7, 7, 4)
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
    assert tex._chunk_size == (5, 5, 2)
    assert tex._chunk_mask.size == 3


def test_chunk_size_large():
    # Caps to 16MB

    a = np.zeros((10, 1024, 1024), np.uint8)
    tex = gfx.Texture(a, dim=3)
    assert tex._chunk_mask.size == 320

    a = np.zeros((100, 1024, 1024), np.uint8)
    tex = gfx.Texture(a, dim=3)
    assert tex._chunk_mask.size == 800

    a = np.zeros((1000, 1024, 1024), np.uint8)
    tex = gfx.Texture(a, dim=3)
    assert tex._chunk_mask.size == 1024

    a = np.zeros((2, 1024, 1024), np.float32)
    tex = gfx.Texture(a, dim=3)
    assert tex._chunk_mask.size == 64

    a = np.zeros((25, 1024, 1024), np.float32)
    tex = gfx.Texture(a, dim=3)
    assert tex._chunk_mask.size == 800

    a = np.zeros((250, 1024, 1024), np.float32)
    tex = gfx.Texture(a, dim=3)
    assert tex._chunk_mask.size == 1024


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
    chunk = tex._gfx_get_chunk_data((0, 0, 0), (*im1.shape, 1))
    assert chunk.flags.c_contiguous

    # Dito when the data is a memoryview (did not work in an earlier version).
    # For textures with non-contiguous data this takes a performance hit due to an extra data copy.
    tex = gfx.Texture(memoryview(im1), dim=2)
    chunk = tex._gfx_get_chunk_data((0, 0, 0), (*im1.shape, 1))
    assert chunk.flags.c_contiguous

    # This works, and avoids the aforementioned copy
    tex = gfx.Texture(im2, dim=2)
    chunk = tex._gfx_get_chunk_data((0, 0, 0), (*im2.shape, 1))
    assert chunk.flags.c_contiguous
    assert chunk is tex.view


def test_endianness():
    data1 = np.random.uniform(size=(10, 10)).astype("<f4")
    data2 = data1.astype(">f4")

    tex1 = gfx.Texture(data1, dim=2)
    tex2 = gfx.Texture(data2, dim=2)

    chunk1 = tex1._gfx_get_chunk_data((0, 0, 0), (1, 10, 10))
    chunk2 = tex2._gfx_get_chunk_data((0, 0, 0), (1, 10, 10))

    assert data2.dtype.byteorder == ">"
    assert chunk2.dtype.byteorder == "<"
    assert np.all(chunk1 == chunk2)

    # Little endian support is for convenience, but forbidden when performance matters
    gfx.Texture(data1, dim=2, force_contiguous=True)
    with pytest.raises(ValueError):
        gfx.Texture(data2, dim=2, force_contiguous=True)


def test_rgb_support():
    # 1D
    im = np.zeros((100, 3), np.float32)
    tex = gfx.Texture(im, dim=1)
    chunk = tex._gfx_get_chunk_data((0, 0, 0), tex.size)
    assert im.shape == (100, 3)
    assert chunk.shape == (1, 1, 100, 4)

    # 2D
    im = np.zeros((100, 100, 3), np.float32)
    tex = gfx.Texture(im, dim=2)
    chunk = tex._gfx_get_chunk_data((0, 0, 0), tex.size)
    assert im.shape == (100, 100, 3)
    assert chunk.shape == (1, 100, 100, 4)

    # 3D
    im = np.zeros((100, 100, 100, 3), np.float32)
    tex = gfx.Texture(im, dim=3)
    chunk = tex._gfx_get_chunk_data((0, 0, 0), tex.size)
    assert im.shape == (100, 100, 100, 3)
    assert chunk.shape == (100, 100, 100, 4)

    # Not allowed when force_contiguous is set
    im = np.zeros((100, 100, 3), np.float32)
    with pytest.raises(ValueError):
        tex = gfx.Texture(im, dim=2, force_contiguous=True)


# %% Upload validity tests


def upload_validity_checker_2d(func):
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
                data = np.zeros((10, 10, nchannels), dtype)
            else:
                data = np.zeros((10, 10, nchannels + 1), dtype)[:, :, :nchannels]

            if nchannels != 3:
                synced_data = data.copy().reshape(1, 10, 10, nchannels)
            else:
                # Include rgb enumlation support
                synced_data = np.zeros((1, 10, 10, nchannels + 1), data.dtype)
                synced_data[:, :, :, :nchannels] = data

            tex = gfx.Texture(data, dim=2)
            tex._gfx_get_chunk_descriptions()  # flush

            # Appy changes
            func(tex)

            # Do what the pygfx internals would do to sync to the gpu
            chunks = tex._gfx_get_chunk_descriptions()
            for offset, size in chunks:
                chunk = tex._gfx_get_chunk_data(offset, size)
                synced_data[
                    offset[2] : offset[2] + size[2],
                    offset[1] : offset[1] + size[1],
                    offset[0] : offset[0] + size[0],
                ] = chunk

            # Check
            assert np.all(tex.data == synced_data[:, :, :, :nchannels])

    wrapper.__name__ = func.__name__
    return wrapper


def upload_validity_checker_3d(func):
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
                data = np.zeros((40, 30, 20, nchannels), dtype)
            else:
                data = np.zeros((40, 30, 20, nchannels + 1), dtype)[:, :, :, :nchannels]

            if nchannels != 3:
                synced_data = data.copy()
            else:
                # Include rgb enumlation support
                synced_data = np.zeros((40, 30, 20, nchannels + 1), data.dtype)
                synced_data[:, :, :, :nchannels] = data

            tex = gfx.Texture(data, dim=3)
            tex._gfx_get_chunk_descriptions()  # flush

            # Appy changes
            func(tex)

            # Do with the pygfx internals would do to sync to the gpu
            for offset, size in tex._gfx_get_chunk_descriptions():
                chunk = tex._gfx_get_chunk_data(offset, size)
                synced_data[
                    offset[2] : offset[2] + size[2],
                    offset[1] : offset[1] + size[1],
                    offset[0] : offset[0] + size[0],
                ] = chunk

            # Check
            assert np.all(tex.data == synced_data[:, :, :, :nchannels])

    wrapper.__name__ = func.__name__
    return wrapper


@upload_validity_checker_2d
def test_upload_validity_full1(tex):
    new_data = tex.data.copy()
    new_data.fill(1)
    tex.set_data(new_data)  # efficient, provided you already have the new_data
    tex.update_full()


@upload_validity_checker_2d
def test_upload_validity_full2(tex):
    tex.data[:] = 1
    tex.update_full()


@upload_validity_checker_2d
def test_upload_validity_every_2x(tex):
    step = 2
    tex.data[:, ::step] = 1
    indices_x = np.asarray(range(0, tex.size[0], step))
    tex.update_indices(indices_x, None, None)


@upload_validity_checker_2d
def test_upload_validity_every_9x(tex):
    step = 9
    tex.data[:, ::step] = 1
    indices_x = np.asarray(range(0, tex.size[0], step))
    tex.update_indices(indices_x, None, None)


@upload_validity_checker_2d
def test_upload_validity_every_77x(tex):
    step = 77
    tex.data[:, ::step] = 1
    indices_x = np.asarray(range(0, tex.size[0], step))
    tex.update_indices(indices_x, None, None)


@upload_validity_checker_2d
def test_upload_validity_every_335x(tex):
    step = 335
    tex.data[:, ::step] = 1
    indices_x = np.asarray(range(0, tex.size[0], step))
    tex.update_indices(indices_x, None, None)


@upload_validity_checker_2d
def test_upload_validity_every_2y(tex):
    step = 2
    tex.data[::step] = 1
    indices_y = np.asarray(range(0, tex.size[1], step))
    tex.update_indices(None, indices_y, None)


@upload_validity_checker_2d
def test_upload_validity_every_9y(tex):
    step = 9
    tex.data[::step] = 1
    indices_y = np.asarray(range(0, tex.size[1], step))
    tex.update_indices(None, indices_y, None)


@upload_validity_checker_2d
def test_upload_validity_every_77y(tex):
    step = 77
    tex.data[::step] = 1
    indices_y = np.asarray(range(0, tex.size[1], step))
    tex.update_indices(None, indices_y, None)


@upload_validity_checker_2d
def test_upload_validity_every_335y(tex):
    step = 335
    tex.data[::step] = 1
    indices_y = np.asarray(range(0, tex.size[1], step))
    tex.update_indices(None, indices_y, None)


@upload_validity_checker_2d
def test_upload_validity_single_1(tex):
    x = np.random.randint(0, tex.size[0])
    y = np.random.randint(0, tex.size[1])
    tex.data[y, x] = 1
    tex.update_indices(x, y, 0)


@upload_validity_checker_2d
def test_upload_validity_single_20(tex):
    for _i in range(20):
        x = np.random.randint(0, tex.size[0])
        y = np.random.randint(0, tex.size[1])
        tex.data[y, x] = 1
        tex.update_indices(x, y, 0)


@upload_validity_checker_2d
def test_upload_validity_range_1x(tex):
    i = np.random.randint(0, tex.size[0])
    n = np.random.randint(0, tex.size[0] // 2)
    tex.data[:, i : i + n] = 1
    tex.update_range((i, 0, 0), (n, tex.size[1], 1))


@upload_validity_checker_2d
def test_upload_validity_range_10x(tex):
    for i in range(10):
        i = np.random.randint(0, tex.size[0])
        n = np.random.randint(0, tex.size[0] // 8)
        tex.data[:, i : i + n] = 1
        tex.update_range((i, 0, 0), (n, tex.size[1], 1))


@upload_validity_checker_2d
def test_upload_validity_range_1y(tex):
    i = np.random.randint(0, tex.size[1])
    n = np.random.randint(0, tex.size[1] // 2)
    tex.data[i : i + n] = 1
    tex.update_range((0, i, 0), (tex.size[0], n, 1))


@upload_validity_checker_2d
def test_upload_validity_range_10y(tex):
    for i in range(10):
        i = np.random.randint(0, tex.size[1])
        n = np.random.randint(0, tex.size[1] // 8)
        tex.data[i : i + n] = 1
        tex.update_range((0, i, 0), (tex.size[0], n, 1))


##


@upload_validity_checker_3d
def test_3d_upload_validity_full1(tex):
    new_data = tex.data.copy()
    new_data.fill(1)
    tex.set_data(new_data)  # efficient, provided you already have the new_data
    tex.update_full()


@upload_validity_checker_3d
def test_3d_upload_validity_full2(tex):
    tex.data[:] = 1
    tex.update_full()


@upload_validity_checker_3d
def test_3d_upload_validity_every_2z(tex):
    step = 2
    tex.data[::step] = 1
    indices_z = np.asarray(range(0, tex.size[2], step))
    tex.update_indices(None, None, indices_z)


@upload_validity_checker_3d
def test_3d_upload_validity_every_9z(tex):
    step = 9
    tex.data[::step] = 1
    indices_z = np.asarray(range(0, tex.size[2], step))
    tex.update_indices(None, None, indices_z)


@upload_validity_checker_3d
def test_3d_upload_validity_every_77z(tex):
    step = 77
    tex.data[::step] = 1
    indices_z = np.asarray(range(0, tex.size[2], step))
    tex.update_indices(None, None, indices_z)


@upload_validity_checker_3d
def test_3d_upload_validity_single_1(tex):
    x = np.random.randint(0, tex.size[0])
    y = np.random.randint(0, tex.size[1])
    z = np.random.randint(0, tex.size[2])
    tex.data[z, y, x] = 1
    tex.update_indices(x, y, z)


@upload_validity_checker_3d
def test_3d_upload_validity_single_20(tex):
    for _i in range(20):
        x = np.random.randint(0, tex.size[0])
        y = np.random.randint(0, tex.size[1])
        z = np.random.randint(0, tex.size[2])
        tex.data[z, y, x] = 1
        tex.update_indices(x, y, z)


@upload_validity_checker_3d
def test_3d_upload_validity_range_1z(tex):
    i = np.random.randint(0, tex.size[2])
    n = np.random.randint(0, tex.size[2] // 2)
    tex.data[i : i + n] = 1
    tex.update_range((0, 0, i), (tex.size[0], tex.size[1], n))


@upload_validity_checker_3d
def test_3d_upload_validity_range_10z(tex):
    for i in range(10):
        i = np.random.randint(0, tex.size[2])
        n = np.random.randint(0, tex.size[2] // 8)
        tex.data[i : i + n] = 1
        tex.update_range((0, 0, i), (tex.size[0], tex.size[1], n))


if __name__ == "__main__":
    test_different_data_types()
    test_unsupported_dtypes()
    test_simple_shapes()
    test_ambiguous_shapes()
    test_unsupported_shapes()
    test_set_data()
    test_chunk_size_small()
    test_chunk_size_large()
    test_custom_chunk_size()
    test_contiguous()
    test_endianness()
    test_rgb_support()

    test_upload_validity_range_1x()
    test_3d_upload_validity_full1()
