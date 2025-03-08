import os
import time

import numpy as np
from pygfx.utils.text._shaper import TemporalCache, shape_text_hb, shape_text_ft
from pygfx.utils.text import font_manager


def test_cache():
    c = TemporalCache(0.1, getter=hash)

    # === Store one value
    assert c["foo"] == hash("foo")
    assert len(c) == 1

    # Wait for the entry to become old.
    # The item should remain in there since
    # no other item was fetched
    time.sleep(0.11)
    assert "foo" in c
    # But because we get the item, it activates again
    # And it won't get evicted when we fetch bar after
    # it
    assert c["foo"] == hash("foo")
    assert c["bar"] == hash("bar")
    assert "foo" in c
    assert "bar" in c
    assert c["spam"]  # make "bar" not the last lookup

    # Wait again, now get a previously nonexistent item to trigger removing foo
    time.sleep(0.11)
    assert c["bar"] == hash("bar")
    assert "foo" not in c
    assert "bar" in c
    assert len(c) == 1

    time.sleep(0.11)
    # === Again, but with two
    assert c["foo"] == hash("foo")
    assert c["bar"] == hash("bar")
    assert len(c) == 2

    # Wait for the entry to become old.
    # The first one we get() is valid, but the get triggers a cleanup.
    time.sleep(0.11)
    assert c["foo"] == hash("foo")
    assert "bar" not in c
    assert len(c) == 1

    # Wait again, now manually trigger cleanup
    time.sleep(0.11)
    c.check_lifetimes()
    assert "foo" not in c
    assert "bar" not in c
    assert len(c) == 0

    # === Put a lot of items in there
    for i in range(1000):
        c[i]
    assert len(c) == 1000

    for i in [0, 1, 500, 998, 999]:  # testing 1000 can take too long in debug mode
        assert c[i] == hash(i)

    time.sleep(1.1)
    assert c[0] == hash(0)
    for i in range(1, 1000):
        assert i not in c
    assert len(c) == 1


def test_cache_minimum_items():
    c = TemporalCache(0.1, getter=hash, minimum_items=5)

    assert c["foo0"] == hash("foo0")
    assert c["foo1"] == hash("foo1")
    assert c["foo2"] == hash("foo2")
    assert c["foo3"] == hash("foo3")
    assert c["foo4"] == hash("foo4")

    # Wait for the entry to become old.
    # The item should remain in there since
    # no other item was fetched
    time.sleep(0.11)
    assert "foo0" in c
    assert "foo1" in c
    assert "foo2" in c
    assert "foo3" in c
    assert "foo4" in c

    # Typically, without minimum_items set to 2, the following would
    # cause an eviction
    assert c["foo4"] == hash("foo4")
    # However, we expect the fiels to stay in
    assert "foo0" in c
    assert "foo1" in c
    assert "foo2" in c
    assert "foo3" in c
    assert "foo4" in c

    time.sleep(0.11)
    # Even manually clearing shouldn't remove our 5 files
    c.check_lifetimes()
    assert "foo0" in c
    assert "foo1" in c
    assert "foo2" in c
    assert "foo3" in c
    assert "foo4" in c

    # Adding 2 new items in the cache should evict the 2 oldest ones
    time.sleep(0.11)
    assert c["bar0"] == hash("bar0")
    assert c["bar1"] == hash("bar1")
    # foo0 and foo1 were inserted first
    # never accessed again, so they should be evicted
    assert "foo0" not in c
    assert "foo1" not in c
    # foo2, foo3 and foo4 should still be in the cache
    # given that we want to keep at least 5 items
    assert "foo2" in c
    assert "foo3" in c
    assert "foo4" in c

    # Now access all the items in the cache
    # And have more than 5 items
    # since no item is older than the lifetime, they should all
    # remain in the cache
    assert len(c) == 5
    # Refresh all the lifetimes
    assert c["foo2"] == hash("foo2")
    assert c["foo3"] == hash("foo3")
    assert c["foo4"] == hash("foo4")
    assert c["bar0"] == hash("bar0")
    assert c["bar1"] == hash("bar1")
    # New items to add
    assert c["foo0"] == hash("foo0")
    assert c["foo1"] == hash("foo1")
    assert len(c) == 7
    assert "foo2" in c
    assert "foo3" in c
    assert "foo4" in c
    assert "bar0" in c
    assert "bar1" in c
    assert "foo0" in c
    assert "foo1" in c


def test_shape_text_hb():
    font = font_manager._fallback_font
    text = "HelloWorld"

    result = shape_text_hb(text, font.filename)
    check_result(*result)


def test_shape_text_ft():
    font = font_manager._fallback_font
    text = "HelloWorld"

    result = shape_text_ft(text, font.filename)
    check_result(*result)


def check_result(indices, positions, meta):
    assert isinstance(indices, np.ndarray)
    assert indices.dtype == np.uint32 and indices.ndim == 1

    assert isinstance(positions, np.ndarray)
    assert (
        positions.dtype == np.float32
        and positions.ndim == 2
        and positions.shape[1] == 2
    )

    assert isinstance(meta, dict)
    assert "extent" in meta.keys()
    assert meta["extent"] > positions[:, 0].max()
    assert "ascender" in meta.keys()
    assert "descender" in meta.keys()
    assert "direction" in meta.keys()
    assert "script" in meta.keys()


def test_shape_direction_hb():
    font = font_manager._fallback_font
    text = "HelloWorld"

    _, positions1, meta1 = shape_text_hb(text, font.filename, "ltr")
    _, positions2, meta2 = shape_text_hb(text, font.filename, "ttb")

    assert meta1["extent"] > 0
    assert meta2["extent"] > 0
    assert np.all(positions1[2:, 0] > 1)
    assert np.all(positions1[:, 1] == 0)
    assert np.all(np.abs(positions2[:, 0]) < 0.5)  # x-offsets
    assert np.all(np.abs(positions2[2:, 1]) > 1)


def test_glyph_size():
    font = font_manager._fallback_font

    _, positions1, meta1 = shape_text_hb("iiii", font.filename)
    _, positions2, meta2 = shape_text_hb("MMMM", font.filename)

    xx1 = positions1[1:, 0]
    xx2 = positions2[1:, 0]
    assert np.all(xx1 < xx2)

    assert meta2["extent"] > meta1["extent"]


def check_speed():
    font = font_manager._fallback_font
    text = "HelloWorld"

    t0 = time.perf_counter()
    for _i in range(1000):
        shape_text_hb(text, font.filename)
    dt = time.perf_counter() - t0
    print(
        f"shape_text_hb: {1000 * dt:0.1f} ms total",
        f"{1000 * dt / (10000):0.3f} ms per glyph",
    )

    t0 = time.perf_counter()
    for _i in range(1000):
        shape_text_ft(text, font.filename)
    dt = time.perf_counter() - t0
    print(
        f"shape_text_ft: {1000 * dt:0.1f} ms total",
        f"{1000 * dt / (10000):0.3f} ms per glyph",
    )

    # No cache:    about 0.03  and 0.02  ms per glyph for Harfbuzz and FreeType, respectively.
    # Witch cache: about 0.001 and 0.003 ms per glyph for Harfbuzz and FreeType, respectively.


def check_mem_to_store_face_objects():
    """Function to test how much memory it costs to store a FreeType
    or HarfBuzz font object. Intended to be run locally, for reference.

    Results:
    * For FT, about 700 KB for NotoSansSC-Regular.otf
    * For FT, about  37 KB for NotoSans-Regular.ttf
    * For HB, about 16 KB for NotoSansSC-Regular.otf
    * For HB, about 16 KB for NotoSans-Regular.ttf
    """
    import psutil, freetype, uharfbuzz  # noqa

    font = font_manager._fallback_font
    font_filename = font.filename
    # font_filename = "/Users/almar/dev/py/pygfx/pygfx/data_files/fonts/noto_cjk_man/NotoSansSC-Regular.otf"

    m0 = psutil.Process(os.getpid()).memory_info().rss / 1024
    ft_faces = []

    for _j in range(10):
        for _i in range(100):
            face = freetype.Face(font_filename)
            face.set_pixel_sizes(48, 48)
            ft_faces.append(face)
        m1 = psutil.Process(os.getpid()).memory_info().rss / 1024
        dm = (m1 - m0) / len(ft_faces)
        print(f"{dm:0.0f} KiB per ft face")

    m0 = psutil.Process(os.getpid()).memory_info().rss / 1024
    hb_faces = []

    for _j in range(10):
        for _i in range(100):
            blob = uharfbuzz.Blob.from_file_path(font_filename)
            face = uharfbuzz.Face(blob)
            font = uharfbuzz.Font(face)
            font.scale = 48, 48
            hb_faces.append((blob, face, font))
        m1 = psutil.Process(os.getpid()).memory_info().rss / 1024
        dm = (m1 - m0) / len(hb_faces)
        print(f"{dm:0.0f} KiB per hb face")


if __name__ == "__main__":
    for ob in list(globals().values()):
        if callable(ob) and ob.__name__.startswith("test_"):
            print(f"{ob.__name__} ...")
            ob()
    print("done")

    check_speed()
    # check_mem_to_store_face_objects()
