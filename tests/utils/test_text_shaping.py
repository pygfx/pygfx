import os
import time

import numpy as np
from pygfx.utils.text._shaper import TemperalCache, shape_text_hb, shape_text_ft
from pygfx.utils.text import font_manager


def test_cache():
    c = TemperalCache(0.1)

    # === Store one value
    c.set("foo", 42)
    assert c.get("foo") == 42
    assert c.get("foo") == 42
    assert len(c._cache) == 1

    # Wait for the entry to become old.
    # But because we get the item, it activates again
    time.sleep(0.11)
    assert c.get("foo") == 42
    assert c.get("foo") == 42

    # Wait again, now get a nonexistent item to trigger removing foo
    time.sleep(0.11)
    assert c.get("bar") is None
    assert c.get("foo") is None
    assert len(c._cache) == 0

    # === Again, but with two
    c.set("foo", 42)
    c.set("bar", 90)
    assert c.get("foo") == 42
    assert c.get("bar") == 90
    assert len(c._cache) == 2

    # Wait for the entry to become old.
    # The first one we get() is valid, but the get triggers a cleanup.
    time.sleep(0.11)
    assert c.get("foo") == 42
    assert c.get("bar") is None
    assert len(c._cache) == 1

    # Wait again, now get a nonexistent item to trigger removing foo
    time.sleep(0.11)
    assert c.get("bar") is None
    assert c.get("foo") is None
    assert len(c._cache) == 0

    # === Put a lot of items in there
    for i in range(1000):
        c.set(i, i)
    assert len(c._cache) == 1000

    for i in [0, 1, 500, 998, 999]:  # testing 1000 can take too long in debug mode
        assert c.get(i) == i

    time.sleep(1.1)
    assert c.get(0) == 0
    for i in range(1, 1000):
        assert c.get(i) is None
    assert len(c._cache) == 1


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
    assert meta2["extent"] < 0
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
    for i in range(1000):
        shape_text_hb(text, font.filename)
    dt = time.perf_counter() - t0
    print(
        f"shape_text_hb: {1000*dt:0.1f} ms total",
        f"{1000*dt/(10000):0.3f} ms per glyph",
    )

    t0 = time.perf_counter()
    for i in range(1000):
        shape_text_ft(text, font.filename)
    dt = time.perf_counter() - t0
    print(
        f"shape_text_ft: {1000*dt:0.1f} ms total",
        f"{1000*dt/(10000):0.3f} ms per glyph",
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
    # font_filename = "/Users/almar/dev/py/pygfx/pygfx/pkg_resources/fonts/noto_cjk_man/NotoSansSC-Regular.otf"

    m0 = psutil.Process(os.getpid()).memory_info().rss / 1024
    ft_faces = []

    for j in range(10):
        for i in range(100):
            face = freetype.Face(font_filename)
            face.set_pixel_sizes(48, 48)
            ft_faces.append(face)
        m1 = psutil.Process(os.getpid()).memory_info().rss / 1024
        dm = (m1 - m0) / len(ft_faces)
        print(f"{dm:0.0f} KiB per ft face")

    m0 = psutil.Process(os.getpid()).memory_info().rss / 1024
    hb_faces = []

    for j in range(10):
        for i in range(100):
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
