import time
import numpy as np
from pygfx.utils.text._sdf import generate_glyph
from pygfx.utils.text import font_manager, shape_text, glyph_atlas


def test_generate_glyph():
    text = "helloFooBar"

    font = font_manager._fallback_font
    indices, positions, meta = shape_text(text, font.filename)

    atlas_indices = generate_glyph(indices, font.filename)

    assert isinstance(atlas_indices, np.ndarray)
    assert atlas_indices.dtype == np.uint32
    assert atlas_indices.ndim == 1


def test_glyph_size():
    font = font_manager._fallback_font

    indices, _, _ = shape_text("iiii", font.filename)
    atlas_indices1 = generate_glyph(indices, font.filename)

    indices, _, _ = shape_text("MMMM", font.filename)
    atlas_indices2 = generate_glyph(indices, font.filename)

    indices, _, _ = shape_text("____", font.filename)
    atlas_indices3 = generate_glyph(indices, font.filename)

    h1, w1 = glyph_atlas.get_region(atlas_indices1[0]).shape
    h2, w2 = glyph_atlas.get_region(atlas_indices2[0]).shape
    h3, w3 = glyph_atlas.get_region(atlas_indices3[0]).shape

    # The "i" is narrow
    assert w1 < w2 and w1 < w3

    # The "_" is flat
    assert h3 < h1 and h3 < h2


def check_speed():
    text = "HelloWorld"

    font = font_manager._fallback_font
    indices, positions, meta = shape_text(text, font.filename)

    # Make sure that the atlas has the glyphs
    generate_glyph(indices, font.filename)

    t0 = time.perf_counter()
    for _i in range(1000):
        generate_glyph(indices, font.filename)
    dt = time.perf_counter() - t0
    print(
        f"generate_glyph: {1000 * dt:0.1f} ms total",
        f"{1000 * dt / (10000):0.3f} ms per glyph",
    )

    # If I tweak generate_glyph() to include the time in the hash,
    # I can measure that it takes about 1 ms per glyph when SDF must be generated.
    # Otherwise, looking up the atlas indices takes about 0.5 us per glyph.


if __name__ == "__main__":
    for ob in list(globals().values()):
        if callable(ob) and ob.__name__.startswith("test_"):
            print(f"{ob.__name__} ...")
            ob()
    print("done")

    check_speed()
