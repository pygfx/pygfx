import numpy as np
from pytest import raises

from pygfx.utils.text._atlas import GlyphAtlas
from pygfx.utils.text import atlas as global_atlas


def glyphgen(size):
    return np.ones((size, size), np.uint8)


def test_atlas():

    assert isinstance(global_atlas, GlyphAtlas)

    atlas = GlyphAtlas()

    assert atlas.capacity > 0
    assert atlas.glyph_size > 0

    # Check capacity
    # This _ensure_capacity is the only private method we'll use
    atlas._ensure_capacity(4)
    assert atlas.capacity == 4

    atlas._ensure_capacity(400)
    assert atlas.capacity == 400

    atlas._ensure_capacity(0)
    assert atlas.capacity == 4  # minimum

    atlas._ensure_capacity(19)
    assert atlas.capacity == 25  # nrows == ncols

    # Test adding glyphs
    gs = atlas.glyph_size
    atlas._ensure_capacity(4)
    array_id = id(atlas._array)
    i0 = atlas.register_glyph("0", glyphgen, gs)
    i1 = atlas.register_glyph("1", glyphgen, gs)
    i2 = atlas.register_glyph("2", glyphgen, gs)
    i3 = atlas.register_glyph("3", glyphgen, gs)

    # Glyph must have the correct size
    with raises(ValueError):
        atlas.register_glyph("99", glyphgen, gs - 1)
    with raises(ValueError):
        atlas.register_glyph("99", glyphgen, gs + 1)

    assert [i0, i1, i2, i3] == [0, 1, 2, 3]
    assert atlas.capacity == 4
    assert array_id == id(atlas._array)

    # Adding one more triggers a resize
    i4 = atlas.register_glyph("4", glyphgen, gs)
    assert i4 == 4
    assert atlas.capacity == 9
    assert array_id != id(atlas._array)
    array_id = id(atlas._array)

    # Cannot reduce size
    atlas._ensure_capacity(4)
    assert atlas.capacity == 9
    assert array_id == id(atlas._array)

    # Oh, and setting same capacity is a no-op
    atlas._ensure_capacity(9)
    assert atlas.capacity == 9
    assert array_id == id(atlas._array)

    # Remove some
    atlas.free_slot(0)
    atlas.free_slot(1)
    atlas.free_slot(2)

    # Slot at index 4 is holding things up
    atlas._ensure_capacity(4)
    assert atlas.capacity == 9

    # This should work
    atlas.free_slot(4)
    atlas._ensure_capacity(4)
    assert atlas.capacity == 4

    # Allocate three slots. The atlas will prefer the lower slots!
    i0 = atlas.register_glyph("0", glyphgen, gs)
    i1 = atlas.register_glyph("1", glyphgen, gs)
    i2 = atlas.register_glyph("2", glyphgen, gs)
    assert [i0, i1, i2] == [0, 1, 2]
    assert atlas.capacity == 4

    # The next will go in pos 4
    i4 = atlas.register_glyph("4", glyphgen, gs)
    assert i4 == 4
    assert atlas.capacity == 9

    # Registering with existing hash will not populate a new slot
    for i in range(10):
        assert atlas.register_glyph("0", glyphgen, gs) == 0
        assert atlas.register_glyph("1", glyphgen, gs) == 1
        assert atlas.register_glyph("2", glyphgen, gs) == 2
        assert atlas.register_glyph("3", glyphgen, gs) == 3

    assert atlas.capacity == 9


if __name__ == "__main__":
    for ob in list(globals().values()):
        if callable(ob) and ob.__name__.startswith("test_"):
            print(f"{ob.__name__} ...")
            ob()
    print("done")
