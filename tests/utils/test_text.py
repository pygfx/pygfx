import numpy as np

from pygfx.utils.text._atlas import GlyphAtlas
from pygfx.utils.text import glyph_atlas as global_atlas


def glyphgen(size):
    return np.ones((size, size), np.uint8)


def test_atlas_alloc1():
    # Test basic allocations

    atlas = GlyphAtlas(8, 8)

    # Start small: allocate 4 regions
    sizes = [(10, 10), (12, 8), (8, 20), (20, 3)]
    indices = []
    for i, (w, h) in enumerate(sizes):
        index = atlas.allocate_region(w, h)
        indices.append(index)
        atlas.set_region(index, 100 + i)

    # Check
    for i, index in enumerate(indices):
        region = atlas.get_region(index)
        assert (region.shape[1], region.shape[0]) == sizes[i]
        assert np.all(region[:] == 100 + i)

    # Now fill up to 100 regions
    for i in range(len(sizes), 100):
        w, h = np.random.randint(5, 20), np.random.randint(5, 20)
        sizes.append((w, h))
        index = atlas.allocate_region(w, h)
        indices.append(index)
        atlas.set_region(index, 100 + i)

    # Check again
    for i, index in enumerate(indices):
        region = atlas.get_region(index)
        assert (region.shape[1], region.shape[0]) == sizes[i]
        assert np.all(region[:] == 100 + i)

    assert atlas.allocated_area > 1000
    assert atlas.total_area > 10000

    # De-allocate
    for index in indices:
        atlas.free_region(index)

    assert atlas.allocated_area == 0

    # Sanity check
    assert len(indices) == len(set(indices))


def test_atlas_alloc2():
    # Now with de-allocations

    atlas = GlyphAtlas(8, 8)

    assert atlas.allocated_area == 0
    assert atlas.total_area == 64

    # Allocate 1000 50x50 regions
    indices1 = []
    for i in range(1000):
        index = atlas.allocate_region(50, 50)
        indices1.append(index)

    assert atlas.region_count == 1000
    assert atlas.allocated_area == 2500000
    assert atlas.total_area == 4064256

    # De-allocate half
    for index in indices1[::2]:
        atlas.free_region(index)

    assert atlas.region_count == 500
    assert atlas.allocated_area == 1250000
    assert atlas.total_area == 4064256

    # Allocate 500
    indices2 = []
    for i in range(500):
        index = atlas.allocate_region(50, 50)
        indices2.append(index)

    assert atlas.region_count == 1000
    assert set(indices2) == set(indices1[::2])  # Verify index re-use
    assert atlas.allocated_area == 2500000
    assert atlas.total_area == 4064256

    # De-allocate half
    for index in indices1[1::2]:
        atlas.free_region(index)

    assert atlas.region_count == 500
    assert atlas.allocated_area == 1250000
    assert atlas.total_area == 4064256

    # De-allocate the rest
    for index in indices1[::2]:
        atlas.free_region(index)

    assert atlas.region_count == 0
    assert atlas.allocated_area == 0
    assert atlas.total_area == 6400  # atlas is reduced in size


def test_atlas_alloc3():
    # Re-use allocations

    atlas = GlyphAtlas(8, 100)

    assert atlas.region_count == 0
    assert atlas.allocated_area == 0
    assert atlas.total_area == 10000

    # Allocate so the atlas is *exactly* filled
    for i in range(100):
        atlas.allocate_region(10, 10)

    assert atlas.region_count == 100
    assert atlas.allocated_area == 10000
    assert atlas.total_area == 10000

    # Deallocate over half of regions
    for i in range(50):
        atlas.free_region(i)

    assert atlas.region_count == 50
    assert atlas.allocated_area == 5000
    assert atlas.total_area == 10000

    # We can now allocate 50 regions for free
    for i in range(50):
        atlas.allocate_region(10, 10)

    assert atlas.region_count == 100
    assert atlas.allocated_area == 10000
    assert atlas.total_area == 10000

    # And another allocation causes a resize
    atlas.allocate_region(10, 10)

    assert atlas.region_count == 101
    assert atlas.allocated_area == 10100
    assert atlas.total_area == 20736


def test_atlas_glyps():
    assert isinstance(global_atlas, GlyphAtlas)

    atlas = GlyphAtlas()

    # Test adding glyphs
    gs = 50
    array_id = id(atlas._array)

    # No glyph with this hash yet
    assert atlas.get_index_from_hash("0") is None

    # Now there is
    i0 = atlas.store_region_with_hash("0", glyphgen(gs))
    assert isinstance(i0, int)
    atlas.get_index_from_hash("0") == i0

    # More ...
    i1 = atlas.store_region_with_hash("1", glyphgen(gs))
    i2 = atlas.store_region_with_hash("2", glyphgen(gs))
    i3 = atlas.store_region_with_hash("3", glyphgen(gs))

    assert [i0, i1, i2, i3] == [0, 1, 2, 3]
    assert array_id == id(atlas._array)

    # Adding one more triggers a resize
    i4 = atlas.store_region_with_hash("4", glyphgen(gs))
    assert i4 == 4
    assert array_id != id(atlas._array)
    array_id = id(atlas._array)


if __name__ == "__main__":
    for ob in list(globals().values()):
        if callable(ob) and ob.__name__.startswith("test_"):
            print(f"{ob.__name__} ...")
            ob()
    print("done")
