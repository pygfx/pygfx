import random

import numpy as np
import pygfx as gfx
from pygfx.utils.text._atlas import GlyphAtlas, SIZES, get_suitable_size
from pygfx.utils.text import glyph_atlas as global_atlas

from pytest import raises


def glyphgen(size):
    return np.ones((size, size), np.uint8)


def test_sizes():
    # Test that the predefined atlas sizes are multiples of 8,
    # and that the areas are about doubled at each step.
    prev_area = 128
    for size, area, _ in SIZES:
        assert size % 8 == 0
        assert size * size == area
        assert 1.7 * prev_area <= area < 2.3 * prev_area
        prev_area = area

    # Test that going back and forth between sizes works as expected
    for i in range(1, len(SIZES)):
        size1 = SIZES[i - 1][0]
        size2 = SIZES[i][0]
        assert get_suitable_size(size1 * size1 * 2) == size2
        assert get_suitable_size(size2 * size2 / 2) == size1


def test_atlas_resize_edge_cases():
    # Create a small atlas 16x16
    atlas = GlyphAtlas(256, 16)

    # Allocate a region that's bigger than the atlas
    atlas.allocate_region(20, 20)
    assert atlas.total_area == 24 * 24

    # And yet bigger again
    atlas.allocate_region(80, 80)
    assert atlas.total_area == 128 * 128

    # And bigger than we can handle
    too_large = SIZES[-1][0] + 1
    with raises(RuntimeError):
        atlas.allocate_region(too_large, too_large)


def test_atlas_resize():
    # Test the exact points of resize with controlled allocs

    # Create a small atlas 16x16
    atlas = GlyphAtlas(256, 16)

    assert atlas.allocated_area == 0
    assert atlas.total_area == 256

    prev_array = atlas._array

    # Allocate 4 regions
    for _i in range(4):
        atlas.allocate_region(8, 8)

    # This should just fit
    assert atlas.total_area == 256
    assert atlas.allocated_area == 256
    assert atlas._array is prev_array

    # Allocate another
    atlas.allocate_region(8, 8)

    # Now the array was resized to 24x24
    assert atlas.total_area == 576
    assert atlas.allocated_area == 256 + 64
    assert atlas._array is not prev_array

    prev_array = atlas._array

    # We can fit 9 regions of 8x8 in it. So 4 more.
    for _i in range(4):
        atlas.allocate_region(8, 8)

    # Again, it should just fit
    assert atlas.total_area == 576
    assert atlas.allocated_area == 576
    assert atlas._array is prev_array

    # Allocate another
    atlas.allocate_region(8, 8)

    # Now the array was resized to 40x40
    assert atlas.total_area == 1024
    assert atlas.allocated_area == 576 + 64
    assert atlas._array is not prev_array


def test_atlas_alloc_resize_with_freeing():
    # Test the exact points of resize, now with freeing regions

    def get_index_in_use():
        for index in range(atlas._index_counter):
            if atlas.get_region(index).size > 0:
                return index

    # Create a small atlas 24x24
    atlas = GlyphAtlas(256, 24)

    assert atlas.allocated_area == 0
    assert atlas.total_area == 576  # 24x24

    prev_array = atlas._array

    # Allocate 9 regions
    for _i in range(9):
        atlas.allocate_region(8, 8)

    # This should just fit
    assert atlas.total_area == 576
    assert atlas.allocated_area == 9 * 64
    assert atlas._array is prev_array

    # Free half of the area
    for _i in range(4):
        atlas.free_region(get_index_in_use())
    assert atlas.allocated_area == 5 * 64
    assert len(atlas._free_indices) == 4

    # Allocate another
    atlas.allocate_region(8, 8)

    # Now the atlas was repacked, but not resized
    assert atlas.total_area == 576
    assert atlas.region_count == 6
    assert atlas.allocated_area == 6 * 64

    # The free indices (into the infos array) are still there
    assert len(atlas._free_indices) == 3  # one it taken by the allocation

    # Not only is the atlas not resized, the array is also still the
    # same object. This is important, because this means that the
    # texture that wraps it also has not changed, which means that
    # text objects don't have to rebuild their bindings and pipeline.
    assert atlas._array is prev_array

    # Allocate 6 regions. Now we have 12 regions in total, which causes a resize.
    for _i in range(6):
        atlas.allocate_region(8, 8)
    assert atlas.region_count == 12

    # Now the array was resized to 40x40
    assert atlas.total_area == 1024
    assert atlas.allocated_area == 12 * 64
    assert atlas._array is not prev_array

    # The free indices (into the infos array) are still there
    assert len(atlas._free_indices) == 0  # because we used all the free region

    # Now we free some regions
    for _i in range(6):
        atlas.free_region(get_index_in_use())

    # We now have half the area freed. Not enough to cause a downsize.
    assert atlas.region_count == 6
    assert atlas.total_area == 1024
    assert atlas.allocated_area == 6 * 64

    # Free more regions
    for _i in range(3):
        atlas.free_region(get_index_in_use())

    # We now have three quarters of the area freed. Downsize!
    assert atlas.region_count == 3
    assert atlas.total_area == 576
    assert atlas.allocated_area == 3 * 64
    assert len(atlas._free_indices) == 9

    # Free all we have
    for _i in range(3):
        atlas.free_region(get_index_in_use())

    # The size does not shrink lower than the initial size
    assert atlas.region_count == 0
    assert atlas.total_area == 576
    assert atlas.allocated_area == 0 * 64


def test_atlas_alloc1():
    # Test basic allocations and verify the region, also after resizes

    atlas = GlyphAtlas(16, 16)

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


def test_atlas_alloc_empty():
    # Allocate empty regions (zero size)
    # This is not even such a special case; it just works.

    atlas = GlyphAtlas(16, 16)

    i1 = atlas.allocate_region(5, 5)
    i2 = atlas.allocate_region(0, 0)
    i3 = atlas.allocate_region(0, 0)
    i4 = atlas.allocate_region(5, 5)

    # Unique indices
    assert len({i1, i2, i3, i4}) == 4

    assert atlas.get_region(i1).size == 25
    assert atlas.get_region(i2).size == 0
    assert atlas.get_region(i3).size == 0
    assert atlas.get_region(i4).size == 25


def test_atlas_alloc2():
    # Now with de-allocations

    atlas = GlyphAtlas(16, 16)

    assert atlas.allocated_area == 0
    assert atlas.total_area == 256

    # Allocate 1000 50x50 regions
    indices1 = []
    for _i in range(1000):
        index = atlas.allocate_region(50, 50)
        indices1.append(index)

    assert atlas.region_count == 1000
    assert atlas.allocated_area == 2500000
    assert atlas.total_area == 2048**2

    # De-allocate half
    for index in indices1[::2]:
        atlas.free_region(index)

    assert atlas.region_count == 500
    assert atlas.allocated_area == 1250000
    assert atlas.total_area == 2048**2

    # Allocate 500
    indices2 = []
    for _i in range(500):
        index = atlas.allocate_region(50, 50)
        indices2.append(index)

    assert atlas.region_count == 1000
    assert set(indices2) == set(indices1[::2])  # Verify index re-use
    assert atlas.allocated_area == 2500000
    assert atlas.total_area == 2048**2

    # De-allocate half
    for index in indices1[1::2]:
        atlas.free_region(index)

    assert atlas.region_count == 500
    assert atlas.allocated_area == 1250000
    assert atlas.total_area == 2048**2

    # De-allocate the rest
    for index in indices1[::2]:
        atlas.free_region(index)

    assert atlas.region_count == 0
    assert atlas.allocated_area == 0
    assert atlas.total_area == 256**2  # atlas is reduced in size


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
    assert atlas.get_index_from_hash("0") == i0

    # More ...
    i1 = atlas.store_region_with_hash("1", glyphgen(gs))
    i2 = atlas.store_region_with_hash("2", glyphgen(gs))
    i3 = atlas.store_region_with_hash("3", glyphgen(gs))

    assert [i0, i1, i2, i3] == [0, 1, 2, 3]
    assert array_id == id(atlas._array)


def test_against_glyph_bleeding():
    big_tex = """
    Lorem ipsum odor amet, consectetuer adipiscing elit. Congue aliquet fusce hendrerit leo fames ac. Proin nec sit mauris lobortis quam ultrices. Senectus habitasse ad orci posuere fusce. Ut lectus inceptos commodo taciti porttitor a habitasse. Vulputate tempus ullamcorper aptent molestie vestibulum massa. Tristique nec ac sagittis morbi; egestas nisl donec morbi. Et nisi donec conubia duis rutrum tellus?
    """
    chars = list(big_tex)

    camera = gfx.OrthographicCamera(340, 200)
    target = gfx.Texture(dim=2, size=(3400, 2000), format="rgba8unorm")
    renderer = gfx.WgpuRenderer(target, pixel_ratio=1)

    def render_text():
        text = gfx.Text(
            text=big_tex,
            max_width=300,
            material=gfx.TextMaterial(color="#fff", outline_thickness=5),
        )
        scene = gfx.Scene().add(gfx.Background.from_color("#fff"), text)
        renderer.render(scene, camera)
        return renderer.snapshot()

    # Render the text
    im1 = render_text()

    # Reset and shuffle the atlas
    global_atlas.__init__()
    random.shuffle(chars)
    gfx.Text(text="".join(chars))

    # Render the text again, glyphs are now in a different spot in the atlas
    im2 = render_text()

    # Get the difference
    diff = (im1 != im2).max(axis=2)
    diff_count = diff.sum()
    # diff_image = diff.astype("u1") * 255
    if diff_count:
        print("Number of pixels that differ:", diff_count)
    assert diff_count == 0


if __name__ == "__main__":
    for ob in list(globals().values()):
        if callable(ob) and ob.__name__.startswith("test_"):
            print(f"{ob.__name__} ...")
            ob()
    print("done")
