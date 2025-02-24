import time

import numpy as np
import pytest

from pygfx import TextGeometry, MultiTextGeometry, TextBlock


def test_text_geometry_text():
    # Let's try some special cases first

    min_blocks = 8
    min_glyphs = 16

    # Can instantiate empty text
    geo = TextGeometry()
    assert len(geo._text_blocks) == 0
    assert geo._glyph_count == 0
    assert geo.positions.nitems == min_blocks

    # Empty string - still has one item (whitespace)
    geo = TextGeometry(text="")
    assert len(geo._text_blocks) == 0
    assert geo._glyph_count == 0
    assert geo.positions.nitems == min_blocks
    assert geo.glyph_atlas_indices.nitems == min_glyphs

    # Only a newline
    geo = TextGeometry("\n")  # also test that text is a positional arg
    assert len(geo._text_blocks) == 1
    assert geo._glyph_count == 0
    assert geo.positions.nitems == min_blocks
    assert geo.glyph_atlas_indices.nitems == min_glyphs

    # Only a space
    geo = TextGeometry(" ")  # also test that text is a positional arg
    assert len(geo._text_blocks) == 1
    assert geo._glyph_count == 0
    assert geo.positions.nitems == min_blocks
    assert geo.glyph_atlas_indices.nitems == min_glyphs

    # One char
    geo = TextGeometry(text="a")
    assert len(geo._text_blocks) == 1
    assert len(geo._text_blocks[0]._text_items) == 1
    assert geo._glyph_count == 1
    assert geo.positions.nitems == min_blocks
    assert geo.glyph_atlas_indices.nitems == min_glyphs

    # Two words with 3 chars in total
    geo = TextGeometry(text="a bc")
    assert len(geo._text_blocks) == 1
    assert len(geo._text_blocks[0]._text_items) == 2
    assert geo._glyph_count == 3
    assert geo.positions.nitems == min_blocks
    assert geo.glyph_atlas_indices.nitems == min_glyphs

    # Can set new text, buffers are recreated
    geo.set_text("foo\nbar")
    assert len(geo._text_blocks) == 2
    assert geo._glyph_count == 6
    assert geo.positions.nitems == min_blocks
    assert geo.glyph_atlas_indices.nitems == min_glyphs

    # Mo text
    geo.set_text("foo bar spam eggs\naap noot mies")
    assert len(geo._text_blocks) == 2
    assert geo._glyph_count == 25
    assert geo.positions.nitems == min_blocks
    assert geo.glyph_atlas_indices.nitems == 32

    # If setting smaller text, buffer size is oversized
    geo.set_text("x")
    assert len(geo._text_blocks) == 1
    assert geo._glyph_count == 1
    assert geo.positions.nitems == min_blocks
    assert geo.glyph_atlas_indices.nitems == 32

    # Sizes are used to invalidate unused slots
    assert np.all(geo.glyph_sizes.data[1:] == 0)


def test_text_geometry_blocks():
    geo = MultiTextGeometry()

    block0 = geo.create_text_block()
    blocks19 = geo.create_text_blocks(9)
    blocks = [block0, *blocks19]
    assert geo.get_text_block_count() == 10
    for b in blocks:
        assert isinstance(b, TextBlock)

    # Set to same amount of blocks, no effect
    geo.set_text_block_count(10)
    for i in range(10):
        assert blocks[i] is geo.get_text_block(i)

    # Set to more blocks
    geo.set_text_block_count(12)
    for i in range(10):
        assert blocks[i] is geo.get_text_block(i)

    # Set to less blocks
    geo.set_text_block_count(8)
    for i in range(8):
        assert blocks[i] is geo.get_text_block(i)
    with pytest.raises(IndexError):
        geo.get_text_block(8)

    # Set back to 10. The last two blocks are now different.
    geo.set_text_block_count(10)
    for i in range(8):
        assert blocks[i] is geo.get_text_block(i)
    for i in range(8, 10):
        assert blocks[i] is not geo.get_text_block(i)


def test_text_geometry_blocks_buffer():
    geo = MultiTextGeometry()
    assert geo.positions.nitems == 8

    geo.set_text_block_count(8)
    assert geo.positions.nitems == 8

    geo.set_text_block_count(9)
    assert geo.positions.nitems == 16

    geo.set_text_block_count(16)
    assert geo.positions.nitems == 16

    geo.set_text_block_count(17)
    assert geo.positions.nitems == 32

    geo.set_text_block_count(33)
    assert geo.positions.nitems == 64

    # Don't scale back too soon
    geo.set_text_block_count(32)
    assert geo.positions.nitems == 64

    # Don't scale back too soon
    geo.set_text_block_count(16)
    assert geo.positions.nitems == 64

    # Still not
    geo.set_text_block_count(8)
    assert geo.positions.nitems == 8


def test_text_geometry_blocks_reuse():
    geo = TextGeometry()

    geo.set_text("foo\nbar\nspam")
    blocks1 = geo._text_blocks.copy()
    assert len(blocks1) == 3

    geo.set_text("aap\nnoot\nmies")
    blocks2 = geo._text_blocks.copy()
    assert len(blocks2) == 3

    for i in range(3):
        assert blocks1[i] is blocks2[i]


def test_text_geometry_items_reuse():
    geo = TextGeometry()

    geo.set_text("foo bar spam")
    blocks1 = geo._text_blocks.copy()
    items1 = geo._text_blocks[0]._text_items.copy()
    assert len(blocks1) == 1
    assert len(items1) == 3

    geo.set_text("aap noot mies")
    blocks2 = geo._text_blocks.copy()
    items2 = geo._text_blocks[0]._text_items.copy()
    assert len(blocks2) == 1
    assert len(items2) == 3

    assert blocks1[0] is blocks2[0]
    for i in range(3):
        assert items1[i] is items2[i]


def test_text_geometry_whitespace():
    # Tests how whitespace is handled during itemization and layout.

    # With left anchor

    t1 = TextGeometry(text="abc def", anchor="baseline-left")
    t2 = TextGeometry(text="abc   def", anchor="baseline-left")
    t3 = TextGeometry(text=" abc def", anchor="baseline-left")
    t4 = TextGeometry(text="abc def ", anchor="baseline-left")

    x1 = t1.glyph_positions.data[t1._glyph_count - 1, 0]
    x2 = t2.glyph_positions.data[t2._glyph_count - 1, 0]
    x3 = t3.glyph_positions.data[t3._glyph_count - 1, 0]
    x4 = t4.glyph_positions.data[t4._glyph_count - 1, 0]

    assert x2 > x1  # Extra spaces in between words are preserved
    assert x3 > x1  # Extra space at the start is also preserved
    assert x4 == x1  # Extra space at the end is dropped

    # With right anchor

    t1 = TextGeometry(text="abc def", anchor="baseline-right")
    t2 = TextGeometry(text="abc   def", anchor="baseline-right")
    t3 = TextGeometry(text=" abc def", anchor="baseline-right")
    t4 = TextGeometry(text="abc def ", anchor="baseline-right")

    x1 = t1.glyph_positions.data[0, 0]
    x2 = t2.glyph_positions.data[0, 0]
    x3 = t3.glyph_positions.data[0, 0]
    x4 = t4.glyph_positions.data[0, 0]

    x1 += t1.positions.data[0, 0]
    x2 += t2.positions.data[0, 0]
    x3 += t3.positions.data[0, 0]
    x4 += t4.positions.data[0, 0]

    assert x2 < x1  # Extra spaces in between words are preserved
    assert x3 == x1  # Extra space at the start is dropped
    assert x4 < x1  # Extra space at the end is preserved


def test_text_geometry_markdown():
    # The same text, but the 2nd one is formatted. Should still
    # result in same number of glyphs and equally positioned.
    t1 = TextGeometry(text="abc def")
    t2 = TextGeometry(markdown="*abc* **def**")

    assert np.all(t1.positions.data == t2.positions.data)


def test_text_geometry_anchor():
    t = TextGeometry("")

    assert t.anchor == "middle-center"

    t.anchor = "top-left"
    assert t.anchor == "top-left"
    t.anchor = "bottom-right"
    assert t.anchor == "bottom-right"
    t.anchor = "middle-center"
    assert t.anchor == "middle-center"
    t.anchor = "baseline-left"
    assert t.anchor == "baseline-left"

    # No dash with alias allows shortcut
    t.anchor = "center"
    assert t.anchor == "middle-center"
    t.anchor = "middle"
    assert t.anchor == "middle-center"

    # No dash
    t.anchor = "topleft"
    assert t.anchor == "top-left"
    t.anchor = "bottomright"
    assert t.anchor == "bottom-right"
    t.anchor = "middlecenter"
    assert t.anchor == "middle-center"
    t.anchor = "baselineleft"
    assert t.anchor == "baseline-left"

    # Default
    t.anchor = None
    assert t.anchor == "middle-center"


def test_text_geometry_direction_ttb():
    t1 = TextGeometry("abc def", direction="ltr")
    x1, y1 = t1.glyph_positions.data[:, 0], t1.glyph_positions.data[:, 1]
    assert (x1.max() - x1.min()) > (y1.max() - y1.min())

    t2 = TextGeometry("abc def", direction="ttb")
    x2, y2 = t2.glyph_positions.data[:, 0], t2.glyph_positions.data[:, 1]
    assert (x2.max() - x2.min()) < (y2.max() - y2.min())


def test_text_geometry_direction_rtl_1():
    # This is a very adversary/weird text: two words of Arabic in the
    # middle of Latin. Put as words on separate lines. Depending on the editor
    # the below list may look weird, but in any case the first Arabic word
    # is shorter (hello) and the second word is longer (world).

    # Since the sentence stars with an Latin word, the block's primary
    # direction is LTR. But the two Arabic words are swapped so the
    # "subsentence" is correct.

    words = [
        "foo",
        "مرحبا",
        "بالعالم",
        "baaaar",
    ]

    # Make it one text
    text = " ".join(words)

    t = TextGeometry(text=text)
    items = t._text_blocks[0]._text_items
    assert len(items) == 4

    assert items[0].direction == "ltr"
    assert items[1].direction == "rtl"
    assert items[2].direction == "rtl"
    assert items[3].direction == "ltr"

    # Use the lengths of the words to identify them.
    # Note that the words are in their original order, no layout yet.
    assert items[0].extent < items[3].extent
    assert items[1].extent < items[2].extent

    x0 = items[0].offset[1]
    x1 = items[1].offset[1]
    x2 = items[2].offset[1]
    x3 = items[3].offset[1]

    # Now check that the middle two words are actually swapped
    assert x0 < x2
    assert x2 < x1
    assert x1 < x3


def test_text_geometry_direction_rt_2():
    # Same thing, but now two latin words in an arabic text.

    # Since the sentence stars with an Arabic word, the block's primary
    # direction is RTL. But the two Latin words are swapped so the
    # "subsentence" is correct.
    words = [
        "مرحبا",
        "foo",
        "baaaar",
        "بالعالم",
    ]

    # Make it one text
    text = " ".join(words)

    t = TextGeometry(text=text)
    items = t._text_blocks[0]._text_items
    assert len(items) == 4

    assert items[0].direction == "rtl"
    assert items[1].direction == "ltr"
    assert items[2].direction == "ltr"
    assert items[3].direction == "rtl"

    # Use the lengths of the words to identify them.
    # Note that the words are in their original order, no layout yet.
    assert items[0].extent < items[3].extent
    assert items[1].extent < items[2].extent

    x0 = items[0].offset[1]
    x1 = items[1].offset[1]
    x2 = items[2].offset[1]
    x3 = items[3].offset[1]

    # Now check that the middle two words are actually swapped
    assert x0 > x2
    assert x2 > x1
    assert x1 > x3


def test_geometry_text_align_1():
    ref_text = (
        "Lorem ipsum\n"
        "Bonjour World Olá\n"  # some text that isn't equal in line
        "pygfx\n"  # a line with exactly 1 word, definitely must not crash
        "last line"
    )

    text = [ref_text]

    def xpos(geo, block_i, glyph_i):
        return geo.positions.data[block_i, 0] + geo.glyph_positions.data[glyph_i, 0]

    def ypos(geo, block_i, glyph_i):
        return geo.positions.data[block_i, 1] + geo.glyph_positions.data[glyph_i, 1]

    # Setting text to list of text, to make sure to use a single block.
    geometry_left = TextGeometry(
        text=text,
        text_align="left",
        anchor="top-left",
    )
    geometry_center = TextGeometry(
        text=text,
        text_align="center",
        anchor="top-left",
    )
    geometry_right = TextGeometry(
        text=text,
        text_align="right",
        anchor="top-left",
    )
    geometry_justify1 = TextGeometry(
        text=text,
        text_align="justify",
        anchor="top-left",
    )
    geometry_justify2 = TextGeometry(
        text=text,
        text_align="justify",
        anchor="top-left",
        max_width=300,
    )
    geometry_justify3 = TextGeometry(
        text=text,
        text_align="justify-all",
        anchor="top-left",
        max_width=300,
    )

    # Check first char
    j = 0
    assert xpos(geometry_left, 0, j) < xpos(geometry_center, 0, j)
    assert xpos(geometry_center, 0, j) < xpos(geometry_right, 0, j)

    assert xpos(geometry_left, 0, j) == xpos(geometry_justify1, 0, j)
    assert xpos(geometry_left, 0, j) == xpos(geometry_justify2, 0, j)

    # Check last char of first line
    j = 9
    assert xpos(geometry_left, 0, j) < xpos(geometry_center, 0, j)
    assert xpos(geometry_center, 0, j) < xpos(geometry_right, 0, j)

    assert xpos(geometry_right, 0, j) == xpos(geometry_justify1, 0, j)
    assert xpos(geometry_right, 0, j) < xpos(geometry_justify2, 0, j)
    assert xpos(geometry_right, 0, j) < xpos(geometry_justify3, 0, j)

    # new line should be lower (more negative) than the previous line
    assert ypos(geometry_left, 0, 10) < ypos(geometry_center, 0, 9)

    # For last char on last line
    j = geometry_left._glyph_count - 1
    assert xpos(geometry_left, 0, j) < xpos(geometry_center, 0, j)
    assert xpos(geometry_center, 0, j) < xpos(geometry_right, 0, j)

    # justify will not justify the last line
    assert xpos(geometry_left, 0, j) == xpos(geometry_justify1, 0, j)
    assert xpos(geometry_left, 0, j) == xpos(geometry_justify2, 0, j)
    # unless you specify justify-all
    assert xpos(geometry_left, 0, j) < xpos(geometry_justify3, 0, j)
    assert xpos(geometry_right, 0, j) < xpos(geometry_justify3, 0, j)


def test_geometry_text_align_2():
    ref_text = (
        "Lorem ipsum\n"
        "Bonjour World Olá\n"  # some text that isn't equal in line
        "pygfx\n"  # a line with exactly 1 word, definitely must not crash
        "last line"
    )

    text = ref_text

    def xpos(geo, block_i, glyph_i):
        return geo.positions.data[block_i, 0] + geo.glyph_positions.data[glyph_i, 0]

    def ypos(geo, block_i, glyph_i):
        return geo.positions.data[block_i, 1] + geo.glyph_positions.data[glyph_i, 1]

    # Setting text to list of text, to make sure to use a single block.
    geometry_left = TextGeometry(
        text=text,
        text_align="left",
        anchor="top-left",
    )
    geometry_center = TextGeometry(
        text=text,
        text_align="center",
        anchor="top-left",
    )
    geometry_right = TextGeometry(
        text=text,
        text_align="right",
        anchor="top-left",
    )
    geometry_justify2 = TextGeometry(
        text=text,
        text_align="justify",
        anchor="top-left",
        max_width=300,
    )
    geometry_justify1 = TextGeometry(
        text=text,
        text_align="justify",
        anchor="top-left",
    )
    geometry_justify2 = TextGeometry(
        text=text,
        text_align="justify",
        anchor="top-left",
        max_width=300,
    )
    geometry_justify3 = TextGeometry(
        text=text,
        text_align="justify-all",
        anchor="top-left",
        max_width=300,
    )

    # Check first char
    i, j = 0, 0
    assert xpos(geometry_left, i, j) < xpos(geometry_center, i, j)
    assert xpos(geometry_center, i, j) < xpos(geometry_right, i, j)

    # Justify with max-width works
    assert xpos(geometry_left, i, j) == xpos(geometry_justify2, i, j)
    # But without max-width does the same as 'center'
    assert xpos(geometry_left, i, j) < xpos(geometry_justify1, i, j)

    # Check last char of first line
    i, j = 0, 9
    assert xpos(geometry_left, i, j) < xpos(geometry_center, i, j)
    assert xpos(geometry_center, i, j) < xpos(geometry_right, i, j)

    assert xpos(geometry_right, i, j) > xpos(geometry_justify1, i, j)
    assert xpos(geometry_right, i, j) > xpos(geometry_justify2, i, j)
    assert xpos(geometry_right, i, j) < xpos(geometry_justify3, i, j)

    # new line should be lower (more negative) than the previous line
    assert ypos(geometry_left, 1, 10) < ypos(geometry_center, 0, 9)

    # For last char on last line
    i, j = 3, geometry_left._glyph_count - 1
    assert xpos(geometry_left, i, j) < xpos(geometry_center, i, j)
    assert xpos(geometry_center, i, j) < xpos(geometry_right, i, j)

    # justify will not justify the last line
    assert xpos(geometry_left, i, j) < xpos(geometry_justify1, i, j)
    assert xpos(geometry_left, i, j) == xpos(geometry_justify2, i, j)
    # unless you specify justify-all
    assert xpos(geometry_left, i, j) < xpos(geometry_justify3, i, j)
    assert xpos(geometry_right, i, j) < xpos(geometry_justify3, i, j)


def test_geometry_text_with_new_lines():
    geo1 = TextGeometry(
        text=["hello there"],
        text_align="left",
        anchor="top-left",
    )
    geo2 = TextGeometry(
        text=["hello\nthere"],
        text_align="left",
        anchor="top-left",
    )

    def ypos(geo, block_i, glyph_i):
        return geo.positions.data[block_i, 1] + geo.glyph_positions.data[glyph_i, 1]

    assert ypos(geo1, 0, 0) == ypos(geo2, 0, 0)
    assert ypos(geo1, 0, 9) > ypos(geo2, 0, 9)

    geo1.anchor = "bottom-left"
    geo2.anchor = "bottom-left"

    geo1._on_update_object()
    geo2._on_update_object()

    assert ypos(geo1, 0, 9) == ypos(geo2, 0, 9)
    assert ypos(geo1, 0, 0) < ypos(geo2, 0, 0)


def test_alignment_with_spaces():
    geo = TextGeometry(["hello world\n hello world\nhello world "])

    geo.text_align = "left"
    geo._on_update_object()

    i1 = 0
    i2 = 10
    i3 = 20
    assert geo.glyph_positions.data[i1, 0] < geo.glyph_positions.data[i2, 0]
    assert geo.glyph_positions.data[i1, 0] == geo.glyph_positions.data[i3, 0]

    geo.text_align = "right"
    geo._on_update_object()

    assert geo.glyph_positions.data[i1, 0] > geo.glyph_positions.data[i3, 0]
    assert geo.glyph_positions.data[i1, 0] == geo.glyph_positions.data[i2, 0]


def test_geometry_text_align_last():
    text = (
        "Lorem ipsum\n"
        "Bonjour World Olá\n"  # some text that isn't equal in line
        "pygfx\n"  # a line with exactly 1 word, definitely must not crash
        "last line"
    )
    text = [text]
    geometry_default = TextGeometry(
        text=text,
        text_align="left",
        anchor="top-left",
    )
    geometry_left = TextGeometry(
        text=text, text_align="left", anchor="top-left", text_align_last="left"
    )
    geometry_center = TextGeometry(
        text=text, text_align="left", anchor="top-left", text_align_last="center"
    )
    geometry_right = TextGeometry(
        text=text, text_align="left", anchor="top-left", text_align_last="right"
    )
    geometry_justify = TextGeometry(
        text=text, text_align="left", anchor="top-left", text_align_last="justify"
    )
    i = geometry_default._glyph_count - 1
    assert (
        geometry_default.glyph_positions.data[i, 0]
        == geometry_left.glyph_positions.data[i, 0]
    )
    assert (
        geometry_left.glyph_positions.data[i, 0]
        < geometry_center.glyph_positions.data[i, 0]
    )
    assert (
        geometry_center.glyph_positions.data[i, 0]
        < geometry_right.glyph_positions.data[i, 0]
    )
    # unless you specify justify-all
    assert (
        geometry_right.glyph_positions.data[i, 0]
        == geometry_justify.glyph_positions.data[i, 0]
    )


def test_text_geometry_leading_spaces():
    basic_text = TextGeometry(["hello"], anchor="top-left")
    with_1whitespace = TextGeometry(["\n hello"], anchor="top-left")
    with_3whitespace = TextGeometry([" \n   hello"], anchor="top-left")
    # The spaces should not be stripped at the front
    assert (
        basic_text.glyph_positions.data[0, 0]
        < with_1whitespace.glyph_positions.data[0, 0]
    )
    assert (
        with_1whitespace.glyph_positions.data[0, 0]
        < with_3whitespace.glyph_positions.data[0, 0]
    )


def test_check_speed():
    # On 29-11-2022 (with only very basic layout mechanics),
    # this takes about 8ms, which translates to 0.001 ms per glyph.
    #
    # To view the results with pytest run with
    #    pytest -s
    # or
    #    pytest --capture=no
    text = "HelloWorld"
    n_glyphs = len(text)
    t = TextGeometry(text=text)

    t0 = time.perf_counter()
    n_iter = 1000
    for _i in range(n_iter):
        t._dirty_blocks.add(0)
        t._on_update_object()
    dt = time.perf_counter() - t0
    print(
        f"\ngenerate_glyph: {n_iter * dt:0.1f} ms total",
        f"{1e6 * dt / (n_iter * n_glyphs):0.2f} us per glyph",
    )


if __name__ == "__main__":
    for ob in list(globals().values()):
        if callable(ob) and ob.__name__.startswith("test_"):
            print(f"{ob.__name__} ...")
            ob()
    print("done")
