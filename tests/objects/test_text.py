import time

import numpy as np
import pytest

from pygfx import Text, MultiText, TextBlock


def test_text_basic():
    # Let's try some special cases first

    min_blocks = 8
    min_glyphs = 16

    # Can instantiate empty text
    to = Text()
    assert len(to._text_blocks) == 0
    assert to._glyph_count == 0
    assert to.geometry.positions.nitems == min_blocks

    # Empty string - still has one item (whitespace)
    to = Text(text="")
    assert len(to._text_blocks) == 0
    assert to._glyph_count == 0
    assert to.geometry.positions.nitems == min_blocks
    assert to.geometry.glyph_data.nitems == min_glyphs

    # Only a newline
    to = Text("\n")  # also test that text is a positional arg
    assert len(to._text_blocks) == 1
    assert to._glyph_count == 0
    assert to.geometry.positions.nitems == min_blocks
    assert to.geometry.glyph_data.nitems == min_glyphs

    # Only a space
    to = Text(" ")  # also test that text is a positional arg
    assert len(to._text_blocks) == 1
    assert to._glyph_count == 0
    assert to.geometry.positions.nitems == min_blocks
    assert to.geometry.glyph_data.nitems == min_glyphs

    # One char
    to = Text(text="a")
    assert len(to._text_blocks) == 1
    assert len(to._text_blocks[0]._text_items) == 1
    assert to._glyph_count == 1
    assert to.geometry.positions.nitems == min_blocks
    assert to.geometry.glyph_data.nitems == min_glyphs

    # Two words with 3 chars in total
    to = Text(text="a bc")
    assert len(to._text_blocks) == 1
    assert len(to._text_blocks[0]._text_items) == 2
    assert to._glyph_count == 3
    assert to.geometry.positions.nitems == min_blocks
    assert to.geometry.glyph_data.nitems == min_glyphs

    # Can set new text, buffers are recreated
    to.set_text("foo\nbar")
    assert len(to._text_blocks) == 2
    assert to._glyph_count == 6
    assert to.geometry.positions.nitems == min_blocks
    assert to.geometry.glyph_data.nitems == min_glyphs

    # Mo text
    to.set_text("foo bar spam eggs\naap noot mies")
    assert len(to._text_blocks) == 2
    assert to._glyph_count == 25
    assert to.geometry.positions.nitems == min_blocks
    assert to.geometry.glyph_data.nitems == 32

    # If setting smaller text, buffer size is oversized
    to.set_text("x")
    assert len(to._text_blocks) == 1
    assert to._glyph_count == 1
    assert to.geometry.positions.nitems == min_blocks
    assert to.geometry.glyph_data.nitems == 32

    # Sizes are used to invalidate unused slots
    assert np.all(to.geometry.glyph_data.data["size"][1:] == 0)


def test_text_blocks():
    to = MultiText()

    block0 = to.create_text_block()
    blocks19 = to.create_text_blocks(9)
    blocks = [block0, *blocks19]
    assert to.get_text_block_count() == 10
    for b in blocks:
        assert isinstance(b, TextBlock)

    # Set to same amount of blocks, no effect
    to.set_text_block_count(10)
    for i in range(10):
        assert blocks[i] is to.get_text_block(i)

    # Set to more blocks
    to.set_text_block_count(12)
    for i in range(10):
        assert blocks[i] is to.get_text_block(i)

    # Set to less blocks
    to.set_text_block_count(8)
    for i in range(8):
        assert blocks[i] is to.get_text_block(i)
    with pytest.raises(IndexError):
        to.get_text_block(8)

    # Set back to 10. The last two blocks are now different.
    to.set_text_block_count(10)
    for i in range(8):
        assert blocks[i] is to.get_text_block(i)
    for i in range(8, 10):
        assert blocks[i] is not to.get_text_block(i)


def test_text_blocks_buffer():
    class MyText(MultiText):
        allocation_count = 0

        def _allocate_block_buffers(self, n):
            super()._allocate_block_buffers(n)
            self.allocation_count += 1

    to = MyText()

    assert to.geometry.positions.nitems == 8
    assert to.allocation_count == 0

    # Note that the minimum is 8

    to.set_text_block_count(1)
    assert to.geometry.positions.nitems == 8
    assert to.allocation_count == 0

    to.set_text_block_count(1)
    assert to.geometry.positions.nitems == 8
    assert to.allocation_count == 0

    to.set_text_block_count(2)
    assert to.geometry.positions.nitems == 8
    assert to.allocation_count == 0

    to.set_text_block_count(8)
    assert to.geometry.positions.nitems == 8
    assert to.allocation_count == 0

    # Now we bump

    to.set_text_block_count(9)
    assert to.geometry.positions.nitems == 16
    assert to.allocation_count == 1

    to.set_text_block_count(9)
    assert to.geometry.positions.nitems == 16
    assert to.allocation_count == 1

    to.set_text_block_count(16)
    assert to.geometry.positions.nitems == 16
    assert to.allocation_count == 1

    to.set_text_block_count(17)
    assert to.geometry.positions.nitems == 32
    assert to.allocation_count == 2

    to.set_text_block_count(33)
    assert to.geometry.positions.nitems == 64

    # Don't scale back too soon
    to.set_text_block_count(32)
    assert to.geometry.positions.nitems == 64

    # Don't scale back too soon
    to.set_text_block_count(16)
    assert to.geometry.positions.nitems == 64

    # Still not
    to.set_text_block_count(8)
    assert to.geometry.positions.nitems == 8


def test_text_blocks_reuse():
    to = Text()

    to.set_text("foo\nbar\nspam")
    blocks1 = to._text_blocks.copy()
    assert len(blocks1) == 3

    to.set_text("aap\nnoot\nmies")
    blocks2 = to._text_blocks.copy()
    assert len(blocks2) == 3

    for i in range(3):
        assert blocks1[i] is blocks2[i]


def test_text_items_reuse():
    to = Text()

    to.set_text("foo bar spam")
    blocks1 = to._text_blocks.copy()
    items1 = to._text_blocks[0]._text_items.copy()
    assert len(blocks1) == 1
    assert len(items1) == 3

    to.set_text("aap noot mies")
    blocks2 = to._text_blocks.copy()
    items2 = to._text_blocks[0]._text_items.copy()
    assert len(blocks2) == 1
    assert len(items2) == 3

    assert blocks1[0] is blocks2[0]
    for i in range(3):
        assert items1[i] is items2[i]


def test_text_whitespace():
    # Tests how whitespace is handled during itemization and layout.

    # With left anchor

    t1 = Text(text="abc def", anchor="baseline-left")
    t2 = Text(text="abc   def", anchor="baseline-left")
    t3 = Text(text=" abc def", anchor="baseline-left")
    t4 = Text(text="abc def ", anchor="baseline-left")

    x1 = t1.geometry.glyph_data.data["pos"][t1._glyph_count - 1, 0]
    x2 = t2.geometry.glyph_data.data["pos"][t2._glyph_count - 1, 0]
    x3 = t3.geometry.glyph_data.data["pos"][t3._glyph_count - 1, 0]
    x4 = t4.geometry.glyph_data.data["pos"][t4._glyph_count - 1, 0]

    assert x2 > x1  # Extra spaces in between words are preserved
    assert x3 > x1  # Extra space at the start is also preserved
    assert x4 == x1  # Extra space at the end is dropped

    # With right anchor

    t1 = Text(text="abc def", anchor="baseline-right")
    t2 = Text(text="abc   def", anchor="baseline-right")
    t3 = Text(text=" abc def", anchor="baseline-right")
    t4 = Text(text="abc def ", anchor="baseline-right")

    x1 = t1.geometry.glyph_data.data["pos"][0, 0]
    x2 = t2.geometry.glyph_data.data["pos"][0, 0]
    x3 = t3.geometry.glyph_data.data["pos"][0, 0]
    x4 = t4.geometry.glyph_data.data["pos"][0, 0]

    x1 += t1.geometry.positions.data[0, 0]
    x2 += t2.geometry.positions.data[0, 0]
    x3 += t3.geometry.positions.data[0, 0]
    x4 += t4.geometry.positions.data[0, 0]

    assert x2 < x1  # Extra spaces in between words are preserved
    assert x3 == x1  # Extra space at the start is dropped
    assert x4 < x1  # Extra space at the end is preserved


def test_text_markdown():
    # The same text, but the 2nd one is formatted. Should still
    # result in same number of glyphs and equally positioned.
    t1 = Text(text="abc def")
    t2 = Text(markdown="*abc* **def**")

    assert np.all(t1.geometry.positions.data == t2.geometry.positions.data)


def test_text_anchor():
    t = Text("")

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


def test_text_direction_ttb():
    t1 = Text("abc def", direction="ltr")
    x1, y1 = (
        t1.geometry.glyph_data.data["pos"][:, 0],
        t1.geometry.glyph_data.data["pos"][:, 1],
    )
    assert (x1.max() - x1.min()) > (y1.max() - y1.min())

    t2 = Text("abc def", direction="ttb")
    x2, y2 = (
        t2.geometry.glyph_data.data["pos"][:, 0],
        t2.geometry.glyph_data.data["pos"][:, 1],
    )
    assert (x2.max() - x2.min()) < (y2.max() - y2.min())


def test_text_direction_rtl_1():
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

    t = Text(text=text)
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


def test_text_direction_rt_2():
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

    t = Text(text=text)
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

    def xpos(to, block_i, glyph_i):
        return (
            to.geometry.positions.data[block_i, 0]
            + to.geometry.glyph_data.data["pos"][glyph_i, 0]
        )

    def ypos(to, block_i, glyph_i):
        return (
            to.geometry.positions.data[block_i, 1]
            + to.geometry.glyph_data.data["pos"][glyph_i, 1]
        )

    # Setting text to list of text, to make sure to use a single block.
    text_left = Text(
        text=text,
        text_align="left",
        anchor="top-left",
    )
    text_center = Text(
        text=text,
        text_align="center",
        anchor="top-left",
    )
    text_right = Text(
        text=text,
        text_align="right",
        anchor="top-left",
    )
    text_justify1 = Text(
        text=text,
        text_align="justify",
        anchor="top-left",
    )
    text_justify2 = Text(
        text=text,
        text_align="justify",
        anchor="top-left",
        max_width=300,
    )
    text_justify3 = Text(
        text=text,
        text_align="justify-all",
        anchor="top-left",
        max_width=300,
    )

    # Check first char
    j = 0
    assert xpos(text_left, 0, j) < xpos(text_center, 0, j)
    assert xpos(text_center, 0, j) < xpos(text_right, 0, j)

    assert xpos(text_left, 0, j) == xpos(text_justify1, 0, j)
    assert xpos(text_left, 0, j) == xpos(text_justify2, 0, j)

    # Check last char of first line
    j = 9
    assert xpos(text_left, 0, j) < xpos(text_center, 0, j)
    assert xpos(text_center, 0, j) < xpos(text_right, 0, j)

    assert xpos(text_right, 0, j) == xpos(text_justify1, 0, j)
    assert xpos(text_right, 0, j) < xpos(text_justify2, 0, j)
    assert xpos(text_right, 0, j) < xpos(text_justify3, 0, j)

    # new line should be lower (more negative) than the previous line
    assert ypos(text_left, 0, 10) < ypos(text_center, 0, 9)

    # For last char on last line
    j = text_left._glyph_count - 1
    assert xpos(text_left, 0, j) < xpos(text_center, 0, j)
    assert xpos(text_center, 0, j) < xpos(text_right, 0, j)

    # justify will not justify the last line
    assert xpos(text_left, 0, j) == xpos(text_justify1, 0, j)
    assert xpos(text_left, 0, j) == xpos(text_justify2, 0, j)
    # unless you specify justify-all
    assert xpos(text_left, 0, j) < xpos(text_justify3, 0, j)
    assert xpos(text_right, 0, j) < xpos(text_justify3, 0, j)


def test_geometry_text_align_2():
    ref_text = (
        "Lorem ipsum\n"
        "Bonjour World Olá\n"  # some text that isn't equal in line
        "pygfx\n"  # a line with exactly 1 word, definitely must not crash
        "last line"
    )

    text = ref_text

    def xpos(to, block_i, glyph_i):
        return (
            to.geometry.positions.data[block_i, 0]
            + to.geometry.glyph_data.data["pos"][glyph_i, 0]
        )

    def ypos(to, block_i, glyph_i):
        return (
            to.geometry.positions.data[block_i, 1]
            + to.geometry.glyph_data.data["pos"][glyph_i, 1]
        )

    # Setting text to list of text, to make sure to use a single block.
    text_left = Text(
        text=text,
        text_align="left",
        anchor="top-left",
    )
    text_center = Text(
        text=text,
        text_align="center",
        anchor="top-left",
    )
    text_right = Text(
        text=text,
        text_align="right",
        anchor="top-left",
    )
    text_justify2 = Text(
        text=text,
        text_align="justify",
        anchor="top-left",
        max_width=300,
    )
    text_justify1 = Text(
        text=text,
        text_align="justify",
        anchor="top-left",
    )
    text_justify2 = Text(
        text=text,
        text_align="justify",
        anchor="top-left",
        max_width=300,
    )
    text_justify3 = Text(
        text=text,
        text_align="justify-all",
        anchor="top-left",
        max_width=300,
    )

    # Check first char
    i, j = 0, 0
    assert xpos(text_left, i, j) < xpos(text_center, i, j)
    assert xpos(text_center, i, j) < xpos(text_right, i, j)

    # Justify with max-width works
    assert xpos(text_left, i, j) == xpos(text_justify2, i, j)
    # But without max-width does the same as 'center'
    assert xpos(text_left, i, j) < xpos(text_justify1, i, j)

    # Check last char of first line
    i, j = 0, 9
    assert xpos(text_left, i, j) < xpos(text_center, i, j)
    assert xpos(text_center, i, j) < xpos(text_right, i, j)

    assert xpos(text_right, i, j) > xpos(text_justify1, i, j)
    assert xpos(text_right, i, j) > xpos(text_justify2, i, j)
    assert xpos(text_right, i, j) < xpos(text_justify3, i, j)

    # new line should be lower (more negative) than the previous line
    assert ypos(text_left, 1, 10) < ypos(text_center, 0, 9)

    # For last char on last line
    i, j = 3, text_left._glyph_count - 1
    assert xpos(text_left, i, j) < xpos(text_center, i, j)
    assert xpos(text_center, i, j) < xpos(text_right, i, j)

    # justify will not justify the last line
    assert xpos(text_left, i, j) < xpos(text_justify1, i, j)
    assert xpos(text_left, i, j) == xpos(text_justify2, i, j)
    # unless you specify justify-all
    assert xpos(text_left, i, j) < xpos(text_justify3, i, j)
    assert xpos(text_right, i, j) < xpos(text_justify3, i, j)


def test_geometry_text_with_new_lines():
    text1 = Text(
        text=["hello there"],
        text_align="left",
        anchor="top-left",
    )
    text2 = Text(
        text=["hello\nthere"],
        text_align="left",
        anchor="top-left",
    )

    def ypos(to, block_i, glyph_i):
        return (
            to.geometry.positions.data[block_i, 1]
            + to.geometry.glyph_data.data["pos"][glyph_i, 1]
        )

    assert ypos(text1, 0, 0) == ypos(text2, 0, 0)
    assert ypos(text1, 0, 9) > ypos(text2, 0, 9)

    text1.anchor = "bottom-left"
    text2.anchor = "bottom-left"

    text1._update_object()
    text2._update_object()

    assert ypos(text1, 0, 9) == ypos(text2, 0, 9)
    assert ypos(text1, 0, 0) < ypos(text2, 0, 0)


def test_alignment_with_spaces():
    to = Text(["hello world\n hello world\nhello world "])

    to.text_align = "left"
    to._update_object()

    i1 = 0
    i2 = 10
    i3 = 20
    assert (
        to.geometry.glyph_data.data["pos"][i1, 0]
        < to.geometry.glyph_data.data["pos"][i2, 0]
    )
    assert (
        to.geometry.glyph_data.data["pos"][i1, 0]
        == to.geometry.glyph_data.data["pos"][i3, 0]
    )

    to.text_align = "right"
    to._update_object()

    assert (
        to.geometry.glyph_data.data["pos"][i1, 0]
        > to.geometry.glyph_data.data["pos"][i3, 0]
    )
    assert (
        to.geometry.glyph_data.data["pos"][i1, 0]
        == to.geometry.glyph_data.data["pos"][i2, 0]
    )


def test_geometry_text_align_last():
    text = (
        "Lorem ipsum\n"
        "Bonjour World Olá\n"  # some text that isn't equal in line
        "pygfx\n"  # a line with exactly 1 word, definitely must not crash
        "last line"
    )
    text = [text]
    text_default = Text(
        text=text,
        text_align="left",
        anchor="top-left",
    )
    text_left = Text(
        text=text, text_align="left", anchor="top-left", text_align_last="left"
    )
    text_center = Text(
        text=text, text_align="left", anchor="top-left", text_align_last="center"
    )
    text_right = Text(
        text=text, text_align="left", anchor="top-left", text_align_last="right"
    )
    text_justify = Text(
        text=text, text_align="left", anchor="top-left", text_align_last="justify"
    )
    i = text_default._glyph_count - 1
    assert (
        text_default.geometry.glyph_data.data["pos"][i, 0]
        == text_left.geometry.glyph_data.data["pos"][i, 0]
    )
    assert (
        text_left.geometry.glyph_data.data["pos"][i, 0]
        < text_center.geometry.glyph_data.data["pos"][i, 0]
    )
    assert (
        text_center.geometry.glyph_data.data["pos"][i, 0]
        < text_right.geometry.glyph_data.data["pos"][i, 0]
    )
    # unless you specify justify-all
    assert (
        text_right.geometry.glyph_data.data["pos"][i, 0]
        == text_justify.geometry.glyph_data.data["pos"][i, 0]
    )


def test_text_leading_spaces():
    basic_text = Text(["hello"], anchor="top-left")
    with_1whitespace = Text(["\n hello"], anchor="top-left")
    with_3whitespace = Text([" \n   hello"], anchor="top-left")
    # The spaces should not be stripped at the front
    assert (
        basic_text.geometry.glyph_data.data["pos"][0, 0]
        < with_1whitespace.geometry.glyph_data.data["pos"][0, 0]
    )
    assert (
        with_1whitespace.geometry.glyph_data.data["pos"][0, 0]
        < with_3whitespace.geometry.glyph_data.data["pos"][0, 0]
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
    t = Text(text=text)

    t0 = time.perf_counter()
    n_iter = 1000
    for _i in range(n_iter):
        t._dirty_blocks.add(0)
        t._update_object()
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
