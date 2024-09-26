import time

import numpy as np
from pytest import raises

from pygfx import TextGeometry, TextItem


def test_text_geometry_text():
    # Let's try some special cases first

    # Must specify either text or markdown
    with raises(TypeError):
        TextGeometry(text="foo", markdown="foo")

    # Empty string - still has one item (whitespace)
    geo = TextGeometry(text="")
    assert geo.positions.nitems == 1
    assert geo.indices.nitems == 1
    assert geo.sizes.nitems == 1

    # Only a space
    geo = TextGeometry(" ")  # also test that text is a positional arg
    assert geo.positions.nitems == 1
    assert geo.indices.nitems == 1
    assert geo.sizes.nitems == 1

    # One char
    geo = TextGeometry(text="a")
    assert geo.positions.nitems == 1
    assert geo.indices.nitems == 1
    assert geo.sizes.nitems == 1

    # Two words with 3 chars in total
    geo = TextGeometry(text="a bc")
    assert geo.positions.nitems == 3
    assert geo.indices.nitems == 3
    assert geo.sizes.nitems == 3

    # Can set new text, buffers are recreated
    geo.set_text("foo bar")
    assert geo.positions.nitems == 6
    assert geo.indices.nitems == 6
    assert geo.sizes.nitems == 6

    # If setting smaller text, buffer size is oversized
    geo.set_text("x")
    assert geo.positions.nitems == 6
    assert geo.indices.nitems == 6
    assert geo.sizes.nitems == 6

    # Last parts are not used
    assert np.all(geo.sizes.data[1:] == 0)


def test_text_geometry_items():
    geo = TextGeometry()

    items = [TextItem("foo"), TextItem("baar")]
    geo.set_text_items(items)
    assert geo.positions.nitems == 7
    assert geo.indices.nitems == 7

    items = [TextItem("foo"), TextItem("baar"), TextItem("!")]
    geo.set_text_items(items)

    assert geo.positions.nitems == 8
    assert geo.indices.nitems == 8


def test_text_geometry_whitespace():
    # Tests how whitespace is handled during itemization and layout.

    # With left anchor

    t1 = TextGeometry(text="abc def", anchor="baselineleft")
    t2 = TextGeometry(text="abc   def", anchor="baselineleft")
    t3 = TextGeometry(text=" abc def", anchor="baselineleft")
    t4 = TextGeometry(text="abc def ", anchor="baselineleft")

    x1 = t1.positions.data[-1, 0]
    x2 = t2.positions.data[-1, 0]
    x3 = t3.positions.data[-1, 0]
    x4 = t4.positions.data[-1, 0]

    assert x2 > x1  # Extra spaces in between words are preserved
    assert x3 > x1  # Extra space at the start is also preserved
    assert x4 == x1  # Extra space at the end is dropped

    # With right anchor

    t1 = TextGeometry(text="abc def", anchor="baselineright")
    t2 = TextGeometry(text="abc   def", anchor="baselineright")
    t3 = TextGeometry(text=" abc def", anchor="baselineright")
    t4 = TextGeometry(text="abc def ", anchor="baselineright")

    x1 = t1.positions.data[0, 0]
    x2 = t2.positions.data[0, 0]
    x3 = t3.positions.data[0, 0]
    x4 = t4.positions.data[0, 0]

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

    # Aliases for middle/center
    t.anchor = "center-middle"
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

    # No dash with alias allows shortcut
    t.anchor = "center"
    assert t.anchor == "middle-center"
    t.anchor = "middle"
    assert t.anchor == "middle-center"

    # Default
    t.anchor = None
    assert t.anchor == "middle-center"


def test_text_geometry_direction_ttb():
    t1 = TextGeometry("abc def", direction="lrt")
    t2 = TextGeometry("abc def", direction="ttb")

    assert t1.positions.data[:, 0].max() > t1.positions.data[:, 1].max()
    assert t2.positions.data[:, 1].max() > t2.positions.data[:, 0].max()


def test_text_geometry_direction_rtl():
    # This is a very adversary/weird text: two words of Arabic in the
    # middle of Latin. In a text editor it looks like the first Arabic
    # word is longer than the second, but in fact the word after "foo"
    # is the shorter one. Try stepping your cursor through the string.
    # Pygfx should re-order the items so they appear as they should
    # again. That's what we test here.
    text = "foo عدد النبات baaaar"

    t = TextGeometry(text=text)
    items = t._glyph_items
    assert len(items) == 4

    assert items[0].direction == "ltr"
    assert items[1].direction == "rtl"
    assert items[2].direction == "rtl"
    assert items[3].direction == "ltr"

    # Use the lengths of the words to identify them
    assert items[3].extent > items[0].extent
    assert items[1].extent > items[2].extent


def test_geometry_text_align():
    text = (
        "Lorem ipsum\n"
        "Bonjour World Olá\n"  # some text that isn't equal in line
        "pygfx\n"  # a line with exactly 1 word, definitely must not crash
        "last line"
    )
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
    geometry_justify = TextGeometry(
        text=text,
        text_align="justify",
        anchor="top-left",
    )
    geometry_justify_all = TextGeometry(
        text=text,
        text_align="justify-all",
        anchor="top-left",
    )

    assert geometry_left.positions.data[0, 0] < geometry_center.positions.data[0, 0]
    assert geometry_center.positions.data[0, 0] < geometry_right.positions.data[0, 0]
    assert geometry_left.positions.data[0, 0] == geometry_justify.positions.data[0, 0]

    # Index 9 is the right most character on the first line
    assert geometry_right.positions.data[9, 0] == geometry_justify.positions.data[9, 0]
    assert geometry_left.positions.data[9, 0] < geometry_center.positions.data[9, 0]
    assert geometry_center.positions.data[9, 0] < geometry_right.positions.data[9, 0]

    # new line should be lower than the previous line
    assert geometry_left.positions.data[10, 1] < geometry_center.positions.data[0, 1]

    assert geometry_left.positions.data[-1, 0] < geometry_center.positions.data[-1, 0]
    assert geometry_center.positions.data[-1, 0] < geometry_right.positions.data[-1, 0]
    # justify will not justify the last line
    assert geometry_left.positions.data[-1, 0] == geometry_justify.positions.data[-1, 0]
    # unless you specify justify-all
    assert (
        geometry_right.positions.data[-1, 0]
        == geometry_justify_all.positions.data[-1, 0]
    )


def test_geometry_text_with_new_lines():
    geometry_left = TextGeometry(
        text="hello",
        text_align="left",
        anchor="top-left",
    )
    geometry_with_newline = TextGeometry(
        text="\nhello\n",
        text_align="left",
        anchor="top-left",
    )
    assert (
        geometry_with_newline.positions.data[0, 1] < geometry_left.positions.data[0, 1]
    )

    geometry_with_newline.anchor = "bottom-left"
    geometry_left.anchor = "bottom-left"
    assert (
        geometry_with_newline.positions.data[0, 1] > geometry_left.positions.data[0, 1]
    )


def test_alignment_with_spaces():
    text = TextGeometry("hello world")
    text_leading_spaces = TextGeometry(" hello world")
    text_trailing_spaces = TextGeometry("hello world ")

    text.anchor = "bottom-left"
    text_leading_spaces.anchor = "bottom-left"
    text_trailing_spaces.anchor = "bottom-left"
    assert text.positions.data[0, 0] < text_leading_spaces.positions.data[0, 0]
    assert text.positions.data[0, 0] == text_trailing_spaces.positions.data[0, 0]

    text.anchor = "bottom-right"
    text_leading_spaces.anchor = "bottom-right"
    text_trailing_spaces.anchor = "bottom-right"
    assert text.positions.data[-1, 0] > text_trailing_spaces.positions.data[-1, 0]
    assert text.positions.data[-1, 0] == text_leading_spaces.positions.data[-1, 0]


def test_geometry_text_align_last():
    text = (
        "Lorem ipsum\n"
        "Bonjour World Olá\n"  # some text that isn't equal in line
        "pygfx\n"  # a line with exactly 1 word, definitely must not crash
        "last line"
    )
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
    assert geometry_default.positions.data[-1, 0] == geometry_left.positions.data[-1, 0]
    assert geometry_left.positions.data[-1, 0] < geometry_center.positions.data[-1, 0]
    assert geometry_center.positions.data[-1, 0] < geometry_right.positions.data[-1, 0]
    # unless you specify justify-all
    assert (
        geometry_right.positions.data[-1, 0] == geometry_justify.positions.data[-1, 0]
    )


def test_text_geometry_leading_spaces():
    basic_text = TextGeometry("hello", anchor="top-left")
    with_1whitespace = TextGeometry("\n hello", anchor="top-left")
    with_3whitespace = TextGeometry(" \n   hello", anchor="top-left")
    # The spaces should not be stripped at the front
    assert basic_text.positions.data[0, 0] < with_1whitespace.positions.data[0, 0]
    assert with_1whitespace.positions.data[0, 0] < with_3whitespace.positions.data[0, 0]


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
        t.apply_layout()
    dt = time.perf_counter() - t0
    print(
        f"\ngenerate_glyph: {n_iter * dt:0.1f} ms total",
        f"{1E6 * dt / (n_iter * n_glyphs):0.2f} us per glyph",
    )


if __name__ == "__main__":
    for ob in list(globals().values()):
        if callable(ob) and ob.__name__.startswith("test_"):
            print(f"{ob.__name__} ...")
            ob()
    print("done")
