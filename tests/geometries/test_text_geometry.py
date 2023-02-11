import time
from pygfx import TextGeometry, TextItem
from pytest import raises
import numpy as np


def test_text_geometry_text():
    # Let's try some special cases first

    # Must specify either text or markdown
    with raises(TypeError):
        TextGeometry(text="foo", markdown="foo")

    # Empty string - still has one item (whitespace)
    geo = TextGeometry(text="")
    geo.positions.nitems == 1
    geo.indices.nitems == 1
    geo.sizes.nitems == 1

    # Only a space
    geo = TextGeometry(" ")  # also test that text is a positional arg
    geo.positions.nitems == 1
    geo.indices.nitems == 1
    geo.sizes.nitems == 1

    # One char
    geo = TextGeometry(text="a")
    geo.positions.nitems == 1
    geo.indices.nitems == 1
    geo.sizes.nitems == 1

    # Two words with 3 chars in total
    geo = TextGeometry(text="a bc")
    geo.positions.nitems == 3
    geo.indices.nitems == 3
    geo.sizes.nitems == 3

    # Can set new text, buffers are recreated
    geo.set_text("foo bar")
    geo.positions.nitems == 6
    geo.indices.nitems == 6
    geo.sizes.nitems == 6

    # If setting smaller text, buffer size is oversized
    geo.set_text("x")
    geo.positions.nitems == 6
    geo.indices.nitems == 6
    geo.sizes.nitems == 6

    # Last parts are not used
    assert np.all(geo.sizes.data[1:] == 0)


def test_text_geometry_items():
    geo = TextGeometry()

    items = [TextItem("foo"), TextItem("baar")]
    geo.set_text_items(items)
    geo.positions.nitems == 7
    geo.indices.nitems == 7

    items = [TextItem("foo"), TextItem("baar"), TextItem("!")]
    geo.set_text_items(items)

    geo.positions.nitems == 8
    geo.indices.nitems == 8


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


def check_speed():
    t = TextGeometry(text="HelloWorld")

    t0 = time.perf_counter()
    for i in range(1000):
        t.apply_layout()
    dt = time.perf_counter() - t0
    print(
        f"generate_glyph: {1000*dt:0.1f} ms total",
        f"{1000*dt/(10000):0.3f} ms per glyph",
    )

    # On 29-11-2022 (with only very basic layour mechanics),
    # this takes about 8ms, which translates to 0.001 ms per glyph.


if __name__ == "__main__":
    for ob in list(globals().values()):
        if callable(ob) and ob.__name__.startswith("test_"):
            print(f"{ob.__name__} ...")
            ob()
    print("done")

    check_speed()
