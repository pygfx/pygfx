from pytest import raises
import numpy as np

from pygfx.utils.color import Color


class TColor(Color):
    def matches(self, r, g, b, a):
        eps = 0.501 / 255
        rgba = r, g, b, a
        return all(abs(v1 - v2) < eps for v1, v2 in zip(self.rgba, rgba))


def test_color_basics():

    c = Color(0.1, 0.2, 0.3)
    assert repr(c).startswith("<Color ")
    assert "0.10 0.20 0.30" in repr(c)

    d = Color(c)
    assert list(c) == list(d)


def test_color_tuples():

    # Test setting with values
    assert TColor(0, 0).matches(0, 0, 0, 0)
    assert TColor(1, 1).matches(1, 1, 1, 1)
    assert TColor(0.5, 0.8).matches(0.5, 0.5, 0.5, 0.8)
    assert TColor(0.1, 0.2, 0.3).matches(0.1, 0.2, 0.3, 1)
    assert TColor(0.1, 0.2, 0.3, 0.8).matches(0.1, 0.2, 0.3, 0.8)

    # Need at least two args to provide a color tuple
    with raises(ValueError):
        Color()
    with raises(ValueError):
        Color(0.0)

    # Now with real tuples
    assert TColor((0, 0)).matches(0, 0, 0, 0)
    assert TColor((1, 1)).matches(1, 1, 1, 1)
    assert TColor((0.5, 0.8)).matches(0.5, 0.5, 0.5, 0.8)
    assert TColor((0.1, 0.2, 0.3)).matches(0.1, 0.2, 0.3, 1)
    assert TColor((0.1, 0.2, 0.3, 0.8)).matches(0.1, 0.2, 0.3, 0.8)

    # Can also do a 1-element tuple then
    assert TColor((0.6,)).matches(0.6, 0.6, 0.6, 1)


def test_color_attr():

    c = Color(0.1, 0.2, 0.3, 0.8)

    assert c.rgb == c.rgba[:3]

    assert c.r == c.rgba[0]
    assert c.g == c.rgba[1]
    assert c.b == c.rgba[2]
    assert c.a == c.rgba[3]


def test_color_indexing():

    c = Color(0.1, 0.2, 0.3, 0.8)

    # Indexing
    assert c.r == c[0]
    assert c.g == c[1]
    assert c.b == c[2]
    assert c.a == c[3]

    # Iteration
    assert len(c) == 4  # This is *always* the case
    assert c.rgba == tuple(c)


def test_color_numpy():

    # Map an array to the color data
    c = Color(0.1, 0.2, 0.3, 0.8)
    a = np.array(c, copy=False)
    assert a.dtype == np.float32
    assert (a == c).all()

    # We cannot change the array
    assert not a.flags.writeable

    # But we can change the color, with a hack
    c._val[0] = 9
    assert a[0] == 9


def test_color_hex():

    # Hex -> tuple
    assert TColor("#000000").matches(0, 0, 0, 1)
    assert TColor("#ffffff").matches(1, 1, 1, 1)
    assert TColor("#7f7f7f").matches(0.5, 0.5, 0.5, 1)
    assert TColor("#7f7f7f10").matches(0.5, 0.5, 0.5, 16 / 255)

    # Tuple -> hex
    assert TColor(0, 1).hex == "#000000"
    assert TColor(0, 0.5, 1).hex == "#0080ff"
    assert TColor(1, 0.5, 0).hex == "#ff8000"
    assert TColor(0.1, 0.2, 0.3).hex == "#1a334d"

    # Tuple -> hexa
    assert TColor(0, 1).hexa == "#000000ff"
    assert TColor(0, 0.5, 1).hexa == "#0080ffff"
    assert TColor(1, 0.5, 0).hexa == "#ff8000ff"
    assert TColor(0.1, 0.2, 0.3).hexa == "#1a334dff"
    assert TColor(0.1, 0.2, 0.3, 0.5).hexa == "#1a334d80"

    # Roundtrip between hex and tuple to make sure the
    # values are stable and won't "jump"
    for v in [0.0, 0.1, 0.23, 1 / 7, 0.99, 1.0]:
        c = Color(v, v, v, 1)
        for i in range(10):
            c = TColor(c.hex)
            assert c.matches(v, v, v, 1)

    # Variations
    assert Color("#123").hexa == "#112233ff"
    assert Color("#1234").hexa == "#11223344"
    assert Color("#112233").hexa == "#112233ff"
    assert Color("#11223344").hexa == "#11223344"

    for x in ["#1", "#12", "#12345", "#1234567", "#123456789"]:
        with raises(ValueError):
            Color(x)


def test_color_css():

    assert Color("rgb(10, 20, 30)").hexa == "#0a141eff"
    assert Color("rgba(10, 20, 30, 0.5)").hexa == "#0a141e80"

    assert Color("#0a141eff").css == "rgb(10,20,30)"
    assert Color("#0a141e80").css == "rgba(10,20,30,0.502)"

    with raises(ValueError):
        Color("rgb(10, 20, 30, 40, 50)")
    with raises(ValueError):
        Color("rgb(10, 20)")


def test_color_min_max():

    assert Color(1.1, 1.2, 1.3).rgb == (1, 1, 1)
    assert Color(-0.1, -0.2, -0.3).rgb == (0, 0, 0)

    assert Color("rgb(260, 270, 280)").css == "rgb(255,255,255)"


def test_color_named():

    assert Color("red").hexa == "#ff0000ff"
    assert Color("y").hexa == "#ffff00ff"

    with raises(ValueError):
        Color("notacolorname")


if __name__ == "__main__":
    test_color_tuples()
    test_color_attr()
    test_color_indexing()
    test_color_numpy()
    test_color_hex()
    test_color_css()
    test_color_min_max()
