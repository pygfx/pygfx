from colorsys import ONE_THIRD
from pytest import raises
import numpy as np

from pygfx.utils.color import Color, NAMED_COLORS


class TColor(Color):
    def matches(self, r, g, b, a):
        eps = 0.501 / 255
        rgba = r, g, b, a
        return all(abs(v1 - v2) < eps for v1, v2 in zip(self.rgba, rgba))


def test_color_basics():
    c = Color(0.1, 0.2, 0.3)
    assert repr(c) == "Color(0.1, 0.2, 0.3, 1.0)"
    c = Color(0.012345, 0.2, 0.3, 0.8)
    assert repr(c) == "Color(0.0123, 0.2, 0.3, 0.8)"

    d = Color(c)
    assert list(c) == list(d)


def test_color_tuples():
    # Test setting with values
    assert TColor(0).matches(0, 0, 0, 1)
    assert TColor(1).matches(1, 1, 1, 1)
    assert TColor(0, 0).matches(0, 0, 0, 0)
    assert TColor(1, 1).matches(1, 1, 1, 1)
    assert TColor(0.5, 0.8).matches(0.5, 0.5, 0.5, 0.8)
    assert TColor(0.1, 0.2, 0.3).matches(0.1, 0.2, 0.3, 1)
    assert TColor(0.1, 0.2, 0.3, 0.8).matches(0.1, 0.2, 0.3, 0.8)

    # Need at least two args to provide a color tuple
    with raises(ValueError):
        Color()

    # Now with real tuples
    assert TColor((0, 0)).matches(0, 0, 0, 0)
    assert TColor((1, 1)).matches(1, 1, 1, 1)
    assert TColor((0.5, 0.8)).matches(0.5, 0.5, 0.5, 0.8)
    assert TColor((0.1, 0.2, 0.3)).matches(0.1, 0.2, 0.3, 1)
    assert TColor((0.1, 0.2, 0.3, 0.8)).matches(0.1, 0.2, 0.3, 0.8)

    # Can also do a 1-element tuple then
    assert TColor((0.6,)).matches(0.6, 0.6, 0.6, 1)


def test_color_iterable():
    # Accepts any kind of iterable, like tuples and lists
    assert TColor((0.1, 0.2, 0.3)).matches(0.1, 0.2, 0.3, 1)
    assert TColor([0.1, 0.2, 0.3]).matches(0.1, 0.2, 0.3, 1)

    # Like arrays
    a = np.array([0.1, 0.2, 0.3])
    assert TColor(a).matches(0.1, 0.2, 0.3, 1)

    # And generators
    def get_color():
        yield 0.1
        yield 0.2
        yield 0.3

    assert TColor(get_color()).matches(0.1, 0.2, 0.3, 1)

    # And like a Color object ;)
    c = Color((0.1, 0.2, 0.3))
    assert TColor(c).matches(0.1, 0.2, 0.3, 1)

    # Because its iterable
    assert len(c) == 4  # This is *always* 4
    assert c.rgba == tuple(c)

    # Not iterable
    with raises(TypeError):
        Color(str)
    # Too short
    with raises(ValueError):
        Color([])
    # Too long
    with raises(ValueError):
        Color([0.1, 0.2, 0.3, 0.4, 0.5])


def test_color_attr():
    c = Color(0.1, 0.2, 0.3, 0.8)

    assert c.rgb == c.rgba[:3]

    assert c.r == c.rgba[0]
    assert c.g == c.rgba[1]
    assert c.b == c.rgba[2]
    assert c.a == c.rgba[3]

    assert 0.18 < c.gray < 0.22
    assert 0.3999 < Color(0.4, 0.4, 0.4, 0.8).gray < 0.400001


def test_color_indexing():
    c = Color(0.1, 0.2, 0.3, 0.8)

    assert c.r == c[0]
    assert c.g == c[1]
    assert c.b == c[2]
    assert c.a == c[3]


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
        for _i in range(10):
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
    assert TColor("rgb(10%, 20%, 30%)").matches(0.1, 0.2, 0.3, 1)
    assert TColor("rgba(10%, 20%, 30%, 0.5)").matches(0.1, 0.2, 0.3, 0.5)
    assert TColor("rgba(10%, 20%, 30%, 50%)").matches(0.1, 0.2, 0.3, 0.5)

    assert Color("#0a141eff").css == "rgb(10,20,30)"
    assert Color("#0a141e80").css == "rgba(10,20,30,0.502)"

    with raises(ValueError):
        Color("rgb(10, 20, 30, 40, 50)")
    with raises(ValueError):
        Color("rgb(10, 20)")

    assert Color("hsl(120deg, 100%, 50%)").hexa == "#00ff00ff"
    assert Color("hsla(120deg, 100%, 50%, 50%)").hexa == "#00ff0080"

    assert Color("hsv(120deg, 100%, 50%)").hexa == "#008000ff"
    assert Color("hsva(120deg, 100%, 50%, 50%)").hexa == "#00800080"

    assert Color("hsluv(120deg, 50%, 50%)").hexa == "#5e8052ff"
    assert Color("hsluva(120deg, 50%, 50%, 50%)").hexa == "#5e805280"

    with raises(ValueError):
        Color("hsl(120deg, 100%)")
    with raises(ValueError):
        Color("hsv(120deg, 100%)")
    with raises(ValueError):
        Color("hsluv(120deg, 100%)")


def test_color_min_max():
    assert np.allclose(Color(1.1, 1.2, 1.3).rgb, (1.1, 1.2, 1.3))
    assert Color(1.1, 1.2, 1.3).clip().rgb == (1, 1, 1)
    assert Color(1.1, 1.2, 1.3).hex == "#ffffff"
    assert Color(1.1, 1.2, 1.3, 1.4).a == 1

    assert np.allclose(Color(-0.1, -0.2, -0.3).rgb, (-0.1, -0.2, -0.3))
    assert Color(-0.1, -0.2, -0.3).clip().rgb == (0, 0, 0)
    assert Color(-0.1, -0.2, -0.3).hex == "#000000"
    assert Color(-0.1, -0.2, -0.3, -0.4).a == 0

    assert Color("rgb(260, 270, 280)").css == "rgb(260,270,280)"
    assert Color("rgba(260, 270, 280, 2)").css == "rgb(260,270,280)"
    assert Color("rgba(260, 270, 280, 2)").a == 1
    assert Color("rgb(260, 270, 280)").clip().css == "rgb(255,255,255)"


def test_color_named():
    assert Color("red").hexa == "#ff0000ff"
    assert Color("y").hexa == "#ffff00ff"

    with raises(ValueError):
        Color("notacolorname")

    # Make sure that all named colors can be consumed
    for key in NAMED_COLORS:
        Color(key)


def test_color_compare():
    c1 = Color("#f00")
    c2 = Color("#ff0000")
    c3 = Color("#333")

    assert c1 == c2
    assert c2 == c1
    assert c1 != c3
    assert c3 != c2

    assert c1 == "red"
    assert c3 == 0.2
    assert "#f00" == c1

    assert c3 != "#f00"
    assert "#f00" != c3


def test_color_combine():
    c = Color("rgb(20, 30, 200)") * 2
    assert c.css == "rgb(40,60,400)"
    c = Color("rgb(20, 30, 200)") / 2
    assert c.css == "rgb(10,15,100)"

    c = Color("#234") + Color("#333")
    assert c.hex == "#556677"

    c = Color("rgba(1,2,3,0.5)") + Color("rgb(10,20,30)")
    assert c.css == "rgba(11,22,33,0.500)"


def test_color_colorspaces():
    assert Color.from_physical(0.0, 0.5, 1.0).hex == "#00bcff"
    assert np.allclose(
        Color.from_physical(0.0, 0.5, 1.0).to_physical(), (0.0, 0.5, 1.0)
    )

    assert Color.from_hsv(ONE_THIRD, 1, 0.5).hex == "#008000"
    assert np.allclose(Color.from_hsv(ONE_THIRD, 1, 0.5).to_hsv(), (ONE_THIRD, 1, 0.5))

    assert Color.from_hsl(ONE_THIRD, 1, 0.5).hex == "#00ff00"
    assert np.allclose(Color.from_hsl(ONE_THIRD, 1, 0.5).to_hsl(), (ONE_THIRD, 1, 0.5))

    assert Color.from_hsluv(ONE_THIRD, 0.5, 0.5).hexa == "#5e8052ff"
    assert np.allclose(
        Color.from_hsluv(ONE_THIRD, 0.5, 0.5).to_hsluv(), (ONE_THIRD, 0.5, 0.5)
    )

    assert Color.from_hsv(ONE_THIRD, 1.0, 0.5).hexa == "#008000ff"
    assert np.allclose(
        Color.from_hsv(ONE_THIRD, 1.0, 0.5).to_hsv(), (ONE_THIRD, 1.0, 0.5)
    )

    assert Color.from_hsl(ONE_THIRD, 1.0, 0.5).hexa == "#00ff00ff"
    assert np.allclose(
        Color.from_hsl(ONE_THIRD, 1.0, 0.5).to_hsl(), (ONE_THIRD, 1.0, 0.5)
    )


def test_color_lerp_lighter_darker():
    green = Color("#00ff00")
    red = Color("#ff0000")

    # half way between green and red
    assert np.allclose(green.lerp(red, 0.5).rgb, (0.5, 0.5, 0))
    # green
    assert np.allclose(green.lerp(red, 0.0).rgb, (0.0, 1.0, 0))
    # red
    assert np.allclose(green.lerp(red, 1.0).rgb, (1.0, 0.0, 0.0))

    assert np.allclose(green.lerp_in_hue(red, 0.0, "hsl").to_hsl(), green.to_hsl())
    assert np.allclose(green.lerp_in_hue(red, 1.0, "hsl").to_hsl(), red.to_hsl())

    assert np.allclose(green.lerp_in_hue(red, 0.0, "hsv").to_hsv(), green.to_hsv())
    assert np.allclose(green.lerp_in_hue(red, 1.0, "hsv").to_hsv(), red.to_hsv())

    assert np.allclose(
        green.lerp_in_hue(red, 0.0, "hsluv").to_hsluv(), green.to_hsluv()
    )
    assert np.allclose(green.lerp_in_hue(red, 1.0, "hsluv").to_hsluv(), red.to_hsluv())

    # Test lighter() and darker()
    assert np.allclose(
        green.lighter(0.5, "hsl").to_hsl()[2],
        green.to_hsl()[2] + (1 - green.to_hsl()[2]) * 0.5,
    )
    assert np.allclose(green.darker(0.5, "hsl").to_hsl()[2], green.to_hsl()[2] * 0.5)

    assert np.allclose(
        green.lighter(0.5, "hsluv").to_hsluv()[2],
        green.to_hsluv()[2] + (1 - green.to_hsluv()[2]) * 0.5,
    )
    assert np.allclose(
        green.darker(0.5, "hsluv").to_hsluv()[2], green.to_hsluv()[2] * 0.5
    )

    assert np.allclose(
        green.lighter(0.5, "hsv").to_hsv()[2],
        green.to_hsv()[2] + (1 - green.to_hsv()[2]) * 0.5,
    )
    assert np.allclose(green.darker(0.5, "hsv").to_hsv()[2], green.to_hsv()[2] * 0.5)


if __name__ == "__main__":
    test_color_basics()
    test_color_tuples()
    test_color_iterable()
    test_color_attr()
    test_color_indexing()
    test_color_numpy()
    test_color_hex()
    test_color_css()
    test_color_min_max()
    test_color_named()
    test_color_compare()
    test_color_combine()
    test_color_colorspaces()
