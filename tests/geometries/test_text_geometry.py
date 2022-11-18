from pygfx import TextGeometry
from pytest import raises
import numpy as np


def test_text_geometry1():

    # Let's try some special cases first

    # Must specify either text or markdown
    with raises(TypeError):
        TextGeometry()
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
    assert np.all(geo.positions.data[1:] == 0)


if __name__ == "__main__":
    for ob in list(globals().values()):
        if callable(ob) and ob.__name__.startswith("test_"):
            print(f"{ob.__name__} ...")
            ob()
    print("done")
