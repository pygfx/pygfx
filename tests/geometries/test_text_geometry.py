from pygfx import text_geometry
from pytest import raises


def test_text_geometry1():

    # Let's try some special cases first

    # Must specify text or html or whatever
    with raises(ValueError):
        text_geometry()

    # Empty string - still has one item (whitespace)
    geo = text_geometry(text="")
    geo.positions.nitems == 1
    geo.indices.nitems == 1
    geo.sizes.nitems == 1

    # Only a space
    geo = text_geometry(text=" ")
    geo.positions.nitems == 1
    geo.indices.nitems == 1
    geo.sizes.nitems == 1

    # One char
    geo = text_geometry(text="a")
    geo.positions.nitems == 1
    geo.indices.nitems == 1
    geo.sizes.nitems == 1

    # Two words with 3 chars in total
    geo = text_geometry(text="a bc")
    geo.positions.nitems == 3
    geo.indices.nitems == 3
    geo.sizes.nitems == 3


if __name__ == "__main__":
    for ob in list(globals().values()):
        if callable(ob) and ob.__name__.startswith("test_"):
            print(f"{ob.__name__} ...")
            ob()
    print("done")
