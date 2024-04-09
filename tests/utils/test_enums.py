from pygfx.utils.enums import Enum


def test_enums():
    MyOption = Enum(
        "MyOption",
        auto="auto",
        some_attr="some-attr",  # wgpu-style values
        foo=None,  # value is the same as the key, most-used in pygfx
    )

    # Use dir() to get an (alphabetic) list of keys / options.
    assert dir(MyOption) == ["auto", "foo", "some_attr"]

    # Iterate over the object to get a list of values, in original order.
    assert list(MyOption) == ["auto", "some-attr", "foo"]

    # The repr is actually useful.
    assert (
        str(MyOption) == "<pygfx.MyOption enum with values: 'auto', 'some-attr', 'foo'>"
    )

    # Attribute and map-like lookups are supported
    assert MyOption.some_attr == "some-attr"
    assert MyOption["some_attr"] == "some-attr"


def test_flags():

    # Our Enum class does flags too.

    MyFlag = Enum(
        "MyFlag",
        auto=0,
        foo=1,
        bar=2,
    )

    # Use dir() to get an (alphabetic) list of keys / options.
    assert dir(MyFlag) == ["auto", "bar", "foo"]

    # Iterate over the object to get a list of values, in original order.
    assert list(MyFlag) == [0, 1, 2]

    # The repr is actually useful.
    assert str(MyFlag) == "<pygfx.MyFlag enum with values: 0, 1, 2>"

    # Attribute and map-like lookups are supported
    assert MyFlag.bar == 2
    assert MyFlag["bar"] == 2
