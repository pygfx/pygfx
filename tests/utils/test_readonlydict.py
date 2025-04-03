from pygfx.utils import ReadOnlyDict
import pytest


def test_readonlydict_immutable():
    d = ReadOnlyDict(foo=3, bar=4, spam=5)

    with pytest.raises(TypeError):
        d["foo"] = 1
    with pytest.raises(TypeError):
        d["xxxxxxx"] = 1
    with pytest.raises(TypeError):
        del d["foo"]
    with pytest.raises(TypeError):
        d.update({})
    with pytest.raises(TypeError):
        d.clear()
    with pytest.raises(TypeError):
        d.pop("foo", None)
    with pytest.raises(TypeError):
        d.popitem()
    with pytest.raises(TypeError):
        d.setdefault("foo", 42)
    with pytest.raises(TypeError):
        d.setdefault("foo", 42)

    # Also immutable (using __slots__)
    with pytest.raises(AttributeError):
        d.foo = 3


def test_readonlydict_hash():
    d1 = ReadOnlyDict(foo=3, bar=4, spam=5)
    d2 = ReadOnlyDict(foo=3, bar=4, spam=5)
    d3 = ReadOnlyDict(d1)
    d4 = ReadOnlyDict(bar=4, spam=5, foo=3)
    d5 = ReadOnlyDict(spam=5, bar=4, foo=3)

    for d in [d2, d3, d4, d5]:
        assert d1 == d
        assert hash(d1) == hash(d)

    d11 = ReadOnlyDict(foo=2, bar=4, spam=5)
    d12 = ReadOnlyDict(foo=3, bar=4, spam=5.01)
    d13 = ReadOnlyDict(foo=3, bar=4)
    d14 = ReadOnlyDict(foo=3, spam=5)
    d15 = ReadOnlyDict(bar=4, spam=5)
    d16 = ReadOnlyDict(foo=3, bar=4, spam=5, x=1)
    d17 = ReadOnlyDict(foo=3, x=1, bar=4, spam=5)

    for d in [d11, d12, d13, d14, d15, d16, d17]:
        assert d1 != d
        assert hash(d1) != hash(d)


def test_readonlydict_init():
    d1 = ReadOnlyDict(foo=3, bar=4, spam=5)
    d2 = ReadOnlyDict(d1)
    d3 = ReadOnlyDict({"foo": 3, "bar": 4, "spam": 5})
    d4 = ReadOnlyDict({"foo": 3, "bar": 4}, spam=5)
    d5 = ReadOnlyDict(dict(d1))

    for d in [d2, d3, d4, d5]:
        assert d1 == d
        assert hash(d1) == hash(d)


def test_readonlydict_values():
    # Unhashable types
    with pytest.raises(TypeError):
        d1 = ReadOnlyDict(foo=[])
    with pytest.raises(TypeError):
        d1 = ReadOnlyDict(foo=dict(foo=3))

    # This works
    ReadOnlyDict(foo=ReadOnlyDict(foo=3))

    class Foo:
        def __init__(self, v):
            self.v = v

    # This works, but the Foo object is hashed using its id
    d1 = ReadOnlyDict(foo=Foo(3))
    d2 = ReadOnlyDict(foo=d1["foo"])
    d3 = ReadOnlyDict(foo=Foo(3))

    assert hash(d1) == hash(d2)
    assert hash(d1) != hash(d3)


if __name__ == "__main__":
    test_readonlydict_immutable()
    test_readonlydict_hash()
    test_readonlydict_init()
    test_readonlydict_values()
