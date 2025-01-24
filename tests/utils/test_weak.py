import gc
import weakref

from pygfx.utils.weak import WeakAssociativeContainer


class Foo:
    pass


class Bar:
    pass


class Spam:
    pass


class Something:
    # The object that'd be the value in a WeakKeyDictionary
    pass


def test_weak_associative_container1():
    # Test usage with one argument.
    # This is basically replicating a WeakKeyDictionary.

    wac = WeakAssociativeContainer()

    f1 = Foo()
    f2 = Foo()
    f3 = Foo()
    f9 = Foo()

    something1 = wac.setdefault((f1,), Something())
    something2 = wac.setdefault((f2,), Something())
    something3 = wac[(f3,)] = Something()

    # All somethings are different
    assert something1 is not something2
    assert something2 is not something3
    assert something3 is not something1

    # Proper re-use
    assert wac[(f1,)] is something1
    assert wac[(f2,)] is something2
    assert wac.get((f3,)) is something3

    assert wac.get((f9,)) is None

    assert wac.get_associated(f1) == {something1}

    # Prepare for deleting stuff
    something_refs = weakref.WeakSet((something1, something2, something3))
    del something1, something2, something3

    # Delete f1
    del f1
    gc.collect()

    assert len(something_refs) == 2

    del f2
    gc.collect()
    assert len(something_refs) == 1

    del f3
    gc.collect()
    assert len(something_refs) == 0


def test_weak_associative_container2():
    # Test usage with two arguments.
    # This behavior can be replicated with two WeakKeyDictionary's.

    wac = WeakAssociativeContainer()

    f1 = Foo()
    f2 = Foo()
    b1 = Bar()
    b2 = Bar()
    f9 = Foo()

    something11 = wac.setdefault((f1, b1), Something())
    something12 = wac.setdefault((f1, b2), Something())
    something21 = wac.setdefault((f2, b1), Something())
    something22 = wac.setdefault((f2, b2), Something())

    # Get refs

    # All somethings are different
    somethings = something11, something12, something21, something22
    assert len(set(id(something) for something in somethings)) == 4

    # Proper re-use
    assert wac[(f1, b1)] is something11
    assert wac[(f2, b1)] is something21
    assert wac.get((f1, b2)) is something12
    assert wac.get((f2, b2)) is something22

    assert wac.get((f9, b1)) is None

    assert wac.get_associated(f1) == {something11, something12}

    # Prepare for deleting stuff
    something_refs = weakref.WeakSet(somethings)
    del somethings, something11, something12, something21, something22

    # Delete f1
    del f1
    gc.collect()

    assert len(something_refs) == 2

    del b1
    gc.collect()
    assert len(something_refs) == 1

    del f2, b2
    gc.collect()
    assert len(something_refs) == 0


def test_weak_associative_container3():
    # Test usage with three arguments.
    # This is where this little tool starts to shine.

    wac = WeakAssociativeContainer()

    f1 = Foo()
    f2 = Foo()
    b1 = Bar()
    b2 = Bar()
    s1 = Spam()
    s2 = Spam()
    f9 = Foo()

    something111 = wac.setdefault((f1, b1, s1), Something())
    something222 = wac.setdefault((f2, b2, s2), Something())
    something211 = wac.setdefault((f2, b1, s1), Something())
    something121 = wac.setdefault((f1, b2, s1), Something())
    something112 = wac.setdefault((f1, b1, s2), Something())
    something122 = wac.setdefault((f1, b2, s2), Something())
    something212 = wac.setdefault((f2, b1, s2), Something())
    something221 = wac.setdefault((f2, b2, s1), Something())

    # Get refs

    # All somethings are different
    somethings = (
        something111,
        something222,
        something211,
        something121,
        something112,
        something122,
        something212,
        something221,
    )

    assert len(set(id(something) for something in somethings)) == 8

    # Proper re-use
    assert wac[(f1, b1, s1)] is something111
    assert wac[(f2, b2, s2)] is something222
    assert wac[(f2, b1, s1)] is something211
    assert wac[(f1, b2, s1)] is something121
    assert wac[(f1, b1, s2)] is something112

    assert wac.get((f9, b1, s1)) is None

    assert wac.get_associated(f1) == {
        something111,
        something121,
        something112,
        something122,
    }

    # Prepare for deleting stuff
    something_refs = weakref.WeakSet(somethings)
    del (
        somethings,
        something111,
        something222,
        something211,
        something121,
        something112,
        something122,
        something212,
        something221,
    )

    # Delete f1
    del f1
    gc.collect()

    assert len(something_refs) == 4

    del b1
    gc.collect()
    assert len(something_refs) == 2

    del s1
    gc.collect()
    assert len(something_refs) == 1

    del f2, b2, s2
    gc.collect()
    assert len(something_refs) == 0


if __name__ == "__main__":
    test_weak_associative_container1()
    test_weak_associative_container2()
    test_weak_associative_container3()
