import time

from pygfx.utils.trackable import Trackable, RootTrackable


class Mixin:
    def __init__(self):
        super().__init__()
        self._store["foo"] = 0

    @property
    def foo(self):
        return self._store.foo

    @foo.setter
    def foo(self, value):
        self._store.foo = value

    @property
    def sub1(self):
        return self._store.sub1

    @sub1.setter
    def sub1(self, value):
        self._store.sub1 = value

    @property
    def sub2(self):
        return self._store.sub2

    @sub2.setter
    def sub2(self, value):
        self._store.sub2 = value

    @property
    def sub3(self):
        return self._store.sub3

    @sub3.setter
    def sub3(self, value):
        self._store.sub3 = value


class MyRootTrackable(Mixin, RootTrackable):
    pass


class MyTrackable(Mixin, Trackable):
    pass


def test_changes_on_root():

    rt = MyRootTrackable()

    # Set an attribute, nothing special
    rt.foo = 42
    assert not rt.pop_changed()

    # Now mark the attribute
    with rt.track_usage("L1", False):
        rt.foo
    assert not rt.pop_changed()

    # Set to a different value
    rt.foo = 43
    assert rt.pop_changed() == {"L1"}

    # Setting to same does not trigger a change
    rt.foo = 43
    assert not rt.pop_changed()

    # But back to the old value does
    rt.foo = 42
    assert rt.pop_changed() == {"L1"}

    # Value is remembered
    rt.foo = 43
    rt.foo = 42
    assert not rt.pop_changed()


def test_changes_on_sub():
    root = MyRootTrackable()
    t1 = MyTrackable()
    t2 = MyTrackable()

    make_changes_on_sub(root, root, t1, t2)


def test_changes_on_sub_sub():
    root = MyRootTrackable()
    t1 = MyTrackable()
    t2 = MyTrackable()

    root.sub1 = MyTrackable()
    make_changes_on_sub(root, root.sub1, t1, t2)


def test_changes_on_sub_sub_sub():
    root = MyRootTrackable()
    t1 = MyTrackable()
    t2 = MyTrackable()

    root.sub2 = MyTrackable()
    root.sub2.sub1 = MyTrackable()
    make_changes_on_sub(root, root.sub2.sub1, t1, t2)


def make_changes_on_sub(root, parent, t1, t2):

    parent.sub1 = t1

    # -- all the same as above

    # Set an attribute, nothing special
    t1.foo = 42
    assert not root.pop_changed()

    # Now mark the attribute
    with root.track_usage("L1", False):
        parent.sub1.foo
    assert not root.pop_changed()

    # Set to a different value
    t1.foo = 43
    assert root.pop_changed() == {"L1"}

    # Setting to same does not trigger a change
    t1.foo = 43
    assert not root.pop_changed()

    # But back to the old value does
    t1.foo = 42
    assert root.pop_changed() == {"L1"}

    # Value is remembered
    t1.foo = 43
    t1.foo = 42
    assert not root.pop_changed()

    # -- Removing the sub

    parent.sub1 = None

    # Indeed the object has changed
    assert root.pop_changed() == {"L1"}

    # Nothing happens if we change the value
    t1.foo = 2
    assert not root.pop_changed()
    t1.foo = 42
    assert not root.pop_changed()

    # Putting it back again
    parent.sub1 = t1
    assert root.pop_changed() == {"L1"}

    # -- Replacing the sub

    # Replace with sub with different value
    t1.foo = 42
    parent.sub1 = t1
    root.pop_changed()
    #
    t2.foo = 43
    parent.sub1 = t2
    assert root.pop_changed() == {"L1"}

    # Replace with sub with same value
    t1.foo = 42
    parent.sub1 = t1
    root.pop_changed()
    #
    t2.foo = 42
    parent.sub1 = t2
    assert not root.pop_changed()

    # Replace with sub and back again
    t1.foo = 42
    parent.sub1 = t1
    root.pop_changed()
    #
    t2.foo = 43
    parent.sub1 = t2
    parent.sub1 = t1
    assert not root.pop_changed()

    # Replace with sub and back again, but now its value has changed
    t1.foo = 42
    parent.sub1 = t1
    root.pop_changed()
    #
    t2.foo = 42
    parent.sub1 = t2
    t1.foo = 43
    parent.sub1 = t1
    assert root.pop_changed() == {"L1"}


def test_whole_tree_get_removed():

    root = MyRootTrackable()
    t1 = MyTrackable()
    t2 = MyTrackable()
    t3 = MyTrackable()
    root.sub1 = t1
    root.sub1.sub1 = t2
    root.sub1.sub1.sub1 = t3

    with root.track_usage("x", False):
        root.foo
        root.sub1.foo
        root.sub1.sub1.foo
        root.sub1.sub1.sub1.foo

    assert not root.pop_changed()

    # Check that is reacts to all subs

    root.foo = 1
    assert root.pop_changed() == {"x"}

    t1.foo = 1
    assert root.pop_changed() == {"x"}

    t2.foo = 1
    assert root.pop_changed() == {"x"}

    t3.foo = 1
    assert root.pop_changed() == {"x"}

    # Remove the hole tree

    root.sub1 = None
    assert root.pop_changed() == {"x"}

    # This should still work
    root.foo = 2
    assert root.pop_changed() == {"x"}

    # But these not

    t1.foo = 2
    assert not root.pop_changed()

    t2.foo = 2
    assert not root.pop_changed()

    t3.foo = 2
    assert not root.pop_changed()


def test_deep_prop():

    root = MyRootTrackable()
    tree1 = MyTrackable()
    tree1.sub1 = MyTrackable()
    tree1.sub1.sub1 = t1 = MyTrackable()
    tree2 = MyTrackable()
    tree2.sub1 = MyTrackable()
    tree2.sub1.sub1 = t2 = MyTrackable()

    root.sub1 = tree1

    with root.track_usage("x", False):
        root.sub1.sub1.sub1.foo

    assert not root.pop_changed()

    # Make a change to t1
    t1.foo = 42
    assert root.pop_changed() == {"x"}

    # Make a change in t2
    t2.foo = 42
    assert not root.pop_changed()

    # Replace the tree (same values)
    root.sub1 = tree2

    assert not root.pop_changed()

    # Make a change to t1
    t1.foo = 43
    assert not root.pop_changed()

    # Make a change in t2
    t2.foo = 44
    assert root.pop_changed() == {"x"}

    # Replace the tree (different values)
    root.sub1 = tree1
    assert root.pop_changed() == {"x"}


def test_reacting_to_trackable_presence():

    wobject = MyRootTrackable()
    wobject.sub1 = MyTrackable()  # geometry

    with wobject.track_usage("shader", False):
        res = hasattr(wobject.sub1._store, "sub2")
        assert not res

    assert not wobject.pop_changed()

    wobject.sub1.sub2 = MyTrackable()  # e.g. normal

    assert wobject.pop_changed() == {"shader"}


def test_track_setting_objects():
    pass  # TODO


def test_object_attached_twice_under_different_name():
    pass  # TODO


def test_garbage_collection():
    pass  # TODO


def profile_runner():
    n_objects = 1000
    n_levels = 4
    n_atts = 20 // n_levels
    n_access = 8

    root = MyRootTrackable()
    root.sub1 = MyTrackable()
    root.sub2 = MyTrackable()
    root.sub1.leaf1 = MyTrackable()
    root.sub1.leaf2 = MyTrackable()

    # Setup tracking
    for level in range(n_levels):
        with root.track_usage(level, False):
            for attr in range(n_atts):
                key = f"attr_{level}_{attr}"
                root.sub1._store[key] = 0
                getattr(root.sub1._store, key)

    t0 = time.perf_counter()
    # Do a buch of work
    for ob in range(n_objects):
        for i in range(n_access):
            setattr(root.sub1._store, f"attr_0_{i}", i // 2)
    t1 = time.perf_counter()
    assert root.pop_changed() == {0}  # set(range(n_levels))

    return t1 - t0


def profile_speed():
    import cProfile, io, pstats

    pr = cProfile.Profile()
    pr.enable()

    profile_runner()

    pr.disable()
    s = io.StringIO()
    sortby = pstats.SortKey.CUMULATIVE
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print(s.getvalue())

    print(profile_runner(), "secs")


if __name__ == "__main__":
    for ob in list(globals().values()):
        if callable(ob) and ob.__name__.startswith("test_"):
            print(ob.__name__)
            ob()
    print("done")

    # profile_speed()
