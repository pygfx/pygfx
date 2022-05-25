import time

from pygfx.objects._trackable import Trackable, RootTrackable



class MyRootTrackable(RootTrackable):

    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        if not name.startswith("_"):
            self._track_set(name, value)


class MyTrackable(Trackable):

    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        if not name.startswith("_"):
            self._track_set(name, value)


def test_changes_on_root():

    rt = MyRootTrackable()

    # Set an attribute, nothing special
    rt.foo = 42
    assert not rt.pop_changed()

    # Now mark the attribute
    with rt.track_usage("L1", False):
        rt._track_get("foo", 42)
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

    parent.sub = t1

    # -- all the same as above

    # Set an attribute, nothing special
    t1.foo = 42
    assert not root.pop_changed()

    # Now mark the attribute
    with root.track_usage("L1", False):
        t1._track_get("foo", 42)
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

    parent.sub = None

    # Indeed the object has changed
    assert root.pop_changed() == {"L1"}

    # Nothing happens if we change the value
    t1.foo = 2
    assert not root.pop_changed()
    t1.foo = 42
    assert not root.pop_changed()

    # Putting it back again
    parent.sub = t1
    assert root.pop_changed() == {"L1"}

    # -- Replacing the sub

    # Replace with sub with different value
    t1.foo = 42
    parent.sub = t1
    root.pop_changed()
    #
    t2.foo = 43
    parent.sub = t2
    assert root.pop_changed() == {"L1"}

    # Replace with sub with same value
    t1.foo = 42
    parent.sub = t1
    root.pop_changed()
    #
    t2.foo = 42
    parent.sub = t2
    assert not root.pop_changed()

    # Replace with sub and back again
    t1.foo = 42
    parent.sub = t1
    root.pop_changed()
    #
    t2.foo = 43
    parent.sub = t2
    parent.sub = t1
    assert not root.pop_changed()

    # Replace with sub and back again, but now its value has changed
    t1.foo = 42
    parent.sub = t1
    root.pop_changed()
    #
    t2.foo = 42
    parent.sub = t2
    t1.foo = 43
    parent.sub = t1
    assert root.pop_changed() == {"L1"}


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
                root.sub1._track_get(f"attr_{level}_{attr}", 4)

    t0 = time.perf_counter()
    # Do a buch of work
    for ob in range(n_objects):
        for i in range(n_access):
            setattr(root.sub1, f"attr_0_{i}", i//2)
    t1 = time.perf_counter()
    assert root.pop_changed() == {0}#set(range(n_levels))

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

