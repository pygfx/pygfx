import gc
import time
import weakref

from pygfx.utils.trackable import Trackable, Store, PropTracker
from pygfx.utils import ReadOnlyDict

# ruff: noqa: B018 - in these tests we access values for their side-effect


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


class MyRootTrackable(Mixin, Trackable):
    def __init__(self):
        super().__init__()
        self._tracker = PropTracker()

    @property
    def tracker(self):
        return self._tracker

    def track_usage(self, label):
        return self._tracker.track_usage(label)

    def pop_changed(self):
        return self._tracker.pop_changed()

    @property
    def all_known_store_ids(self):
        return set(s["_trackable_id"] for s in self._tracker._stores.keys())


class MyTrackable(Mixin, Trackable):
    @property
    def known_trackers(self):
        return set(self._store["_trackable_trackers"])


class MyTrackable2(MyTrackable):
    pass


def test_changes_on_root():
    # Test basic stuff on a root object

    rt = MyRootTrackable()

    # Set an attribute, nothing special
    rt.foo = 42
    assert not rt.pop_changed()

    # Now mark the attribute
    with rt.track_usage("L1"):
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
    # Test basic stuff on a sub object

    root = MyRootTrackable()
    t1 = MyTrackable()
    t2 = MyTrackable()

    make_changes_on_sub(root, root, t1, t2)


def test_changes_on_sub_sub():
    # Test basic stuff on a sub sub object

    root = MyRootTrackable()
    t1 = MyTrackable()
    t2 = MyTrackable()

    root.sub1 = MyTrackable()
    make_changes_on_sub(root, root.sub1, t1, t2)


def test_changes_on_sub_sub_sub():
    # Test basic stuff on a sub sub sub object
    root = MyRootTrackable()
    t1 = MyTrackable()
    t2 = MyTrackable()

    root.sub2 = MyTrackable()
    root.sub2.sub1 = MyTrackable()
    make_changes_on_sub(root, root.sub2.sub1, t1, t2)


def test_changes_on_sub_external_root():
    # Test basic stuff on a trackable object, with a separate root

    root = MyRootTrackable()
    parent = MyTrackable()
    t1 = MyTrackable()
    t2 = MyTrackable()

    make_changes_on_sub(root, parent, t1, t2)


def make_changes_on_sub(root, parent, t1, t2):
    parent.sub1 = t1

    # -- all the same as above

    # Set an attribute, nothing special
    t1.foo = 42
    assert not root.pop_changed()

    # Now mark the attribute
    with root.track_usage("L1"):
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

    # None is a valid value
    t1.foo = None
    assert root.pop_changed() == {"L1"}

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

    with root.track_usage("L1"):
        parent.sub1.foo
    assert not root.pop_changed()

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


def test_tuple_values():
    rt = MyRootTrackable()

    offset = 0

    v1 = (1, offset + 2)
    v2 = (3, offset + 4)
    v3 = (1, offset + 2)

    assert v1 is not v3

    with rt.track_usage("foo"):
        rt.foo
    assert rt.pop_changed() == set()

    # Changing the value marks a change
    rt.foo = v1
    assert rt.pop_changed() == {"foo"}
    rt.foo = v2
    assert rt.pop_changed() == {"foo"}

    # If the object is different, but the value is the same, it checks the value!
    rt.foo = v1
    assert rt.pop_changed() == {"foo"}
    rt.foo = v3
    assert rt.pop_changed() == set()

    # This means that everything works as expected
    v3 += (8,)
    assert rt.foo != v3
    rt.foo = v3
    assert rt.pop_changed() == {"foo"}

    # So does this
    rt.foo = (0, offset + 1)
    assert rt.pop_changed() == {"foo"}
    rt.foo = (0, offset + 2)
    rt.foo = (0, 0 + 3)
    assert rt.pop_changed() == {"foo"}
    rt.foo = (0, offset + 4)
    rt.foo = (0, offset + 2)
    rt.foo = (0, offset + 3)
    assert rt.pop_changed() == set()


def test_list_values():
    # Lists are not hashable, so are tracked by id

    rt = MyRootTrackable()

    v1 = [1, 2]
    v2 = [3, 4]
    v3 = [1, 2]

    with rt.track_usage("foo"):
        rt.foo
    assert rt.pop_changed() == set()

    # Changing the value marks a change
    rt.foo = v1
    assert rt.pop_changed() == {"foo"}
    rt.foo = v2
    assert rt.pop_changed() == {"foo"}

    # Even if the value is the same - it tracks the object!
    rt.foo = v1
    assert rt.pop_changed() == {"foo"}
    rt.foo = v3
    assert rt.pop_changed() == {"foo"}

    # This means that ...
    v3.append(8)
    assert rt.foo == v3  # changed in-place
    rt.foo = v3  # so this does not do much
    assert rt.pop_changed() == set()

    # It can even mean that the second assert below fails, because when
    # that last list is allocated, it may use the memory previously
    # occupied by [0, 1], so it has the same id ... This is hard to
    # test reliably, but it *can* happen.
    rt.foo = [0, 1]
    assert rt.pop_changed() == {"foo"}
    rt.foo = [0, 2]  # on this line, the previous list is freed
    rt.foo = [0, 3]  # a new list object is allocated
    # assert rt.pop_changed() == {"foo"}


def test_dict_and_readonlydict():
    rt = MyRootTrackable()

    with rt.track_usage("foo"):
        rt.foo
    assert rt.pop_changed() == set()

    # Can set a dict
    d = {"foo": 42}
    rt.foo = d
    assert rt.pop_changed() == {"foo"}

    # Which tracks by id ...
    rt.foo = d
    assert rt.pop_changed() == set()

    # ... not by value
    rt.foo = {"foo": 42}
    assert rt.pop_changed() == {"foo"}

    # But ReadOnlyDict is tracked by value, because it is hashable
    rt.foo = ReadOnlyDict({"foo": 42})
    assert rt.pop_changed() == {"foo"}

    # Changing again with the same value
    rt.foo = ReadOnlyDict({"foo": 42})
    assert rt.pop_changed() == set()


def test_whole_tree_get_removed():
    # When a branch is removed, check that the rest of that branch stops tracking too.

    root = MyRootTrackable()
    t1 = MyTrackable()
    t2 = MyTrackable()
    t3 = MyTrackable()
    root.sub1 = t1
    root.sub1.sub1 = t2
    root.sub1.sub1.sub1 = t3

    with root.track_usage("x"):
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
    # Test changing a value on a deep prop

    root = MyRootTrackable()
    tree1 = MyTrackable()
    tree1.sub1 = MyTrackable()
    tree1.sub1.sub1 = t1 = MyTrackable()
    tree2 = MyTrackable()
    tree2.sub1 = MyTrackable()
    tree2.sub1.sub1 = t2 = MyTrackable()

    root.sub1 = tree1

    with root.track_usage("x"):
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


def test_reacting_to_trackable_presence1():
    # Test that changing from None to something triggers a change
    # This could represent changing geometry.normals from None to a Buffer.

    wobject = MyRootTrackable()
    wobject.sub1 = MyTrackable()  # geometry
    wobject.sub1.sub2 = None

    with wobject.track_usage("shader"):
        assert wobject.sub1.sub2 is None

    assert not wobject.pop_changed()

    wobject.sub1.sub2 = MyTrackable()  # e.g. normal

    assert wobject.pop_changed() == {"shader"}


def test_reacting_to_trackable_presence2():
    # Test that changing from nothing to something triggers a change
    # This could represent changing geometry.normals from undefined to a Buffer.

    wobject = MyRootTrackable()
    wobject.sub1 = MyTrackable()  # geometry

    with wobject.track_usage("shader"):
        res = hasattr(wobject.sub1._store, "sub2")
        assert not res

    assert not wobject.pop_changed()

    wobject.sub1.sub2 = MyTrackable()  # e.g. normal

    assert wobject.pop_changed() == {"shader"}


def test_reacting_to_trackable_presence3():
    # Test that changing from nothing to something triggers a change
    # This could represent changing geometry.normals from undefined to None.

    wobject = MyRootTrackable()
    wobject.sub1 = MyTrackable()  # geometry

    with wobject.track_usage("shader"):
        res = hasattr(wobject.sub1._store, "sub2")
        assert not res

    assert not wobject.pop_changed()

    wobject.sub1.sub2 = None

    assert wobject.pop_changed() == {"shader"}


def test_multiple_labels():
    # Test that you can track multiple labels

    root = MyRootTrackable()
    root.sub1 = MyTrackable()
    root.sub2 = MyTrackable()

    with root.track_usage("foo"):
        root.sub1.foo

    with root.track_usage("bar"):
        root.sub2.foo

    assert not root.pop_changed()

    root.sub1.foo = 42
    assert root.pop_changed() == {"foo"}

    root.sub2.foo = 52
    assert root.pop_changed() == {"bar"}

    root.sub1.foo = 43
    root.sub2.foo = 53
    assert root.pop_changed() == {"foo", "bar"}


def test_multiple_labels2():
    # Now the same object is known under different names!

    root = MyRootTrackable()
    t1 = MyTrackable()
    root.sub1 = t1
    root.sub2 = t1

    with root.track_usage("foo"):
        root.sub1.foo

    with root.track_usage("bar"):
        root.sub2.foo

    root.sub1.foo = 42
    assert root.pop_changed() == {"foo", "bar"}

    root.sub2.foo = 52
    assert root.pop_changed() == {"foo", "bar"}

    root.sub1.foo = 53
    root.sub2.foo = 53
    assert root.pop_changed() == {"foo", "bar"}


def test_track_trackables0():
    # This represents tracking the resource objects themselves - without "!"

    root = MyRootTrackable()
    t1 = MyTrackable()
    t2 = MyTrackable()

    t1.foo = t2.foo = 42

    root.sub1 = t1

    with root.track_usage("format"):
        root.sub1.foo

    with root.track_usage("resources"):
        root.sub1

    t1.foo = 43
    assert root.pop_changed() == {"format"}

    root.sub1 = t2
    assert root.pop_changed() == {"format"}  # no resources

    t2.foo = 43
    assert root.pop_changed() == {"format"}

    root.sub1 = t1
    assert root.pop_changed() == set()  # no resources


def test_track_trackables1():
    # This represents tracking the resource objects themselves - with "!"

    root = MyRootTrackable()
    t1 = MyTrackable()
    t2 = MyTrackable()

    t1.foo = t2.foo = 42

    root.sub1 = t1

    with root.track_usage("format"):
        root.sub1.foo

    with root.track_usage("!resources"):
        root.sub1

    t1.foo = 43
    assert root.pop_changed() == {"format"}

    root.sub1 = t2
    assert root.pop_changed() == {"format", "resources"}

    t2.foo = 43
    assert root.pop_changed() == {"format"}

    root.sub1 = t1
    assert root.pop_changed() == {"resources"}


def test_track_trackables2():
    # This represents tracking the resource objects themselves - deeper

    root = MyRootTrackable()
    tree1 = MyTrackable()
    tree2 = MyTrackable()
    tree1.sub1 = t1 = MyTrackable()
    tree2.sub1 = t2 = MyTrackable()

    tree1.sub1.foo = tree2.sub1.foo = 42

    root.sub1 = tree1

    with root.track_usage("format"):
        root.sub1.sub1.foo

    with root.track_usage("!resources"):
        root.sub1.sub1

    t1.foo = 43
    assert root.pop_changed() == {"format"}

    root.sub1 = tree2
    assert root.pop_changed() == {"format", "resources"}

    t2.foo = 43
    assert root.pop_changed() == {"format"}

    root.sub1 = tree1
    assert root.pop_changed() == {"resources"}


def test_track_substore0():
    # This represents tracking the resource objects themselves - without "!"

    root = MyRootTrackable()
    t1 = Store()
    t2 = Store()

    t1.foo = t2.foo = 42

    root.sub1 = t1

    with root.track_usage("format"):
        root.sub1.foo

    with root.track_usage("resources"):
        root.sub1

    t1.foo = 43
    assert root.pop_changed() == {"format"}

    root.sub1 = t2
    assert root.pop_changed() == {"format"}  # no resources

    t2.foo = 43
    assert root.pop_changed() == {"format"}

    root.sub1 = t1
    assert root.pop_changed() == set()  # no resources


def test_track_substore1():
    # This represents tracking the resource objects themselves - with "!"

    root = MyRootTrackable()
    t1 = Store()
    t2 = Store()

    t1.foo = t2.foo = 42

    root.sub1 = t1

    with root.track_usage("format"):
        root.sub1.foo

    with root.track_usage("!resources"):
        root.sub1

    t1.foo = 43
    assert root.pop_changed() == {"format"}

    root.sub1 = t2
    assert root.pop_changed() == {"format", "resources"}

    t2.foo = 43
    assert root.pop_changed() == {"format"}

    root.sub1 = t1
    assert root.pop_changed() == {"resources"}


def test_track_substore2():
    # This represents tracking the resource objects themselves - deeper

    root = MyRootTrackable()
    tree1 = Store()
    tree2 = Store()
    tree1.sub1 = t1 = Store()
    tree2.sub1 = t2 = Store()

    tree1.sub1.foo = tree2.sub1.foo = 42

    root.sub1 = tree1

    with root.track_usage("format"):
        root.sub1.sub1.foo

    with root.track_usage("!resources"):
        root.sub1.sub1

    t1.foo = 43
    assert root.pop_changed() == {"format"}

    root.sub1 = tree2
    assert root.pop_changed() == {"format", "resources"}

    t2.foo = 43
    assert root.pop_changed() == {"format"}

    root.sub1 = tree1
    assert root.pop_changed() == {"resources"}


def test_track_trackables_typing1():
    # This represents tracking the resource objects themselves

    root = MyRootTrackable()
    t1 = MyTrackable()
    t2 = MyTrackable2()  # different type

    t1.foo = t2.foo = 42

    root.sub1 = t1

    with root.track_usage("format"):
        root.sub1.foo

    with root.track_usage("!resources"):
        root.sub1

    root.sub1 = t2
    assert root.pop_changed() == {"format", "resources"}


def test_track_trackables_typing2():
    # Test that if the trackable changes type, everything triggers

    root = MyRootTrackable()
    tree1 = MyTrackable()
    tree2 = MyTrackable()
    tree1.sub1 = MyTrackable()
    tree2.sub1 = MyTrackable2()  # different type

    tree1.sub1.foo = tree2.sub1.foo = 42

    root.sub1 = tree1

    with root.track_usage("format"):
        root.sub1.sub1.foo

    with root.track_usage("!resources"):
        root.sub1.sub1

    root.sub1 = tree2
    assert root.pop_changed() == {"format", "resources"}


def test_track_externals():
    root1 = MyRootTrackable()
    root2 = MyRootTrackable()
    root3 = MyRootTrackable()
    ext = MyTrackable()

    with root1.track_usage("x"):
        ext.foo
    with root2.track_usage("y"):
        ext.foo

    assert not root1.pop_changed()
    assert not root2.pop_changed()
    assert not root3.pop_changed()

    ext.foo = 42

    assert root1.pop_changed() == {"x"}
    assert root2.pop_changed() == {"y"}
    assert not root3.pop_changed()


def test_cleanup1():
    # Test that removing a sub cleans it up internally
    root = MyRootTrackable()
    root.sub1 = MyTrackable()

    with root.track_usage("x"):
        root.sub1.foo

    id = root.sub1._store["_trackable_id"]
    assert id in root.all_known_store_ids

    root.sub1 = None
    assert id not in root.all_known_store_ids


def test_cleanup2():
    # Test that removing a sub cleans it up internally, also nested
    root = MyRootTrackable()
    root.sub1 = MyTrackable()
    root.sub1.sub1 = MyTrackable()

    with root.track_usage("x"):
        root.sub1.sub1.foo

    id = root.sub1.sub1._store["_trackable_id"]
    assert id in root.all_known_store_ids

    root.sub1 = None
    assert id not in root.all_known_store_ids


def test_cleanup3():
    # Test that removing a sub cleans it up internally, also deeper
    root = MyRootTrackable()
    root.sub1 = MyTrackable()
    root.sub1.sub1 = MyTrackable()

    with root.track_usage("x"):
        root.sub1.sub1.foo

    id = root.sub1.sub1._store["_trackable_id"]
    assert id in root.all_known_store_ids

    root.sub1.sub1 = None
    assert id not in root.all_known_store_ids


def test_cleanup4():
    # Test that removing an ext cleans it up internally

    root = MyRootTrackable()
    ext = MyTrackable()
    ext.sub1 = MyTrackable()

    # Done nothin' yet

    id0 = root._store["_trackable_id"]
    id1 = ext._store["_trackable_id"]
    id2 = ext.sub1._store["_trackable_id"]

    assert id0 not in root.all_known_store_ids
    assert id1 not in root.all_known_store_ids
    assert id2 not in root.all_known_store_ids

    assert root not in ext.known_trackers
    assert root not in ext.sub1.known_trackers

    # Listen to stuff in ext

    with root.track_usage("x"):
        ext.sub1.foo

    assert id0 not in root.all_known_store_ids
    assert id1 in root.all_known_store_ids
    assert id2 in root.all_known_store_ids

    assert root.tracker in ext.known_trackers
    assert root.tracker in ext.sub1.known_trackers

    # Now listen to stuff in root's tree

    with root.track_usage("x"):
        root.foo

    assert id0 in root.all_known_store_ids
    assert id1 not in root.all_known_store_ids
    assert id2 not in root.all_known_store_ids

    assert root.tracker not in ext.known_trackers
    assert root.tracker not in ext.sub1.known_trackers

    ext.sub1.foo = 42
    assert not root.pop_changed()


def test_gc1():
    # Test that the root does not hold on to sub objects
    root = MyRootTrackable()
    root.sub1 = MyTrackable()

    with root.track_usage("x"):
        root.sub1.foo

    ref = weakref.ref(root.sub1)

    root.sub1 = None
    assert not ref()


def test_gc2():
    # Test that the root does not hold onto other objects
    root = MyRootTrackable()
    ext = MyTrackable()

    with root.track_usage("x"):
        ext.foo

    ext.foo = 32

    ref = weakref.ref(ext)
    del ext
    gc.collect()
    assert not ref()


def profile_runner():
    n_objects = 1000
    n_labels = 4
    n_atts = 20 // n_labels
    n_access = 8

    root = MyRootTrackable()
    root.sub1 = MyTrackable()
    root.sub2 = MyTrackable()
    root.sub1.leaf1 = MyTrackable()
    root.sub1.leaf2 = MyTrackable()

    # Setup tracking
    for label in range(n_labels):
        with root.track_usage(str(label)):
            for attr in range(n_atts):
                key = f"attr_{label}_{attr}"
                root.sub1._store[key] = 0
                getattr(root.sub1._store, key)

    t0 = time.perf_counter()
    # Do a buch of work
    for _ob in range(n_objects):
        for i in range(n_access):
            setattr(root.sub1._store, f"attr_0_{i}", i // 2)
    t1 = time.perf_counter()
    assert root.pop_changed() == {"0"}

    return t1 - t0


def profile_speed():
    import cProfile, io, pstats  # noqa: E401

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
