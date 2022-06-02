"""
Implements the base classes for trackable classes.

## Introduction

This module implemented in a more or less generic way. The code does not
refer to world objects, geometry, materials or resources. But don't be
fooled - this code is *very* much tailored to the needs of such objects.
In particular:

* we want to track properties (that may need to trigger a pipeline).
* we want to be able to track objects themselves depending on the situation
  (because the pipeline only cares about the object's properties while the
  bindings much match to the correct buffers/textures).
* we want to be able to swap out similar objects and not trigger of the
  new object has the same properties as before (so that we can e.g. replace
  colormaps or texture atlasses).
* we also want to track "external objects" that are not in the tree of the root.

The last two points mean that the code must be aware of the tree
structure, but must also be able to track external Trackable objects
(wich may also have a tree).

## High level overview

We define a Trackable object as something of which the properties can
be tracked (its properties can be other Trackable objects).

The RootTrackable is the object that keeps track of the changes. One
first tracks usage of properties, after which any changes to these
properties get communicated to this root. Any trackable can participate
(not only the trackables in the tree of the root).

## Finer details

The Trackable uses a Store - a custom dictionary object that has hooks
for attribute access. This is where the magic happens. The Trackable
object itself is just a container for the store. This has the benefit
that subclasses of Trackable just store the props that they want to
track on its store.

Similarly the functionality of the RootTrackable is offloaded to an
internal Root object. That way the RootTrackable (and its subclasses)
stay clean, and can implement their own API for the tracking.

The Store keeps a set of what roots need to be notified (using weak
refs). The root keeps a set of what stores notify it, for the sole
reason that the root can disconnect these stores when usage is
re-tracked.

Every store has a unique id, and the root uses (id, prop) tuples as
keys to keep track of things. When a property sets/unsets a Trackable,
the tree is resolved at that moment (i.e. the root is temporary aware
of the tree). This avoids the need for the root (or stores?) to keep
track of a hierarchy which would make the code a lot more complex (I
tried and it was not fun). The only downside (that I can think of) is
that setting `matererial.map = other_map` works while
`matererial.map = None; material.map = other_map` does not, because
there is no persistent knowledge of hierarchy.

"""

import weakref
import threading


global_id_counter = 0
global_lock = threading.RLock()
global_context = None

simple_value_types = None.__class__, bool, int, float, str


class Undefined:
    def __repr__(self):
        return "undefined"


undefined = Undefined()


class Trackable:
    """A base class to make an object trackable."""

    def __init__(self):
        self._store = Store()


class RootTrackable(Trackable):
    """Base class for the root trackable object."""

    def __init__(self):
        super().__init__()
        self._root_tracker = Root()


class TrackContext:
    """A context used when tracking usage of trackable objects."""

    def __init__(self, root, label):
        assert isinstance(root, Root)
        assert isinstance(label, str)
        self.root = root
        self.label = label

    def __enter__(self):
        global global_context
        global_lock.acquire()
        global_context = self
        self.root._track_init(self.label)
        return None

    def __exit__(self, value, type, tb):
        global global_context
        self.root._track_done(self.label)
        self.root = None
        self.label = 0
        global_context = None
        global_lock.release()


class Store(dict):
    """Object to store key-value pairs that are tracked. Each Trackable
    has one such object. Must use attribute access to trigger the tracking
    mechanics. The internals can use index access.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Give it an id
        global global_id_counter
        with global_lock:
            global_id_counter += 1
            self["_trackable_id"] = global_id_counter  # f"t{global_id_counter}"
        # Store what roots we need to notify of changes
        self["_trackable_roots"] = weakref.WeakSet()

    def __hash__(self):
        return self["_trackable_id"]

    def __setattr__(self, key, new_value):
        # Apply
        old_value = dict.get(self, key, undefined)
        self[key] = new_value
        # Notify the roots
        id = self["_trackable_id"]
        for root in self["_trackable_roots"]:
            root._track_set(id, key, old_value, new_value)

    def __getattribute__(self, key):
        value = undefined
        try:
            value = self[key]
            return value
        except KeyError:
            raise AttributeError(key) from None
        finally:
            if global_context:
                global_context.root._track_get(self, key, value, global_context.label)


class Root:
    """Object to store all the tracking data for a RootTrackable."""

    def __init__(self):
        # Keep track of what stores have a reference to *this*,
        # so that we can clear that reference when we want.
        # The store and root are always connected/disconnected together.
        self._stores = {}  # label -> weakset
        # The names to track. Names are (id, key)  -  name -> labels
        self._trackable_names = {}
        # Keep track of values  -  name -> (ref_value, cur_value)
        self._trackable_values = {}
        # Keep track of what has changed  -  name -> labels
        self._trackable_changed = {}

    def track_usage(self, label):
        """Used to track the usage of attributes. The result of this method
        should be used as a context manager. This method must only be
        used by the one system that is tracking this object's changes.
        """
        return TrackContext(self, label)

    def pop_changed(self):
        """Get a set of labels for which the object (and its child
        trackable objects) has changed. Also resets the changed status
        to unchanged. This method must only be used by the one system
        that is tracking this object's changes.
        """
        # Quick version

        # Reset the cached values
        for name in self._trackable_changed:
            try:
                ref_value, cur_value = self._trackable_values[name]
            except KeyError:
                pass
            else:
                self._trackable_values[name] = cur_value, cur_value
        # Collect changed label
        result = set()
        for labels in self._trackable_changed.values():
            result.update(labels)
        # Reset and return
        self._trackable_changed = {}
        return result

    def _track_store(self, store, label):
        # Introduce the store and root to each-other
        store["_trackable_roots"].add(self)
        self._stores[label].add(store)

    def _untrack_store(self, store, label):
        # Break the connection between root and store
        store["_trackable_roots"].discard(self)
        self._stores[label].discard(store)

    def _track_init(self, label):
        # Reset the tracking data for the given label
        for name in list(self._trackable_names):
            labels = self._trackable_names[name]
            labels.discard(label)
            if not labels:
                self._trackable_names.pop(name)
                self._trackable_values.pop(name, None)

        # Break connections with old stores
        for store in self._stores.pop(label, set()):
            store["_trackable_roots"].discard(self)
        self._stores[label] = weakref.WeakSet()

    def _track_done(self, label):
        # Remove backrefs-dict if its empty
        if not self._stores[label]:
            self._stores.pop(label, None)

    def _track_get(self, store, key, value, label):
        # Called when *any* trackable has an attribute GET while a
        # global context is active. The trackable can be in the tree
        # of the root (a (grand)child), but can also be part of a
        # detached tree.

        self._track_store(store, label)

        id = store["_trackable_id"]
        name = id, key

        self._trackable_names.setdefault(name, set()).add(label)
        if isinstance(value, simple_value_types):
            self._trackable_values[name] = value, value
        else:
            self._trackable_values.pop(name, None)

    def _track_set(self, id, key, old_value, new_value):
        # Called when a trackable, that is setup to notify this root,
        # has an attribute SET. When this attribute was itself a
        # trackable, and/or is a trackable, we must update the complete
        # tree behind that trackable (to the extend that we track it).

        is_trackable = 0
        is_trackable |= 1 * isinstance(old_value, Trackable)
        is_trackable |= 2 * isinstance(new_value, Trackable)

        # Get name and labels
        name = id, key
        labels = self._trackable_names.get(name, None)

        if is_trackable:
            # Register / unregister stores
            if is_trackable & 1:
                for label in labels:
                    self._untrack_store(old_value._store, label)
            if is_trackable & 2:
                for label in labels:
                    self._track_store(new_value._store, label)
            # If this was and is a trackable, we only process labels starting with "!"
            if is_trackable == 3:
                labels = set(label for label in labels if label.startswith("!"))

            # Follow the hierarchy
            self._track_set_follow_tree(old_value, new_value)

        # Update changed values
        if labels:
            self._track_value_update(name, new_value, labels)

    def _track_set_follow_tree(self, old_value, new_value):
        def _get_old_tree(trackable, basename):
            known_names_by_path = {}
            id = trackable._store["_trackable_id"]
            if id in id_map:
                for name in id_map[id]:
                    _, key = name  # _ == id
                    try:
                        value = trackable._store[key]
                    except KeyError:
                        continue
                    pathname = basename + (key,)
                    known_names_by_path[pathname] = name, value
                    if isinstance(value, Trackable):
                        known_names_by_path.update(_get_old_tree(value, pathname))
            return known_names_by_path

        # Create idmap for all names that we track
        id_map = {}
        for name2 in self._trackable_names.keys():
            id_map.setdefault(name2[0], set()).add(name2)

        # Get all paths in the hierarchy, mapped to the (id, prop) names
        old_stuff = {}
        if isinstance(old_value, Trackable):
            old_stuff |= _get_old_tree(old_value, ())

        for pathname in old_stuff:
            name1, value1 = old_stuff[pathname]

            # Try get matching path in the new hierarchy
            value2 = undefined
            ob = new_value
            name2 = name1
            for subname in pathname:
                try:
                    name2 = ob._store["_trackable_id"], subname
                    ob = ob._store[subname]
                except (AttributeError, KeyError):
                    break
            else:
                value2 = ob

            if value2 is not undefined:
                # Found it!

                # Rename
                self._trackable_names[name2] = self._trackable_names.pop(name1)
                try:
                    self._trackable_values[name2] = self._trackable_values.pop(name1)
                except KeyError:
                    pass
                try:
                    self._trackable_changed[name2] = self._trackable_changed.pop(name1)
                except KeyError:
                    pass

                is_trackable = 0
                is_trackable |= 1 * isinstance(value1, Trackable)
                is_trackable |= 2 * isinstance(value2, Trackable)

                labels = self._trackable_names.get(name2, ())

                if is_trackable:
                    # Register / unregister stores
                    if is_trackable & 1:
                        for label in labels:
                            self._untrack_store(value1._store, label)
                    if is_trackable & 2:
                        for label in labels:
                            self._track_store(value2._store, label)
                    # If this was and is a trackable, we only process labels starting with "!"
                    if is_trackable == 3:
                        labels = set(label for label in labels if label.startswith("!"))

                self._track_value_update(name2, value2, labels)

            else:
                # Unregister
                labels = self._trackable_names.pop(name1, ())
                self._trackable_values.pop(name1, None)
                dirty_labels_for_this_name = self._trackable_changed.setdefault(
                    name1, set()
                )
                dirty_labels_for_this_name.update(labels)
                # Unregister stores
                if isinstance(value1, Trackable):
                    for label in labels:
                        self._untrack_store(value1._store, label)

    def _track_value_update(self, name, new_value, labels):
        # Mark changed to begin with
        dirty_labels_for_this_name = self._trackable_changed.setdefault(name, set())
        dirty_labels_for_this_name.update(labels)
        # If the value has not changed from the previous (since the
        # last reset) we can un-mark this name as changed.
        try:
            ref_value, cur_value = self._trackable_values[name]
        except KeyError:
            pass
        else:
            if new_value == ref_value:
                dirty_labels_for_this_name.difference_update(labels)
                if not dirty_labels_for_this_name:
                    self._trackable_changed.pop(name)
            elif new_value != cur_value:
                self._trackable_values[name] = ref_value, new_value
