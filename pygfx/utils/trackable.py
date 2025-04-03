"""
Implements the base classes for trackable classes.

## Introduction

This module is implemented in a more or less generic way. The code does not
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
* we want to be able to track multiple trees independently.

The last two points mean that the code must be aware of the tree
structures that trackables make up.

## High level overview

We define a Trackable object as something of which the properties can
be tracked (its properties can be other Trackable objects).

The PropTracker is the object that keeps track of the changes. One
first tracks usage of properties, after which any changes to these
properties get communicated to the tracker.

## Finer details

The Trackable uses a Store - a custom dictionary object that has hooks
for attribute access. This is where the magic happens. The Trackable
object itself is just a container for the store. This has the benefit
that subclasses of Trackable just store the props that they want to
track on its store.

The Store keeps a set of what trackers need to be notified (using weak
refs). The tracker keeps a set of what stores notify it, for the sole
reason that the tracker can disconnect these stores when usage is
re-tracked.

Every store has a unique id, and the tracker uses (id, prop) tuples as
keys to keep track of things. When a property sets/unsets a Trackable,
the tree is resolved at that moment (i.e. the tracker is temporary aware
of the tree). This avoids the need for the tracker (or stores?) to keep
track of a hierarchy which would make the code a lot more complex (I
tried and it was not fun). The only downside (that I can think of) is
that setting `matererial.map = other_map` works while
`matererial.map = None; material.map = other_map` does not, because
there is no persistent knowledge of hierarchy.

"""

import threading
import weakref

global_id_counter = 0
global_lock = threading.RLock()
global_context = None

simple_value_types = None.__class__, bool, int, float, str


def get_comp_value(value):
    if isinstance(value, simple_value_types):
        return value
    elif isinstance(value, tuple):
        return tuple(get_comp_value(v) for v in value)
    else:
        try:
            return "hash:" + str(hash(value))
        except TypeError:
            return "id:" + str(id(value))


class Undefined:
    def __repr__(self):
        return "undefined"


undefined = Undefined()


class Trackable:
    """A base class to make an object trackable."""

    def __init__(self):
        self._store = Store()


class TrackContext:
    """A context used when tracking usage of trackable objects."""

    def __init__(self, tracker, label):
        assert isinstance(tracker, PropTracker)
        assert isinstance(label, str)
        self.tracker = tracker
        self.label = label

    def __enter__(self):
        global global_context
        global_lock.acquire()
        global_context = self
        self.tracker._track_init(self.label)
        return None

    def __exit__(self, value, type, tb):
        global global_context
        self.tracker._track_done(self.label)
        self.tracker = None
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
        # Store what trackers we need to notify of changes
        self["_trackable_trackers"] = weakref.WeakSet()

    def __hash__(self):
        return self["_trackable_id"]

    def __setattr__(self, key, new_value):
        # Apply
        old_value = dict.get(self, key, undefined)
        self[key] = new_value
        # Notify the trackers
        id = self["_trackable_id"]
        for tracker in self["_trackable_trackers"]:
            tracker._track_set((id, key), old_value, new_value)

    def __getattribute__(self, key):
        if key.startswith("__"):
            return dict.__getattribute__(self, key)
        value = undefined
        try:
            value = self[key]
            return value
        except KeyError:
            return dict.__getattribute__(self, key)
        finally:
            if global_context:
                global_context.tracker._track_get(
                    self, key, value, global_context.label
                )


class PropTracker:
    """Object to store all the tracking data."""

    def __init__(self):
        # Keep track of what stores have a reference to *this*,
        # so that we can clear that reference when we want.
        # The store and tracker are always connected/disconnected together.
        self._stores = weakref.WeakKeyDictionary()  # store -> labels
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
        if not self._trackable_changed:
            return set()

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
        return set(label.lstrip("!") for label in result)

    def _track_store(self, store, label):
        # Introduce the store and tracker to each-other
        store["_trackable_trackers"].add(self)
        labels = self._stores.setdefault(store, set())
        labels.add(label)

    def _untrack_store(self, store, label):
        # Break the connection between tracker and store
        labels = self._stores.setdefault(store, set())
        labels.discard(label)
        if not labels:
            self._stores.pop(store, None)
            store["_trackable_trackers"].discard(self)

    def _track_init(self, label):
        # Reset the tracking data for the given label
        for name in list(self._trackable_names):
            labels = self._trackable_names[name]
            labels.discard(label)
            if not labels:
                self._trackable_names.pop(name)
                self._trackable_values.pop(name)

        # Break connections with old stores
        to_remove = []
        for store, labels in self._stores.items():
            labels.discard(label)
            if not labels:
                to_remove.append(store)
        for store in to_remove:
            self._stores.pop(store, None)
            store["_trackable_trackers"].discard(self)

    def _track_done(self, label):
        pass

    def _track_get(self, store, key, value, label):
        # Called when *any* trackable has an attribute GET while a
        # global context is active. The trackable can be in the tree
        # of the tracker (a (grand)child), but can also be part of a
        # detached tree.

        self._track_store(store, label)

        id = store["_trackable_id"]
        name = id, key

        self._trackable_names.setdefault(name, set()).add(label)
        comp_value = get_comp_value(value)
        self._trackable_values[name] = comp_value, comp_value

    def _track_set(self, name, old_value, new_value):
        # Called when a trackable, that is setup to notify this tracker,
        # has an attribute SET. When this attribute was itself a
        # trackable, and/or is a trackable, we must update the complete
        # tree behind that trackable (to the extend that we track it).

        is_trackable = 0
        old_store = new_store = None
        if isinstance(old_value, (Store, Trackable)):
            is_trackable |= 1
            old_store = old_value if isinstance(old_value, Store) else old_value._store
        if isinstance(new_value, (Store, Trackable)):
            is_trackable |= 2
            new_store = new_value if isinstance(new_value, Store) else new_value._store

        # Get labels
        labels = self._trackable_names.get(name, None)

        if is_trackable and labels:
            # Register / unregister stores
            if is_trackable & 1:
                for label in labels:
                    self._untrack_store(old_store, label)
            if is_trackable & 2:
                for label in labels:
                    self._track_store(new_store, label)
            # If this was and is a trackable, we only process labels starting with "!"
            if is_trackable == 3 and type(old_value) is type(new_value):
                labels = set(label for label in labels if label.startswith("!"))

            # Follow the hierarchy
            if old_store is not None:
                self._track_set_follow_tree(old_store, new_store)

        # Update changed values
        if labels:
            # Mark changed to begin with
            dirty_labels_for_this_name = self._trackable_changed.setdefault(name, set())
            dirty_labels_for_this_name.update(labels)
            # If the value has not changed from the previous (since the
            # last reset) we can un-mark this name as changed.
            comp_value = get_comp_value(new_value)
            ref_value, cur_value = self._trackable_values[name]
            if comp_value == ref_value:
                dirty_labels_for_this_name.difference_update(labels)
                if not dirty_labels_for_this_name:
                    self._trackable_changed.pop(name)
            elif comp_value != cur_value:
                self._trackable_values[name] = ref_value, comp_value

    def _track_set_follow_tree(self, old_store, new_store):
        # old_store must be a store
        # new_store must be a store or None

        # Get names of stuff we track on the old trackable store
        id = old_store["_trackable_id"]
        sub_names = [n for n in self._trackable_names.keys() if n[0] == id]

        # We need to follow these
        for name1 in sub_names:
            _, key = name1
            try:
                value1 = old_store[key]
            except KeyError:
                continue

            # Try get matching path in the new hierarchy
            value2 = undefined
            if new_store is not None:
                try:
                    name2 = new_store["_trackable_id"], key
                    value2 = new_store[key]
                except (AttributeError, KeyError):
                    pass

            if value2 is not undefined:
                # Rename
                self._trackable_names[name2] = self._trackable_names.pop(name1)
                self._trackable_values[name2] = self._trackable_values.pop(name1)
                try:
                    self._trackable_changed[name2] = self._trackable_changed.pop(name1)
                except KeyError:
                    pass
                # Recurse
                self._track_set(name2, value1, value2)

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
                # Recurse
                if isinstance(value1, Store):
                    self._track_set_follow_tree(value1, None)
                elif isinstance(value1, Trackable):
                    self._track_set_follow_tree(value1._store, None)
