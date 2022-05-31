"""
Implements the base classes for trackable classes.

This is implemented in a more or less generic way. The code does not
refer to world objects, geometry, materials or resources. But don't be
fooled - this code is *very* much tailored to the needs of such objects.
In particular:

* we want to track properties (that may need to trigger a pipeline).
* we want to be able to swap out similar objects and not trigger of the
  new object has the same properties as before (so that we can e.g. replace
  colormaps or texture atlasses).
* we want to be able to track objects themselves depending on the situation
  (because the pipeline only cares about the object's properties while the
  bindings much match to the correct buffers/textures).

"""

import weakref
import threading


global_id_counter = 0
global_lock = threading.RLock()
global_context = None


class TrackContext:
    """A context used when tracking usage of trackable objects."""

    def __init__(self, root, level, include_trackables):
        assert isinstance(root, RootTrackable)
        self.root = root
        self.level = level
        self.include_trackables = include_trackables

    def __enter__(self):
        global global_context
        global_lock.acquire()
        global_context = self
        self.root._clear_trackable_data(self.level)
        return None

    def __exit__(self, value, type, tb):
        global global_context
        self.root = None
        self.level = 0
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
            self["_trackable_id"] = f"t{global_id_counter}"
        # Store what roots are interested in our values
        self["_trackable_roots"] = weakref.WeakSet()

    def __setattr__(self, key, value):
        self[key] = value
        id = self["_trackable_id"]
        for root in self["_trackable_roots"]:
            root._track_set(id, key, value)

    def __getattribute__(self, key):
        value = None
        try:
            value = self[key]
            return value
        except KeyError:
            raise AttributeError(key) from None
        finally:
            if global_context:
                root = global_context.root
                id = self["_trackable_id"]
                root._track_get(id, key, value, global_context.level)
                self["_trackable_roots"].add(root)


class Trackable:
    """A base class to make an object trackable."""

    def __init__(self):
        self._store = Store()


class RootTrackable(Trackable):
    """Base class for the root trackable object. This is where the actual
    reaction data is stored.
    """

    def __init__(self):
        super().__init__()

        # === Keep track of the "tree of dependencies"

        # Keep track of trackables this root depends on  -  name -> trackable
        self._trackable_deps = weakref.WeakValueDictionary()
        # A set of the keys for performance
        self._trackable_deps_keys = set()
        # A lookup to keep track of "path names"  -   id -> name.sub.foo
        self._trackable_ids = {self._store["_trackable_id"]: ""}

        # === Keep track of changes

        # The names to track  -  name -> levels
        self._trackable_names = {}
        # Keep track of values  -  name -> (ref_value, cur_value)
        self._trackable_values = {}
        # Keep track of what has changed  -  name -> levels
        self._trackable_changed = {}

    def track_usage(self, level, include_trackables):
        """Used to track the usage of attributes. The result of this method
        should be used as a context manager. This method must only be
        used by the one system that is tracking this object's changes.
        """
        return TrackContext(self, level, include_trackables)

    def pop_changed(self):
        """Get a set of levels for which the object (and its child
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
        # Collect changed level
        result = set()
        for levels in self._trackable_changed.values():
            result.update(levels)
        # Reset and return
        self._trackable_changed = {}
        return result

    def _clear_trackable_data(self, level):
        # Reset the tracking data for a given level
        for name in list(self._trackable_names):
            s = self._trackable_names[name]
            s.discard(level)
            if not s:
                self._trackable_names.pop(name)
                self._trackable_values.pop(name, None)

    def _track_a_trackable(self, name, trackable):
        id = trackable._store["_trackable_id"]
        self._trackable_ids[id] = name
        self._trackable_deps[name] = trackable
        self._trackable_deps_keys.add(name)
        trackable._store["_trackable_roots"].add(self)
        for key, value in dict.items(trackable._store):
            if isinstance(value, Trackable):
                self._track_a_trackable(name + "." + key, value)

    def _untrack_a_trackable(self, name):
        names_to_drop = [n for n in self._trackable_deps_keys if n.startswith(name)]
        for name2 in names_to_drop:
            self._trackable_deps_keys.discard(name2)
            trackable = self._trackable_deps.pop(name2, None)
            if trackable:
                id = trackable._store["_trackable_id"]
                self._trackable_ids.pop(id, None)
                trackable._store["_trackable_roots"].discard(self)

    def _track_get(self, id, key, value, level):
        # Called when *any* trackable has an attribute GET while a
        # global context is active. The trackable can be in the tree
        # of the root (a (grand)child), but can also be part of a
        # detached tree. In both cases we must track the "path names".
        name = self._trackable_ids.get(id, id) + "." + key

        if isinstance(value, Trackable):
            self._track_a_trackable(name, value)

        self._trackable_names.setdefault(name, set()).add(level)
        if isinstance(value, simple_value_types):
            self._trackable_values[name] = value, value
        else:
            self._trackable_values.pop(name, None)
        return key

    def _track_set(self, id, key, value):
        # Called when a trackable, that is setup to notify this root,
        # has an attribute SET. When this attribute was itself a
        # trackable, and/or is a trackable, we must update the complete
        # tree behind that trackable (to the extend that we track it).

        name = self._trackable_ids.get(id, id) + "." + key
        is_trackable = False

        # If this attribute *was* a trackable, unregister it, and all things down its tree
        if name in self._trackable_deps_keys:
            is_trackable = True
            self._untrack_a_trackable(name)

        # If the new value *is* a trackable, register it, and (part of) its tree
        if isinstance(value, Trackable):
            is_trackable = True
            self._track_a_trackable(name, value)

        # If this is/was a trackable, check sub-props
        if is_trackable:
            prefix = name + "."
            sub_count = 0
            for name2 in self._trackable_names.keys():
                if name2.startswith(prefix):
                    if name2 not in self._trackable_deps_keys:
                        sub_count += 1
                        self._track_set_trackable_attribute(prefix, value, name2)
            # If the sub is/was a trackable, and we had sub-props of it,
            # we assume it was about these subprops. But if it had not,
            # it could have been a check for an attribute's presence.
            if sub_count:
                return

        # Should setting this name register as a change?
        levels = self._trackable_names.get(name, None)

        # Update changed values
        if levels:
            # Mark changed
            dirty_levels_for_this_name = self._trackable_changed.setdefault(name, set())
            dirty_levels_for_this_name.update(levels)
            # If the value has not changed from the previous (since the
            # last reset) we can un-mark this name as changed.
            try:
                ref_value, cur_value = self._trackable_values[name]
            except KeyError:
                pass
            else:
                if value == ref_value:
                    dirty_levels_for_this_name.difference_update(levels)
                    if not dirty_levels_for_this_name:
                        self._trackable_changed.pop(name)
                elif value != cur_value:
                    self._trackable_values[name] = ref_value, value

    def _track_set_trackable_attribute(self, prefix, trackable, name):
        # Track_set when a trackable is set, so we can inspect the sub-values.
        levels = self._trackable_names[name]
        # Let's assume the value has changed for now
        dirty_levels_for_this_name = self._trackable_changed.setdefault(name, set())
        dirty_levels_for_this_name.update(levels)
        # Check if we have a previous value. If not, return.
        try:
            ref_value, cur_value = self._trackable_values[name]
        except KeyError:
            return
        # Try to obtain the new value. If we fail, we return.
        ob = trackable
        for subname in name[len(prefix) :].split("."):
            try:
                ob = ob._store[subname]
            except (AttributeError, KeyError):
                return
        new_value = ob
        # Update
        self._trackable_values[name] = ref_value, new_value
        # Maybe unset this change
        if new_value == ref_value:
            dirty_levels_for_this_name.difference_update(levels)
            if not dirty_levels_for_this_name:
                self._trackable_changed.pop(name)


simple_value_types = None.__class__, bool, int, float, str

valid_types = simple_value_types  # + Trackable
# todo: allow adding color to it?
