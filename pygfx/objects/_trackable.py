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


global_lock = threading.RLock()
global_context = None


simple_value_types = None.__class__, bool, int, float, str


class TrackContext:
    """ A context used when tracking usage of trackable objects.
    """
    def __init__(self, root, level, include_trackables):
        assert isinstance(root, RootTrackable)
        self.root = root
        self.level = level
        self.include_trackables = include_trackables
        self.new_names = set()

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


class Trackable:
    """ A base class to make an object trackable.
    """

    def __init__(self):
        # Keep track of our parents  -  parent -> names
        self._trackable_parents = weakref.WeakKeyDictionary()
        # Keep track of child trackabeles  -  name -> child
        self._trackable_children = weakref.WeakValueDictionary()

    def _track_get(self, name, value):
        """ The subclass should ideally call this in a property getter.
        """
        # Called when getting an attribute to track usage.
        if global_context:
            for parent, names in self._trackable_parents.items():
                for name_at_parent in names:
                    parent._track_get(f"{name_at_parent}.{name}", value)
        return value

    def _track_set(self, name, value):
        """ The subclass should ideally call this in a property setter.
        """
        # Called when setting an attribute, which *may* bump a change at the root.
        if "." not in name:
            self._track_trackable_children(name, value)
        for parent, names in self._trackable_parents.items():
            for name_at_parent in names:
                parent._track_set(f"{name_at_parent}.{name}", value)
        return value

    def _track_trackable_children(self, name, value):
        # Bookkeeping for removing a trackable child/attribute
        assert "." not in name
        bubble_up = False
        if name in self._trackable_children:
            # Technically this outer if-statement (above) is not necessary,
            # but it is a bit faster because it's a weakvalue dict.
            old_child = self._trackable_children.pop(name, None)
            if old_child:
                bubble_up = True
                names = old_child._trackable_parents.setdefault(self, set())
                names.discard(name)
                if not names:
                    old_child._trackable_parents.pop(self)
        # Bookkeeping for adding a trackable child/attribute
        if isinstance(value, Trackable):
            bubble_up = True
            self._trackable_children[name] = value
            names = value._trackable_parents.setdefault(self, set())
            names.add(name)
        if bubble_up:
            self._track_set_trackable(name, value)

    def _track_set_trackable(self, name, value):
        # Bubble up the setting/unsetting of a trackable object.
        for parent, names in self._trackable_parents.items():
            for name_at_parent in names:
                parent._track_set_trackable(f"{name_at_parent}.{name}", value)
        return value


class RootTrackable(Trackable):
    """ Base class for the root trackable object. This is where the actual
    reaction data is stored.
    """

    def __init__(self):
        super().__init__()
        # A root has no parents
        self._trackable_parents = None
        # The names to track  -  name -> levels
        self._trackable_names = {}
        # Keep track of values  -  name -> (ref_value, cur_value)
        self._trackable_values = {}
        # Keep track of what has changed  -  name -> levels
        self._trackable_changed = {}

    def track_usage(self, level, include_trackables):
        """ Used to track the usage of attributes. The result of this method
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
        names = list(self._trackable_changed.keys())
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

    def _track_get(self, name, value):
        # Called when getting an attribute to track usage.
        # Register the given name as a trigger.
        if global_context and global_context.root is self:
            if not (isinstance(value, Trackable) and not global_context.include_trackables):
                self._trackable_names.setdefault(name, set()).add(global_context.level)
                if isinstance(value, simple_value_types):
                    self._trackable_values[name] = value, value
        return value

    def _track_set(self, name, value):
        # Called when setting an attribute to mark the root object as changed.
        if "." not in name:
            self._track_trackable_children(name, value)
        # Mark the object as changed if needed.
        levels = self._trackable_names.get(name, None)
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
        return value

    def _track_set_trackable(self, name, value):
        # If a trackable is set, we must check its attributes.
        # Note that value can also be None, e.g. when a trackable is unset.
        prefix = name + "."
        for name2 in self._trackable_names.keys():
            if name2.startswith(prefix):
                self._track_set_trackable_attribute(prefix, value, name2)

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
        for subname in name[len(prefix):].split("."):
            try:
                ob = getattr(ob, subname)
            except AttributeError:
                return
        new_value = ob
        # Update
        self._trackable_values[name] = ref_value, new_value
        # Maybe unset this change
        if new_value == ref_value:
            dirty_levels_for_this_name.difference_update(levels)
            if not dirty_levels_for_this_name:
                self._trackable_changed.pop(name)
