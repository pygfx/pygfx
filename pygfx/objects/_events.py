from collections import defaultdict
from enum import Enum
from time import perf_counter
from typing import Union
from weakref import ref

from wgpu.gui.base import log_exception


CLICK_DEBOUNCE = 0.5  # in seconds


class EventType(str, Enum):
    # Keyboard
    KEY_DOWN = "key_down"
    KEY_UP = "key_up"
    # Pointer
    POINTER_DOWN = "pointer_down"
    POINTER_MOVE = "pointer_move"
    POINTER_UP = "pointer_up"
    POINTER_ENTER = "pointer_enter"
    POINTER_LEAVE = "pointer_leave"
    CLICK = "click"
    DOUBLE_CLICK = "double_click"
    # Wheel
    WHEEL = "wheel"
    # Window
    CLOSE = "close"
    RESIZE = "resize"


class Event:
    """Event base class.

    If a target is set, an event can bubble up through a hierarchy
    of targets, connected through a ``parent`` property.
    To prevent an event from bubbling up, use ``stop_propagation``.

    It is also possible to cancel events, which will stop any further
    handling of the event (also by the same target).

    Parameters
    ----------
    type : Union[str, EventType]
        The name of the event.
    bubbles : bool
        If True, the event bubbles up through the scene tree.
    target : EventTarget
        The object onto which the event was dispatched.
    root : RootEventHandler
        A reference to the root event handler.
    time_stamp : float
        The time at which the event was created (in seconds). Might not be an actual
        time stamp so please only use this for relative time measurements.
    cancelled : bool
        A boolean value indicating whether the event is cancelled.
    event_type : str
        Unused.

    """

    def __init__(
        self,
        type: Union[str, EventType],
        *,
        bubbles=True,
        target: "EventTarget" = None,
        root: "RootEventHandler" = None,
        time_stamp: float = None,
        cancelled: bool = False,
        # Swallow event_type to ease conversion from wgpu events to Event objects
        event_type: str = None,
    ):
        self._type = type
        # Using perf_counter_ns instead of perf_counter
        # should give us a bit more accuracy
        self._time_stamp = time_stamp or perf_counter()
        self._bubbles = bubbles
        self._target = target
        self._current_target = target
        self._cancelled = cancelled
        self._root = root

    @property
    def type(self) -> str:
        """A string representing the name of the event."""
        return self._type

    @property
    def root(self) -> "RootEventHandler":
        """A reference to the root event handler."""
        return self._root

    @property
    def time_stamp(self) -> float:
        """The time at which the event was created (in seconds). Might not be
        an actual time stamp so please only use this for relative time measurements."""
        return self._time_stamp

    @property
    def bubbles(self) -> bool:
        """A boolean value indicating whether or not the event bubbles up through
        the scene tree."""
        return self._bubbles

    @property
    def target(self) -> "EventTarget":
        """The object onto which the event was dispatched."""
        return self._target

    @property
    def current_target(self) -> "EventTarget":
        """The object that is currently handling the event. During event bubbling
        this property will be updated whenever an event bubbles up in the hierarchy."""
        return self._current_target

    @property
    def cancelled(self) -> bool:
        """A boolean value indicating whether the event is cancelled."""
        return self._cancelled

    def stop_propagation(self):
        """Stops propagation of events further along in the scene tree."""
        self._bubbles = False

    def cancel(self):
        """Cancels the event and stops propagation."""
        self._cancelled = True
        self.stop_propagation()

    def _retarget(self, target):
        self._target = target
        self._update_current_target(target)

    def _update_current_target(self, target):
        self._current_target = target


class KeyboardEvent(Event):
    """Keyboard button press.

    Parameters
    ----------
    args : Any
        Positional arguments are forwarded to the :class:`base class
        <pygfx.objects.Event>`.
    key : str
        The key that was pressed.
    modifiers : tuple
        The modifiers that were pressed while the key was pressed.
    kwargs : Any
        Additional keyword arguments are forward to the :class:`base class
        <pygfx.objects.Event>`.

    """

    def __init__(self, *args, key, modifiers=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.key = key
        self.modifiers = modifiers or ()


class PointerEvent(Event):
    """Mouse/Touch Event.

    Parameters
    ----------
    args : Any
        Positional arguments are forwarded to the :class:`base class
        <pygfx.objects.Event>`.
    x : int
        The x position of the cursor or touch in screen space (px).
    y : int
        Thy y position of the cursor or touch in screen space (px).
    button : int
        The integer value of the button being pushed.
    buttons : tuple
        The list of string name of the buttons being pushed.
    modifiers : tuple
        The modifiers that were pressed while the key was pressed.
    ntouches : int
        The total number of synchronous touches.
    touches : list
        A list of all currently occurring touches.
    pick_info : dict
        Values of pickable fields. The exact content is specific to the
        WorldObject triggering the event.
    clicks : int
        The total number of synchronous clicks.
    pointer_id : Any
        The value set by `set_pointer_capture()`.
    kwargs : Any
        Additional keyword arguments are forward to the :class:`base class
        <pygfx.objects.Event>`.

    Notes
    -----
    The values of this event follow the convention used by jupyter rfb. You can read
    about them `here <https://jupyter-rfb.readthedocs.io/en/stable/events.html>`_.

    """

    def __init__(
        self,
        *args,
        x,
        y,
        button=0,
        buttons=None,
        modifiers=None,
        ntouches=0,
        touches=None,
        pick_info=None,
        clicks=0,
        pointer_id=0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.x = x
        self.y = y
        self.button = button
        self.buttons = buttons or ()
        self.modifiers = modifiers or ()
        self.ntouches = ntouches
        self.touches = touches or {}
        self.pick_info = pick_info or {}
        self.clicks = clicks
        self.pointer_id = pointer_id

    def copy(self, **kwargs):
        values = dict(
            type=self.type,
            x=self.x,
            y=self.y,
            button=self.button,
            buttons=self.buttons,
            modifiers=self.modifiers,
            ntouches=self.ntouches,
            touches=self.touches,
            pick_info=self.pick_info,
            clicks=self.clicks,
            pointer_id=self.pointer_id,
            bubbles=self.bubbles,
            target=self.target,
            cancelled=self.cancelled,
            time_stamp=self.time_stamp,
            root=self.root,
        )
        values.update(kwargs)
        return PointerEvent(**values)


class WheelEvent(PointerEvent):
    """Scrolling of the mouse wheel.

    Parameters
    ----------
    args : Any
        Positional arguments are forwarded to the :class:`base class
        <pygfx.objects.Event>`.
    dx : float
        The amount (in rad) by which the wheel was turned around the x-axis.
    dy : float
        The amount (in rad) by which the wheel was turned around the y-axis.
    kwargs : Any
        Additional keyword arguments are forward to the :class:`base class
        <pygfx.objects.Event>`.

    """

    def __init__(self, *args, dx, dy, **kwargs):
        super().__init__(*args, **kwargs)
        self.dx = dx
        self.dy = dy


class WindowEvent(Event):
    """Window resize event

    Parameters
    ----------
    args : Any
        Positional arguments are forwarded to the :class:`base class
        <pygfx.objects.Event>`.
    width : int
        The new width of the application window in screen space (px).
    height : int
        The new height of the application window in screen space (px).
    pixel_ratio : float
        The new ratio between logical pixels and physical pixels.
    kwargs : Any
        Additional keyword arguments are forward to the :class:`base class
        <pygfx.objects.Event>`.

    """

    def __init__(self, *args, width=None, height=None, pixel_ratio=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.width = width
        self.height = height
        self.pixel_ratio = pixel_ratio


class EventTarget:
    """Targetable object mixin.

    Mixin class that enables event handlers to be attached to objects
    of the mixed-in class.

    Parameters
    ----------
    args : Any
        Arguments are forwarded to allow multiple inheritance.
    kwargs : Any
        Kwargs are forwarded to allow multiple inheritance.

    """

    pointer_captures = {}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._event_handlers = defaultdict(set)

    def add_event_handler(self, *args):
        """Register an event handler.

        Arguments:
            callback (callable): The event handler. Must accept a
                single event argument.
            *types (list of strings): A list of event types.

        For the available event types, see
        https://jupyter-rfb.readthedocs.io/en/stable/events.html

        Can also be used as a decorator.

        Example:

        .. code-block:: py

            def my_handler(event):
                print(event)

            obj.add_event_handler(my_handler, "pointer_up", "pointer_down")

        Decorator usage example:

        .. code-block:: py

            @obj.add_event_handler("pointer_up", "pointer_down")
            def my_handler(event):
                print(event)
        """

        decorating = not callable(args[0])
        callback = None if decorating else args[0]
        types = args if decorating else args[1:]

        if not types:
            raise ValueError("No types registered for callback")
        if not all(isinstance(t, str) for t in types):
            raise TypeError("All types must be string.")

        def decorator(_callback):
            for type in types:
                self._event_handlers[type].add(_callback)
            return _callback

        if decorating:
            return decorator
        return decorator(callback)

    def remove_event_handler(self, callback, *types):
        """Unregister an event handler.

        Arguments:
            callback (callable): The event handler.
            *types (list of strings): A list of event types.
        """
        for type in types:
            self._event_handlers[type].remove(callback)

    def handle_event(self, event: Event):
        """Handle an incoming event.

        Arguments:
            event: The event to handle
        """
        event_type = event.type
        for callback in self._event_handlers[event_type].copy():
            if event.cancelled:
                break
            with log_exception(f"Error during handling {event_type} event"):
                callback(event)

    def set_pointer_capture(self, pointer_id, event_root):
        """Register this object to capture any other pointer events,
        until ``release_pointer_capture`` is called or an ``pointer_up``
        event is encountered.

        Arguments:
            pointer_id: id of pointer to capture (mouse, touch, etc.)
            event_root: the event root that this pointer is captured on
        """
        EventTarget.pointer_captures[pointer_id] = (
            ref(self),
            (event_root and ref(event_root)) or None,
        )

    def release_pointer_capture(self, pointer_id):
        """Release the pointer capture for the object that was registered
        to the given pointer_id.

        Arguments:
            pointer_id: id of pointer to release (mouse, touch, etc.)
        """
        EventTarget.pointer_captures.pop(pointer_id, None)


class RootEventHandler(EventTarget):
    """Pygfx event handler.

    Root event handler for the Pygfx event system.

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Dictionary to track clicks, keyed on pointer_id
        self._click_tracker = {}
        # Dictionary to track targets, keyed on pointer_id
        self._target_tracker = {}

    def dispatch_event(self, event: Event):
        """Dispatch the given event.

        This method will dispatch an event by looking for the right target to
        handle the event. When a target is set on the event, then that target
        will be the first object that gets to handle the event. From there it
        will ask its parents one-by-one to handle the event as long as the event
        bubbles / propagates up or is not cancelled.

        The RootEventHandler object will serve as a virtual root for the tree
        hierarchy.

        Whenever an object has captured the pointer (for a specific pointer_id)
        then that object will get all pointer related events until the object
        releases the capture or a ``pointer_up`` event is encountered.

        This method will also keep track of ``pointer_down`` and ``pointer_up``
        events in order to generate and dispatch ``click`` and ``double_click``
        events.

        Parameters
        ----------
        event : Event
            The event to dispatch.

        """
        pointer_id = getattr(event, "pointer_id", None)

        # Check for captured pointer events
        if pointer_id is not None and pointer_id in EventTarget.pointer_captures:
            captured_target_ref, event_root_ref = EventTarget.pointer_captures[
                pointer_id
            ]
            captured_target, event_root = (
                captured_target_ref(),
                event_root_ref and event_root_ref(),
            )
            # If the pointer was captured in the context of another root event
            # handler, then let's not handle this event. It will be handled by
            # the appropriate RootEventHandler
            if event_root and event_root is not self:
                return

            if captured_target:
                # Encountered an event with pointer_id while there is a
                # capture active, so don't bubble, and retarget the event
                # to the captured target
                event._retarget(captured_target)
                event.stop_propagation()

        # Current target is either something that was under the pointer, or nothing
        # in which case we set the target to the root event handler (self)
        target = event.target or self

        # Update the target tracker on all `pointer_move` events
        if event.type == EventType.POINTER_MOVE:
            # Get the previous target for this pointer (if any)
            previous_target_ref = self._target_tracker.get(pointer_id)
            previous_target = (previous_target_ref and previous_target_ref()) or None
            # Check if the target has changed since the previous move event
            if previous_target is not target:
                # Update the current target for this pointer
                self._target_tracker[pointer_id] = (target and ref(target)) or None
                if previous_target is not None:
                    # Dispatch a `pointer_leave` event for the previous target
                    ev = event.copy(type="pointer_leave", target=previous_target)
                    self.dispatch_event(ev)
                # Dispatch a `pointer_enter` event for the new target
                ev = event.copy(type="pointer_enter")
                self.dispatch_event(ev)

        # Dispatch the event to target and bubble up the hierarchy
        while target:
            # Update the current target
            event._update_current_target(target)
            target.handle_event(event)
            # During handling of the event, the target might capture the pointer events
            if pointer_id is not None and pointer_id in EventTarget.pointer_captures:
                event._retarget(target)
                event.stop_propagation()
                if event.type == EventType.POINTER_UP:
                    captured_target.release_pointer_capture(pointer_id)
            if not event.bubbles or event.cancelled or target is self:
                break
            target = target.parent or self

        # Update the click tracker on all `pointer_down` events
        if event.type == EventType.POINTER_DOWN:
            tracked_click = self._click_tracker.get(pointer_id)
            # Check if the `pointer_id` is already tracked, targets
            # the same target and is within the DEBOUNCE time.
            # Bump the count and update the time_stamp if that is the case.
            # Otherwise, restart counting.
            if (
                tracked_click
                and (
                    tracked_click["target"] is not None
                    and tracked_click["target"]() is not None
                    and tracked_click["target"]() is event.target
                    or (tracked_click["target"] is None and event.target is None)
                )
                and event.time_stamp - tracked_click["time_stamp"] < CLICK_DEBOUNCE
            ):
                tracked_click["count"] += 1
                tracked_click["time_stamp"] = event.time_stamp
            else:
                self._click_tracker[pointer_id] = {
                    "count": 1,
                    "time_stamp": event.time_stamp,
                    "target": (event.target and ref(event.target)) or None,
                }
        # On all ``pointer_up`` events, see if the event is on the same target
        # as for the ``pointer_down``. If so, then a ``click`` event is dispatched.
        # When the counter for the click is at 2, then a ``double_click`` event
        # is dispatched.
        elif event.type == EventType.POINTER_UP:
            tracked_click = self._click_tracker.get(pointer_id)
            if tracked_click and (
                tracked_click["target"] is not None
                and tracked_click["target"]() is not None
                and tracked_click["target"]() is event.target
                or (tracked_click["target"] is None and event.target is None)
            ):
                ev = event.copy(type="click", clicks=tracked_click["count"])
                self.dispatch_event(ev)
                if tracked_click["count"] == 2:
                    double_ev = event.copy(
                        type="double_click", clicks=tracked_click["count"]
                    )
                    self.dispatch_event(double_ev)
