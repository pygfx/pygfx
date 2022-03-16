from collections import defaultdict
from enum import Enum
from time import perf_counter_ns
from typing import Union
from weakref import ref, WeakValueDictionary


CLICK_DEBOUNCE = 500  # in milliseconds


class EventType(str, Enum):
    # Keyboard
    KEY_DOWN = "key_down"
    KEY_UP = "key_up"
    # Pointer
    POINTER_DOWN = "pointer_down"
    POINTER_MOVE = "pointer_move"
    POINTER_UP = "pointer_up"
    CLICK = "click"
    DOUBLE_CLICK = "double_click"
    # Wheel
    WHEEL = "wheel"
    # Window
    CLOSE = "close"
    RESIZE = "resize"


class Event:
    """Event base class for creating events.

    If a target is set, an event can bubble up through a hierarchy
    of targets, connected through a ``parent`` property.
    To prevent an event from bubbling up, use ``stop_propagation``.

    It is also possible to cancel events, which will stop any further
    handling of the event (also by the same target).

    Custom events can have any fields. Any unknown ``kwargs`` will be
    captured and can be later retrieved with the square bracket notation.
    """

    def __init__(
        self,
        type: Union[str, EventType],
        *,
        bubbles=True,
        target: "EventTarget" = None,
        time_stamp: float = None,
        cancelled: bool = False,
        **kwargs,
    ):
        self._type = type
        # Using perf_counter_ns instead of perf_counter
        # should give us a bit more accuracy
        self._time_stamp = time_stamp or perf_counter_ns() / 1000000
        self._bubbles = bubbles
        self._target = target
        self._current_target = target
        self._cancelled = cancelled
        # Save extra kwargs to be able to look
        # them up later with `__getitem__`
        self._data = kwargs

    @property
    def type(self) -> str:
        """A string representing the name of the event."""
        return self._type

    @property
    def time_stamp(self) -> float:
        """The time at which the event was created (in milliseconds). Might not be
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

    def __getitem__(self, key):
        """Make the extra kwargs available directly on the event object through
        bracket syntax."""
        return self._data[key]


class KeyboardEvent(Event):
    def __init__(self, *args, key, modifiers=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.key = key
        self.modifiers = modifiers or ()


class PointerEvent(Event):
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
        self.clicks = clicks
        self.pointer_id = pointer_id

    def copy(self, type, clicks):
        result = PointerEvent(
            type=type,
            x=self.x,
            y=self.y,
            button=self.button,
            buttons=self.buttons,
            modifiers=self.modifiers,
            ntouches=self.ntouches,
            touches=self.touches,
            clicks=clicks,
            pointer_id=self.pointer_id,
            bubbles=self.bubbles,
            target=self.target,
            cancelled=self.cancelled,
            time_stamp=self.time_stamp,
            # **self._data,
        )
        result._data = self._data.copy()
        return result


class WheelEvent(Event):
    def __init__(self, *args, dx, dy, x, y, modifiers=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.dx = dx
        self.dy = dy
        self.x = x
        self.y = y
        self.modifiers = modifiers or ()


class WindowEvent(Event):
    def __init__(self, *args, width=None, height=None, pixel_ratio=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.width = width
        self.height = height
        self.pixel_ratio = pixel_ratio


class EventTarget:
    """Mixin class that enables event handlers to be attached to objects
    of the mixed-in class.
    """

    pointer_captures = WeakValueDictionary()

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
        https://jupyter-rfb.readthedocs.io/en/latest/events.html

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

        Subclasses can overload this method. Events include widget
        resize, mouse/touch interaction, key events, and more. An event
        is a dict with at least the key event_type. For details, see
        https://jupyter-rfb.readthedocs.io/en/latest/events.html
        """
        event_type = event.type
        for callback in self._event_handlers[event_type]:
            if event.cancelled:
                break
            callback(event)

    def set_pointer_capture(self, pointer_id):
        """Register this object to capture any other pointer events,
        until ``release_pointer_capture`` is called or an ``pointer_up``
        event is encountered.
        """
        EventTarget.pointer_captures[pointer_id] = self

    def release_pointer_capture(self, pointer_id):
        """Release the pointer capture for the object that was registered
        to the given pointer_id.
        """
        if pointer_id in EventTarget.pointer_captures:
            del EventTarget.pointer_captures[pointer_id]


class RootEventHandler(EventTarget):
    """Root event handler for the Pygfx event system."""

    click_tracker = {}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def dispatch_event(self, event: dict):
        pointer_id = getattr(event, "pointer_id", None)

        # Check for captured pointer events
        if pointer_id is not None and pointer_id in EventTarget.pointer_captures:
            captured_target = EventTarget.pointer_captures.get(pointer_id)
            # Encountered an event with pointer_id while there is a
            # capture active, so don't bubble, and retarget the event
            # to the captured target
            event._retarget(captured_target)
            event.stop_propagation()

        target = event.target
        while target and target is not self:
            # Update the current target
            event._update_current_target(target)
            target.handle_event(event)
            if pointer_id is not None and pointer_id in EventTarget.pointer_captures:
                event._retarget(target)
                event.stop_propagation()
                if event.type == EventType.POINTER_UP:
                    captured_target.release_pointer_capture(pointer_id)
            if not event.bubbles or event.cancelled:
                break
            target = target.parent

        if event.bubbles:
            # Let the renderer as the virtual event root handle the event
            self.handle_event(event)

        if event.type == EventType.POINTER_DOWN:
            tracked_click = RootEventHandler.click_tracker.get(pointer_id)
            if (
                tracked_click
                and (
                    tracked_click["target"]
                    and tracked_click["target"]() is event.target
                    or (not tracked_click["target"] and not event.target)
                )
                and event.time_stamp - tracked_click["time_stamp"] < CLICK_DEBOUNCE
            ):
                tracked_click["count"] += 1
                tracked_click["time_stamp"] = event.time_stamp
            else:
                RootEventHandler.click_tracker[pointer_id] = {
                    "count": 1,
                    "time_stamp": event.time_stamp,
                    "target": (event.target and ref(event.target)) or None,
                }

        elif event.type == EventType.POINTER_UP:
            tracked_click = RootEventHandler.click_tracker.get(pointer_id)
            if tracked_click and (
                tracked_click["target"]
                and tracked_click["target"]() is event.target
                or (not tracked_click["target"] and not event.target)
            ):
                ev = event.copy("click", tracked_click["count"])
                self.dispatch_event(ev)
                if tracked_click["count"] == 2:
                    double_ev = event.copy("double_click", tracked_click["count"])
                    self.dispatch_event(double_ev)
