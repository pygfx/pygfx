from collections import defaultdict
from enum import Enum
from time import perf_counter_ns
from typing import Union


class EventType(str, Enum):
    # Keyboard
    KEY_DOWN = "key_down"
    KEY_UP = "key_up"
    # Pointer
    POINTER_DOWN = "pointer_down"
    POINTER_MOVE = "pointer_move"
    POINTER_UP = "pointer_up"
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
        **kwargs,
    ):
        self._type = type
        # Using perf_counter_ns instead of perf_counter
        # should give us a bit more accuracy
        self._time_stamp = time_stamp or perf_counter_ns() / 1000000
        self._bubbles = bubbles
        self._target = target
        self._current_target = target
        self._cancelled = False
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

    def __getitem__(self, key):
        """Make the extra kwargs available directly on the event object through
        bracket syntax."""
        return self._data[key]

    def __repr__(self):
        prefix = f"<{type(self).__name__}({self.type}) "
        attrs = [
            f"{key}={getattr(self, key)}"
            for key in dir(self)
            if not key.startswith("_")
            and key
            not in [
                "stop_propagation",
                "cancel",
                "type",
            ]
        ]
        attrs.extend([f"{key}={val}" for key, val in self._data.items()])
        middle = ", ".join(attrs)
        suffix = ">"
        return "".join([prefix, middle, suffix])


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
        # TODO: add support for pointer_id to wgpu-py
        self.pointer_id = pointer_id


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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def dispatch_event(self, event: dict):
        # Check for captured pointer events
        pointer_id = getattr(event, "pointer_id", None)
        if pointer_id is not None and pointer_id in EventTarget.pointer_captures:
            captured_target = EventTarget.pointer_captures[pointer_id]
            # Set the target to be the captured target
            event._target = captured_target
            captured_target.handle_event(event)
            if event.type == EventType.POINTER_UP:
                captured_target.release_pointer_capture(pointer_id)
            # Encountered an event with pointer_id while there is a
            # capture active, so don't bubble, just return immediately
            return

        target = event.target
        while target and target is not self:
            # Update the private current target field
            event._current_target = target
            target.handle_event(event)
            if pointer_id is not None and pointer_id in EventTarget.pointer_captures:
                # Prevent people from shooting in their foot by calling set_pointer_capture
                # on POINTER_UP events
                if event.type == EventType.POINTER_UP:
                    captured_target.release_pointer_capture(pointer_id)
                else:
                    # Apparently ``set_pointer_capture`` was called with this
                    # event.pointer_id, so return immediately
                    return
            if not event.bubbles or event.cancelled:
                break
            target = getattr(target, "parent", None)

        if event.bubbles:
            # Let the renderer as the virtual event root handle the event
            self.handle_event(event)
