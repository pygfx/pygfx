from collections import defaultdict
from itertools import count
from time import perf_counter


pointer_id_iter = count()
current_pointer_id = None


def create_event(type: str, *, propagate=True, target=None, **kwargs):
    event = {"type": type, **kwargs}

    if "pointer" in type:
        global current_pointer_id
        global pointer_id_iter
        if type == "pointer_down" or current_pointer_id is None:
            current_pointer_id = next(pointer_id_iter)
        event["pointer_id"] = current_pointer_id

    # Time stamp in seconds
    event["time_stamp"] = perf_counter()
    # Whether the event should propagate
    event["propagate"] = propagate
    # target object on which the event is dispatched
    event["target"] = target

    return event


class EventTarget:
    """Mixin class for objects that need to handle events."""

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

        For the available events, see
        https://jupyter-rfb.readthedocs.io/en/latest/events.html

        Can also be used as a decorator.

        Example:

        .. code-block:: py

            def my_handler(event):
                print(event)

            canvas.add_event_handler(my_handler, "pointer_up", "pointer_down")

        Decorator usage example:

        .. code-block:: py

            @canvas.add_event_handler("pointer_up", "pointer_down")
            def my_handler(event):
                print(event)
        """
        decorating = not callable(args[0])
        callback = None if decorating else args[0]
        types = args if decorating else args[1:]

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

    def handle_event(self, event: dict):
        """Dispatches an event to this EventTarget."""
        for callback in self._event_handlers[event["type"]]:
            callback(event)

    def set_pointer_capture(self, pointer_id):
        """Register this object to capture any other pointer events,
        until ``release_pointer_capture`` is called or an ``pointer_up``
        event is encountered.
        """
        self.pointer_captures[pointer_id] = self

    def release_pointer_capture(self, pointer_id):
        """Release the pointer capture for the object that was registered
        to the given pointer_id.
        """
        if pointer_id in self.pointer_captures:
            del self.pointer_captures[pointer_id]


class RootHandler(EventTarget):
    def handle_event(self, event: dict):
        target = event["target"]
        while target and target is not self:
            pointer_id = event.get("pointer_id")
            if pointer_id and pointer_id in EventTarget.pointer_captures:
                capture = EventTarget.pointer_captures[pointer_id]
                capture.handle_event(event)
                if event["type"] == "pointer_up":
                    capture.release_pointer_capture(pointer_id)
                # Encountered an event with pointer_id while there is a
                # capture active, so don't propagate, just return immediately
                return
            else:
                target.handle_event(event)
                if pointer_id and pointer_id in EventTarget.pointer_captures:
                    # Apparently ``set_pointer_capture`` was called with this
                    # event["pointer_id"], so return immediately
                    return
            if not event["propagate"]:
                break
            target = getattr(target, "parent", None)

        # The root handler itself is not part of the hierarchy
        # so we'll handle the event separately
        if not event["target"] or event["propagate"]:
            super().handle_event(event)
