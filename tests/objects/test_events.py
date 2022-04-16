from pygfx import Scene, WorldObject
from pygfx.objects._events import EventTarget, Event, PointerEvent, RootEventHandler


class Node(EventTarget):
    def __init__(self, parent=None):
        super().__init__()
        self.parent = parent


def test_event_target():
    c = EventTarget()

    # It's event handling mechanism should be fully functional
    events = []

    def handler(event):
        events.append(event.value)

    def event_with_val(type, value):
        ev = Event(type)
        ev.value = value
        return ev

    c.add_event_handler(handler, "foo", "bar")
    c.add_event_handler(handler, "bar")
    c.handle_event(event_with_val(type="foo", value=1))
    c.handle_event(event_with_val(type="bar", value=2))
    c.handle_event(event_with_val(type="spam", value=3))
    c.remove_event_handler(handler, "foo")
    c.handle_event(event_with_val(type="foo", value=4))
    c.handle_event(event_with_val(type="bar", value=5))
    c.handle_event(event_with_val(type="spam", value=6))
    c.remove_event_handler(handler, "bar")
    c.handle_event(event_with_val(type="foo", value=7))
    c.handle_event(event_with_val(type="bar", value=8))
    c.handle_event(event_with_val(type="spam", value=9))

    assert events == [1, 2, 5]


def test_event_bubbling():
    """
    Check that events bubble up in hierarchy
    """
    root_called = 0
    scene_called = 0
    item_called = 0

    def root_callback(event):
        nonlocal root_called
        root_called += 1

    def scene_callback(event):
        nonlocal scene_called
        scene_called += 1

    def item_callback(event):
        nonlocal item_called
        item_called += 1

    root_handler = RootEventHandler()
    scene = Scene()
    item = WorldObject()
    scene.add(item)

    root_handler.add_event_handler(root_callback, "foo")
    scene.add_event_handler(scene_callback, "foo")
    item.add_event_handler(item_callback, "foo")

    event = Event(type="foo", target=item)
    root_handler.dispatch_event(event)

    assert item_called == 1
    assert scene_called == 1
    assert root_called == 1


def test_event_stop_propagation():
    """
    Check that bubbling stops when stop_propagation
    is called on an event.
    """
    root_called = 0
    child_called = 0

    def root_callback(event):
        nonlocal root_called
        root_called += 1

    def child_callback(event):
        nonlocal child_called
        child_called += 1

    def child_prevent_callback(event):
        nonlocal child_called
        # Prevent bubbling up
        event.stop_propagation()
        child_called += 1

    root = Node()
    child = Node(parent=root)

    root_handler = RootEventHandler()
    root.add_event_handler(root_callback, "foo")
    child.add_event_handler(child_callback, "foo")

    event = {"type": "foo", "target": child}
    root_handler.dispatch_event(Event(**event))

    assert child_called == 1
    assert root_called == 1

    child.remove_event_handler(child_callback, "foo")
    child.add_event_handler(child_prevent_callback, "foo")

    root_handler.dispatch_event(Event(**event))

    assert child_called == 2
    assert root_called == 1


def test_event_propagation():
    root_called = 0
    child_called = 0

    def root_callback(event):
        nonlocal root_called
        root_called += 1

    def child_callback(event):
        nonlocal child_called
        child_called += 1

    root = Node()
    child = Node(parent=root)
    root.add_event_handler(root_callback, "foo")
    child.add_event_handler(child_callback, "foo")

    root_handler = RootEventHandler()

    event = Event(type="foo", target=child, bubbles=True)
    root_handler.dispatch_event(event)

    assert child_called == 1
    assert root_called == 1

    event = Event(type="foo", target=child, bubbles=False)
    root_handler.dispatch_event(event)

    assert child_called == 2
    assert root_called == 1


def test_pointer_event_capture():
    root_called = 0
    child_called = 0

    def root_callback(event):
        nonlocal root_called
        root_called += 1

    def child_callback(event):
        nonlocal child_called
        child_called += 1

    root = Node()
    child = Node(parent=root)

    root.add_event_handler(
        root_callback, "pointer_down", "pointer_move", "other", "pointer_up"
    )
    child.add_event_handler(
        child_callback, "pointer_down", "pointer_move", "other", "pointer_up"
    )

    root_handler = RootEventHandler()
    root_handler.dispatch_event(
        PointerEvent(type="pointer_down", x=0, y=0, target=child)
    )
    root_handler.dispatch_event(
        PointerEvent(type="pointer_move", x=0, y=0, target=child)
    )
    root_handler.dispatch_event(PointerEvent(type="pointer_up", x=0, y=0, target=child))

    assert child_called == 3
    assert root_called == 3

    child.add_event_handler(
        lambda e: child.set_pointer_capture(e.pointer_id), "pointer_down"
    )
    child.add_event_handler(
        lambda e: child.release_pointer_capture(e.pointer_id), "pointer_up"
    )

    root_handler.dispatch_event(
        PointerEvent(type="pointer_down", x=0, y=0, target=child)
    )
    root_handler.dispatch_event(
        PointerEvent(type="pointer_move", x=0, y=0, target=child)
    )
    # Test that non-pointer events bubble along just find
    root_handler.dispatch_event(Event(type="other", target=child))
    root_handler.dispatch_event(PointerEvent(type="pointer_up", x=0, y=0, target=child))

    assert child_called == 7
    assert root_called == 4


def test_pointer_event_copy():
    target = object()
    event = PointerEvent(
        "pointer_down",
        x=1,
        y=1,
        button=1,
        buttons=(1,),
        modifiers=("Shift",),
        ntouches=2,
        touches={"foo": "bar"},
        pick_info={"FOO": "BAR"},
        clicks=3,
        pointer_id=3,
        bubbles=False,
        target=target,
        time_stamp=1234,
        cancelled=True,
    )

    other = event.copy(type="click", clicks=5)

    for attr in [
        attr
        for attr in dir(other)
        if not attr.startswith("_") and not callable(getattr(other, attr))
    ]:
        if attr not in ["type", "clicks"]:
            assert getattr(other, attr) == getattr(
                event, attr
            ), f"'{attr}' attribute not equal"

    assert other.type != event.type
    assert other.type == "click"

    assert other.clicks != event.clicks
    assert other.clicks == 5


def test_clicks():
    number_of_clicks = []
    root_handler = RootEventHandler()

    @root_handler.add_event_handler("click")
    def root_callback(event):
        nonlocal number_of_clicks
        number_of_clicks.append(event.clicks)

    for i in range(4):
        down = PointerEvent("pointer_down", x=0, y=0, time_stamp=i * 100)
        up = PointerEvent("pointer_up", x=10, y=30, time_stamp=i * 100 + 50)
        root_handler.dispatch_event(down)
        root_handler.dispatch_event(up)

    assert number_of_clicks == [1, 2, 3, 4]

    number_of_clicks = []

    # Bump the 'time' to trigger a reset of the tracker
    i = 5000

    target = EventTarget()
    target.parent = None
    down = PointerEvent("pointer_down", x=3, y=7, time_stamp=i * 100, target=target)
    up = PointerEvent("pointer_up", x=1, y=2, time_stamp=i * 100 + 50, target=target)
    root_handler.dispatch_event(down)
    root_handler.dispatch_event(up)

    assert number_of_clicks == [1]

    # Bump the 'time' to trigger a reset of the tracker
    i = 10000

    down = PointerEvent("pointer_down", x=9, y=6, time_stamp=i * 100, target=target)
    up = PointerEvent("pointer_up", x=3, y=5, time_stamp=i * 100 + 50)
    root_handler.dispatch_event(down)
    # Delete all references to the target
    del down
    del target
    root_handler.dispatch_event(up)

    assert number_of_clicks == [1]
