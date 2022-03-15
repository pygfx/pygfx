from pygfx import Scene, WorldObject
from pygfx.objects._events import EventTarget, RootHandler, create_event


class Node(EventTarget):
    def __init__(self, parent=None):
        super().__init__()
        self.parent = parent


def test_event_target():
    c = EventTarget()

    # It's event handling mechanism should be fully functional
    events = []

    def handler(event):
        events.append(event["value"])

    c.add_event_handler(handler, "foo", "bar")
    c.add_event_handler(handler, "bar")
    c.handle_event(create_event(type="foo", value=1))
    c.handle_event(create_event(type="bar", value=2))
    c.handle_event(create_event(type="spam", value=3))
    c.remove_event_handler(handler, "foo")
    c.handle_event(create_event(type="foo", value=4))
    c.handle_event(create_event(type="bar", value=5))
    c.handle_event(create_event(type="spam", value=6))
    c.remove_event_handler(handler, "bar")
    c.handle_event(create_event(type="foo", value=7))
    c.handle_event(create_event(type="bar", value=8))
    c.handle_event(create_event(type="spam", value=9))

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

    root_handler = RootHandler()
    scene = Scene()
    item = WorldObject()
    scene.add(item)

    root_handler.add_event_handler(root_callback, "foo")
    scene.add_event_handler(scene_callback, "foo")
    item.add_event_handler(item_callback, "foo")

    event = create_event(type="foo", target=item)
    root_handler.handle_event(event)

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
        event["propagate"] = False
        child_called += 1

    root = Node()
    child = Node(parent=root)

    root_handler = RootHandler()
    root.add_event_handler(root_callback, "foo")
    child.add_event_handler(child_callback, "foo")

    event = {"type": "foo", "target": child}
    root_handler.handle_event(create_event(**event))

    assert child_called == 1
    assert root_called == 1

    child.remove_event_handler(child_callback, "foo")
    child.add_event_handler(child_prevent_callback, "foo")

    root_handler.handle_event(create_event(**event))

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

    root_handler = RootHandler()

    event = create_event(type="foo", target=child, propagate=True)
    root_handler.handle_event(event)

    assert child_called == 1
    assert root_called == 1

    event = create_event(type="foo", target=child, propagate=False)
    root_handler.handle_event(event)

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

    root_handler = RootHandler()
    root_handler.handle_event(create_event(type="pointer_down", target=child))
    root_handler.handle_event(create_event(type="pointer_move", target=child))
    root_handler.handle_event(create_event(type="pointer_up", target=child))

    assert child_called == 3
    assert root_called == 3

    child.add_event_handler(
        lambda e: child.set_pointer_capture(e["pointer_id"]), "pointer_down"
    )
    child.add_event_handler(
        lambda e: child.release_pointer_capture(e["pointer_id"]), "pointer_up"
    )

    root_handler.handle_event(create_event(type="pointer_down", target=child))
    root_handler.handle_event(create_event(type="pointer_move", target=child))
    # Test that non-pointer events bubble along just find
    root_handler.handle_event(create_event(type="other", target=child))
    root_handler.handle_event(create_event(type="pointer_up", target=child))

    assert child_called == 7
    assert root_called == 4
