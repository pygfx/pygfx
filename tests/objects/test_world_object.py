from math import pi
from unittest.mock import Mock, call
from weakref import ref

from pygfx import WorldObject
from pygfx.linalg import Euler, Vector3, Quaternion


def test_traverse():
    root = WorldObject()

    layer1_child1 = WorldObject()
    root.add(layer1_child1)
    layer1_child2 = WorldObject()
    root.add(layer1_child2)

    layer2_child1 = WorldObject()
    layer1_child2.add(layer2_child1)
    layer2_child2 = WorldObject()
    layer1_child2.add(layer2_child2)

    mock = Mock()
    root.traverse(mock)
    mock.assert_has_calls(
        [
            call(root),
            call(layer1_child1),
            call(layer1_child2),
            call(layer2_child1),
            call(layer2_child2),
        ]
    )
    assert len(mock.mock_calls) == 5


def test_remove():
    root = WorldObject()

    layer1_child1 = WorldObject()
    root.add(layer1_child1)
    layer1_child2 = WorldObject()
    root.add(layer1_child2)

    layer2_child1 = WorldObject()
    layer1_child2.add(layer2_child1)
    layer2_child2 = WorldObject()
    layer1_child2.add(layer2_child2)

    root.remove(layer1_child2)
    # layer1_child2 removed
    assert layer1_child2.parent is None
    assert layer1_child2 not in root.children
    # layer1_child1 not removed
    assert layer1_child1.parent is root
    assert layer1_child1 in root.children


def test_update_matrix():
    root = WorldObject()
    root.position.set(3, 6, 8)
    root.scale.set(1, 1.2, 1)
    root.rotation.set_from_euler(Euler(pi / 2, 0, 0))
    root.update_matrix()

    t, r, s = Vector3(), Quaternion(), Vector3()
    root.matrix.decompose(t, r, s)
    assert t == root.position
    # todo: do somehting like np.allclose
    # assert r == root.rotation  # close, but not quite the same
    # assert s == root.scale
    assert root.matrix_world_dirty


def test_update_matrix_world():
    root = WorldObject()
    root.position.set(-5, 8, 0)
    root.rotation.set_from_euler(Euler(pi / 4, 0, 0))
    root.update_matrix()

    child1 = WorldObject()
    child1.position.set(0, 0, 5)
    root.add(child1)

    child2 = WorldObject()
    child2.rotation.set_from_euler(Euler(0, -pi / 4, 0))
    child1.add(child2)

    objs = [root, child1, child2]
    assert all(obj.matrix_world_dirty for obj in objs)

    # test both updating parents and children
    child1.update_matrix_world(update_parents=True)
    assert all(not obj.matrix_world_dirty for obj in objs)

    p = Vector3(10, 10, 10)
    p.apply_matrix4(child2.matrix)
    p.apply_matrix4(child1.matrix)
    p.apply_matrix4(root.matrix)

    x = Vector3(10, 10, 10)
    x.apply_matrix4(child2.matrix_world)

    # if there is a difference it's a floating point error
    assert Vector3().sub_vectors(p, x).length() < 0.00000000001

    # reorganize such that child1 and 2 become siblings
    child1.remove(child2)
    root.add(child2)
    assert not child1.matrix_world_dirty
    # child2 should be flagged as dirty again now
    assert child2.matrix_world_dirty


def test_reparenting():
    root = WorldObject()
    child1 = WorldObject()
    child2 = WorldObject()
    root.add(child1, child2)

    obj = WorldObject()
    child1.add(obj)

    assert obj.parent is child1
    assert obj in child1.children

    child2.add(obj)

    assert obj.parent is child2
    assert obj not in child1.children
    assert obj in child2.children


def test_no_cyclic_references():
    parent = WorldObject()
    child = WorldObject()
    parent.add(child)
    # Add object without creating a strong reference
    child.add(WorldObject())

    # Create a weak ref to the added child
    grandchild_ref = ref(child.children[0])
    # Calling the ref should retrieve the grandchild
    assert grandchild_ref() == child.children[0]

    # When the child is removed (and deleted), the grandchild
    # should also be garbage collected as there should be no
    # direct references anymore
    parent.remove(child)
    del child

    assert not grandchild_ref()


def test_adjust_children_order():
    root = WorldObject()
    child1 = WorldObject()
    child2 = WorldObject()

    root.add(child1, child2)

    assert root.children == (child1, child2)

    root.add(child2, before=child1)

    assert root.children == (child2, child1)

    child3 = WorldObject()
    root.add(child3, child1, before=child2)

    assert root.children == (child3, child1, child2)

    # Assert that nothing happens when adding the same element
    # before itself
    root.add(child1, before=child1)
    assert root.children == (child3, child1, child2)

    non_child = WorldObject()

    # Append item at end when the `before` parameter
    # is not in children
    root.add(child1, before=non_child)
    assert root.children == (child3, child2, child1)
