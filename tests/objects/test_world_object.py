from math import pi
from unittest.mock import Mock, call
from weakref import ref
import pylinalg as pla
import numpy as np

from pygfx import WorldObject
import pygfx as gfx


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
    root.transform.position = (3, 6, 8)
    root.transform.scale = (1, 1.2, 1)
    root.transform.rotation = pla.quaternion_make_from_euler_angles((pi / 2, 0, 0))

    pos, rot, scale = pla.matrix_decompose(root.transform.matrix)
    assert np.allclose(pos, root.transform.position)
    assert np.allclose(rot, root.transform.rotation)
    assert np.allclose(scale, root.transform.scale)


def test_update_matrix_world():
    root = WorldObject()
    root.transform.position = (-5, 8, 0)
    root.transform.rotation = pla.quaternion_make_from_euler_angles((pi / 4, 0, 0))

    child1 = WorldObject()
    child1.transform.position = (0, 0, 5)
    root.add(child1)

    child2 = WorldObject()
    child2.transform.rotation = pla.quaternion_make_from_euler_angles((0, -pi / 4, 0))
    child1.add(child2)

    expected = (
        root.transform
        @ child1.transform
        @ child2.transform
        @ pla.vector_make_homogeneous((10, 10, 10))
    )

    actual = child2.world_transform @ pla.vector_make_homogeneous((10, 10, 10))

    # if there is a difference it's a floating point error
    assert np.allclose(actual, expected)


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

    expected_children = (child1, child2)
    assert len(root.children) == len(expected_children)
    for actual, expected in zip(root.children, expected_children):
        assert actual == expected

    root.add(child2, before=child1)

    expected_children = (child2, child1)
    assert len(root.children) == len(expected_children)
    for actual, expected in zip(root.children, expected_children):
        assert actual == expected

    child3 = WorldObject()
    root.add(child3, child1, before=child2)

    expected_children = (child3, child1, child2)
    assert len(root.children) == len(expected_children)
    for actual, expected in zip(root.children, expected_children):
        assert actual is expected


def test_iter():
    class Foo(WorldObject):
        pass

    root = WorldObject()
    root.add(Foo(), Foo())
    assert len(list(root.iter(lambda x: isinstance(x, Foo)))) == 2

    root = Foo()
    root.add(Foo(), Foo())
    assert len(list(root.iter(lambda x: isinstance(x, Foo)))) == 3

    root = WorldObject()
    root.add(Foo(), WorldObject(), Foo())
    assert len(list(root.iter(lambda x: isinstance(x, Foo)))) == 2


def test_setting_world_transform():
    root = gfx.WorldObject()
    child = gfx.WorldObject()
    root.add(child)

    root.transform.position = (1, 2, 3)
    child.transform.position = (4, 4, 4)

    assert np.allclose(child.transform.position, (4, 4, 4))
    assert np.allclose(child.world_transform.position, (5, 6, 7))

    child.world_transform.position = (1, 2, 3)

    assert np.allclose(child.transform.position, (0, 0, 0))
    assert np.allclose(child.world_transform.position, (1, 2, 3))
