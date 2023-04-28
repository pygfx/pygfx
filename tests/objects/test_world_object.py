from math import pi
from unittest.mock import Mock, call
from weakref import ref
import pylinalg as la
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
    root.local.position = (3, 6, 8)
    root.local.scale = (1, 1.2, 1)
    root.local.rotation = la.quaternion_make_from_euler_angles((pi / 2, 0, 0))

    pos, rot, scale = la.matrix_decompose(root.local.matrix)
    assert np.allclose(pos, root.local.position)
    assert np.allclose(rot, root.local.rotation)
    assert np.allclose(scale, root.local.scale)


def test_update_matrix_world():
    root = WorldObject()
    root.local.position = (-5, 8, 0)
    root.local.rotation = la.quaternion_make_from_euler_angles((pi / 4, 0, 0))

    child1 = WorldObject()
    child1.local.position = (0, 0, 5)
    root.add(child1)

    child2 = WorldObject()
    child2.local.rotation = la.quaternion_make_from_euler_angles((0, -pi / 4, 0))
    child1.add(child2)

    expected = (
        root.local
        @ child1.local
        @ child2.local
        @ la.vector_make_homogeneous((10, 10, 10))
    )

    actual = child2.world @ la.vector_make_homogeneous((10, 10, 10))

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
        assert actual is expected

    root.add(child2, before=child1)

    expected_children = (child2, child1)
    assert len(root.children) == len(expected_children)
    for actual, expected in zip(root.children, expected_children):
        assert actual is expected

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
    child2 = gfx.WorldObject()

    root.add(child)
    child.add(child2)

    root.local.position = (1, 2, 3)
    child.local.position = (4, 4, 4)
    child2.local.position = (10, 0, 0)

    assert np.allclose(child.local.position, (4, 4, 4))
    assert np.allclose(child.world.position, (5, 6, 7))

    assert np.allclose(child2.local.position, (10, 0, 0))
    assert np.allclose(child2.world.position, (15, 6, 7))

    child.world.position = (1, 2, 3)

    assert np.allclose(child.local.position, (0, 0, 0))
    assert np.allclose(child.world.position, (1, 2, 3))

    assert np.allclose(child2.local.position, (10, 0, 0))
    assert np.allclose(child2.world.position, (11, 2, 3))


def test_complex_multi_insert():
    children = []
    for _ in range(5):
        children.append(gfx.WorldObject())

    root = gfx.WorldObject()
    for _ in range(3):
        root.add(gfx.WorldObject())

    first = root.children[0]
    reference = root.children[1]
    root.add(*children, before=reference)

    # ensure children were inserted in order before `before`
    for from_root, child in zip(root.children[1:6], children):
        assert from_root is child
    assert root.children[6] is reference

    root.add(first, before=reference)
    assert root.children[0] is not first
    assert root.children[5] is first
    assert root.children[6] is reference


def test_axis_getters():
    # "normal" euclidean space
    obj = gfx.WorldObject()
    assert np.allclose(obj.world.forward, (0, 0, 1))
    assert np.allclose(obj.world.up, (0, 1, 0))
    assert np.allclose(obj.world.right, (-1, 0, 0))

    # camera space
    camera = gfx.PerspectiveCamera()
    assert np.allclose(camera.world.forward, (0, 0, -1))
    assert np.allclose(camera.world.up, (0, 1, 0))
    assert np.allclose(camera.world.right, (1, 0, 0))

    # position offsets shouldn't matter
    obj.world.position = (0, 0, -10)
    assert np.allclose(obj.world.forward, (0, 0, 1))
    assert np.allclose(obj.world.up, (0, 1, 0))
    assert np.allclose(obj.world.right, (-1, 0, 0))

    # rotations should influence local orientations
    obj.world.rotation = la.quaternion_make_from_euler_angles(np.pi / 2, order="Y")
    assert np.allclose(obj.local.up, (0, 1, 0))
    assert np.allclose(obj.local.right, (0, 0, 1))
    assert np.allclose(obj.local.forward, (1, 0, 0))

    obj.world.rotation = la.quaternion_make_from_euler_angles(
        (np.pi / 4, np.pi / 4), order="XZ"
    )
    assert np.allclose(obj.local.forward, (0, -np.cos(np.pi / 4), np.sin(np.pi / 4)))

    print("")


def test_axis_setters():
    obj = gfx.WorldObject()

    obj.world.forward = (1, 0, 0)
    assert np.allclose(obj.world.forward, (1, 0, 0))
    assert np.allclose(
        obj.world.rotation, (0, np.sin((np.pi / 2) / 2), 0, np.cos((np.pi / 2) / 2))
    )

    obj.world.right = (-1, 0, 0)
    assert np.allclose(obj.world.right, (-1, 0, 0))
    assert np.allclose(obj.world.rotation, (0, 0, 0, 1))

    obj.world.right = (1, 0, 0)
    assert np.allclose(obj.world.right, (1, 0, 0))
    assert np.allclose(obj.world.up, (0, 1, 0))
    assert np.allclose(obj.world.forward, (0, 0, -1))

    obj.world.up = (1, 1, 1)
    assert np.allclose(obj.world.up, (1 / np.sqrt(3), 1 / np.sqrt(3), 1 / np.sqrt(3)))

def test_reference_up():
    group = gfx.WorldObject()
    assert np.allclose(group.world.reference_up, (0, 1, 0))
    assert np.allclose(group.local.reference_up, (0, 1, 0))

    # reference_up is given in parent frame, so it is independent of the transform
    group.world.forward = (1, 1, 1)
    group.world.position = (1, 42, 13)
    assert np.allclose(group.world.reference_up, (0, 1, 0))
    assert np.allclose(group.local.reference_up, (0, 1, 0))

    # local reference_up does change if there is a parent since the parent frame may have
    # a transform relative to world
    obj1 = gfx.WorldObject()
    obj1.world.position = (0, 4, 9)
    group.add(obj1, keep_world_matrix=True)
    assert np.allclose(obj1.world.position, (0, 4, 9))
    reference_up = la.vector_apply_matrix(
        obj1.world.reference_up, group.world.inverse_matrix
    )
    world_origin = la.vector_apply_matrix((0, 0, 0), group.world.inverse_matrix)
    reference_up = reference_up - world_origin
    assert np.allclose(obj1.local.reference_up, reference_up)
    
    # but the parent remains unaffected by its children
    # as does the world reference
    assert np.allclose(group.local.reference_up, (0, 1, 0))
    assert np.allclose(group.world.reference_up, (0, 1, 0))
    assert np.allclose(obj1.world.reference_up, (0, 1, 0))

    # (world) up_reference is independent between objects
    obj2 = gfx.WorldObject()
    obj2.world.rotation = (0, 0, 1, 0)
    group.add(obj1, keep_world_matrix=True)

    obj1.world.reference_up = (1, 2, 3)
    obj2.world.reference_up = (1, 0, 1)

    assert np.allclose(group.local.reference_up, (0, 1, 0))
    assert np.allclose(obj1.world.reference_up, (1, 2, 3))
    assert np.allclose(obj2.world.reference_up, (1, 0, 1))
