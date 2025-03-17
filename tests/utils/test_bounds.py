from pygfx.utils.bounds import Bounds, bounds_to_points
import numpy as np


def test_points0():
    # Zero points -> no bounds

    points = np.zeros((0, 3), float)
    b = Bounds.from_points(points)
    assert b is None


def test_points1():
    # Single point, bounds without volume

    points = np.array([(0, 0, 0)], float)
    b = Bounds.from_points(points)
    assert b.aabb.min() == 0
    assert b.aabb.max() == 0
    assert b.sphere == (0, 0, 0, 0)

    points = np.array([(1, 2, 3)], float)
    b = Bounds.from_points(points)
    assert np.all(b.aabb[0] == points)
    assert np.all(b.aabb[1] == points)
    assert b.sphere == (1, 2, 3, 0)


def test_points2():
    # Two points

    # If point1 < point2, the two points ARE the aabb :)
    points = np.array([(-1, -1, -1), (1, 1, 1)], float)

    b = Bounds(points)
    assert b.aabb is points
    assert b.sphere == (0, 0, 0, 1.7320508075688772)

    b = Bounds.from_points(points)
    assert b.aabb is not points
    assert np.all(b.aabb == points)

    points2a = np.array([(1, 1, 1), (-1, -1, -1)], float)
    b = Bounds.from_points(points2a)
    assert np.all(b.aabb == points)

    points3a = np.array([(1, 1, 1), (0, 0, 0), (-1, -1, -1)], float)
    b = Bounds.from_points(points3a)
    assert np.all(b.aabb == points)


def test_bounding_points():
    def points_to_set(points):
        return set(tuple(float(i) for i in point[:3]) for point in points)

    # An aabb centered around the origin
    aabb = np.array([(-1, -1, -1), (1, 1, 1)], float)
    # No radius
    points = bounds_to_points(aabb, None)
    points_set = points_to_set(points)
    ref_set = {
        (-1, -1, -1),
        (-1, -1, 1),
        (-1, 1, -1),
        (-1, 1, 1),
        (1, -1, -1),
        (1, -1, 1),
        (1, 1, -1),
        (1, 1, 1),
    }
    assert len(points_set) == 8, points_set
    assert points_set == ref_set, points_set
    # With radius
    points = bounds_to_points(aabb, 1)
    points_set = points_to_set(points)
    ref_set = {
        (-1, 0, 0),
        (1, 0, 0),
        (0, -1, 0),
        (0, 1, 0),
        (0, 0, -1),
        (0, 0, 1),
        (-0.707106, -0.707106, -0.707106),
        (-0.707106, -0.707106, +0.707106),
        (-0.707106, +0.707106, -0.707106),
        (-0.707106, +0.707106, +0.707106),
        (+0.707106, -0.707106, -0.707106),
        (+0.707106, -0.707106, +0.707106),
        (+0.707106, +0.707106, -0.707106),
        (+0.707106, +0.707106, +0.707106),
    }
    assert len(points_set) == 14, points_set
    assert points_set == ref_set, points_set

    # An aabb centered around (20, 30, 40)
    aabb = np.array([(19, 28, 37), (21, 32, 43)], float)
    # No radius
    points = bounds_to_points(aabb, None)
    points_set = points_to_set(points)
    ref_set = {
        (19, 28, 37),
        (19, 28, 43),
        (19, 32, 37),
        (19, 32, 43),
        (21, 28, 37),
        (21, 28, 43),
        (21, 32, 37),
        (21, 32, 43),
    }
    assert len(points_set) == 8, points_set
    assert points_set == ref_set, points_set
    # With radius
    points = bounds_to_points(aabb, 1)
    points_set = points_to_set(points)
    ref_set = {
        (19, 30, 40),
        (21, 30, 40),
        (20, 28, 40),
        (20, 32, 40),
        (20, 30, 37),
        (20, 30, 43),
        (20 - 0.707106, 30 - 0.707106, 40 - 0.707106),
        (20 - 0.707106, 30 - 0.707106, 40 + 0.707106),
        (20 - 0.707106, 30 + 0.707106, 40 - 0.707106),
        (20 - 0.707106, 30 + 0.707106, 40 + 0.707106),
        (20 + 0.707106, 30 - 0.707106, 40 - 0.707106),
        (20 + 0.707106, 30 - 0.707106, 40 + 0.707106),
        (20 + 0.707106, 30 + 0.707106, 40 - 0.707106),
        (20 + 0.707106, 30 + 0.707106, 40 + 0.707106),
    }
    assert len(points_set) == 14, points_set
    assert points_set == ref_set, points_set


def test_bounding_points_lean_versions():
    # Some special cases of the bounds need less points

    def points_to_set(points):
        return set(tuple(float(i) for i in point[:3]) for point in points)

    # A single point
    aabb = np.array([(1, 2, 3), (1, 2, 3)], float)
    points = bounds_to_points(aabb, None)
    assert len(points) == 1
    assert tuple(points[0]) == (1, 2, 3)

    # A plane at z==0, centered around the origin
    aabb = np.array([(-1, -1, 0), (1, 1, 0)], float)
    # No radius
    points = bounds_to_points(aabb, None)
    points_set = points_to_set(points)
    ref_set = {(-1, -1, 0), (-1, 1, 0), (1, -1, 0), (1, 1, 0)}
    assert len(points_set) == 4, points_set
    assert points_set == ref_set, points_set
    # With radius
    points = bounds_to_points(aabb, 1)
    points_set = points_to_set(points)
    ref_set = {
        (-1, 0, 0),
        (1, 0, 0),
        (0, -1, 0),
        (0, 1, 0),
        (-0.707106, -0.707106, 0),
        (-0.707106, 0.707106, 0),
        (0.707106, -0.707106, 0),
        (0.707106, 0.707106, 0),
    }
    assert len(points_set) == 8, points_set
    assert points_set == ref_set, points_set

    # Still plane at z==0, centered around (20, 30, 40)
    aabb = np.array([(19, 28, 40), (21, 32, 40)], float)
    # No radius
    points = bounds_to_points(aabb, None)
    points_set = points_to_set(points)
    ref_set = {(19, 28, 40), (19, 32, 40), (21, 28, 40), (21, 32, 40)}
    assert len(points_set) == 4, points_set
    assert points_set == ref_set, points_set
    # With radius
    points = bounds_to_points(aabb, 1)
    points_set = points_to_set(points)
    ref_set = {
        (19, 30, 40),
        (21, 30, 40),
        (20, 28, 40),
        (20, 32, 40),
        (20 - 0.707106, 30 - 0.707106, 40),
        (20 - 0.707106, 30 + 0.707106, 40),
        (20 + 0.707106, 30 - 0.707106, 40),
        (20 + 0.707106, 30 + 0.707106, 40),
    }
    assert len(points_set) == 8, points_set
    assert points_set == ref_set, points_set


if __name__ == "__main__":
    test_points0()
    test_points1()
    test_points2()
    test_bounding_points()
    test_bounding_points_lean_versions()
