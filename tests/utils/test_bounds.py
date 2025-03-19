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

    # Actually 3 points
    points3a = np.array([(1, 1, 1), (0, 0, 0), (-1, -1, -1)], float)
    b = Bounds.from_points(points3a)
    assert np.all(b.aabb == points)


def test_points_many_rectangular():
    n = 10000
    x = np.random.uniform(-7, 12, size=n).reshape(n, 1)
    y = np.random.uniform(101, 102, size=n).reshape(n, 1)
    z = np.random.uniform(10, 30, size=n).reshape(n, 1)

    points = np.hstack([x, y, z])

    b = Bounds.from_points(points, True)
    assert b.radius is None
    assert np.allclose(b.center, (2.5, 101.5, 20), atol=0.01)

    assert np.allclose(b.aabb[:, 0], (-7, 12), atol=0.01)
    assert np.allclose(b.aabb[:, 1], (101, 102), atol=0.01)
    assert np.allclose(b.aabb[:, 2], (10, 30), atol=0.01)


def test_points_circular():
    n = 10000
    r = 12
    ang = np.random.uniform(0, 2 * np.pi, size=n)
    x = np.sin(ang).reshape(n, 1) * r + 100
    y = np.cos(ang).reshape(n, 1) * r + 200
    z = np.zeros_like(x)

    points = np.hstack([x, y, z])

    b = Bounds.from_points(points, True)
    assert b.radius is not None
    assert 11.99 < b.radius < 12.01
    assert np.allclose(b.center, (100, 200, 0), atol=0.01)
    assert np.allclose(b.aabb[:, 0], (88, 112), atol=0.01)
    assert np.allclose(b.aabb[:, 1], (188, 212), atol=0.01)
    assert np.allclose(b.aabb[:, 2], (0, 0), atol=0.01)


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
