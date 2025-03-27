from pygfx.utils.bounds import Bounds
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
