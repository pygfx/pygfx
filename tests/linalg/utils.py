from pygfx.linalg import (
    Euler,
    Matrix4,
    Quaternion,
)


def matrix_equals(a: Matrix4, b: Matrix4, tolerance: float = 0.0001):
    if len(a.elements) != len(b.elements):
        return False
    return all(abs(x - y) < tolerance for x, y in zip(a.elements, b.elements))


def euler_equals(a: Euler, b: Euler, tolerance: float = 0.0001):
    return (abs(a.x - b.x) + abs(a.y - b.y) + abs(a.z - b.z)) < tolerance


def quat_equals(a: Quaternion, b: Quaternion, tolerance: float = 0.0001):
    return (
        abs(a.x - b.x) + abs(a.y - b.y) + abs(a.z - b.z) + abs(a.w - b.w)
    ) < tolerance
