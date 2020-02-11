from math import pi, sqrt

from visvis2.linalg import (
    Vector3,
    Vector4,
    Euler,
    Matrix4,
    Quaternion,
)


x = 2
y = 3
z = 4
w = 5
eps = 0.0001
orders = [
    Euler.RotationOrders.XYZ,
    Euler.RotationOrders.YXZ,
    Euler.RotationOrders.ZXY,
    Euler.RotationOrders.ZYX,
    Euler.RotationOrders.YZX,
    Euler.RotationOrders.XZY,
]
euler_angles = Euler(0.1, -0.3, 0.25)


def q_sub(a, b):
    result = a.clone()
    result.x -= b.x
    result.y -= b.y
    result.z -= b.z
    result.w -= b.w
    return result


def change_euler_order(euler, order):
    return Euler(euler.x, euler.y, euler.z, order)


# INSTANCING
def test_instancing():
    a = Quaternion()
    assert a.x == 0
    assert a.y == 0
    assert a.z == 0
    assert a.w == 1

    a = Quaternion(x, y, z, w)
    assert a.x == x
    assert a.y == y
    assert a.z == z
    assert a.w == w


# PROPERTIES
def test_properties():
    a = Quaternion()
    a.x = x
    a.y = y
    a.z = z
    a.w = w

    assert a.x == x, "Check x"
    assert a.y == y, "Check y"
    assert a.z == z, "Check z"
    assert a.w == w, "Check w"


def test_x():
    a = Quaternion()
    assert a.x == 0

    a = Quaternion(1, 2, 3)
    assert a.x == 1

    a = Quaternion(4, 5, 6, 1)
    assert a.x == 4

    a = Quaternion(7, 8, 9)
    a.x = 10
    assert a.x == 10

    a = Quaternion(11, 12, 13)
    a.x = 14
    assert a.x == 14


def test_y():
    a = Quaternion()
    assert a.y == 0

    a = Quaternion(1, 2, 3)
    assert a.y == 2

    a = Quaternion(4, 5, 6, 1)
    assert a.y == 5

    a = Quaternion(7, 8, 9)
    a.y = 10
    assert a.y == 10

    a = Quaternion(11, 12, 13)
    a.y = 14
    assert a.y == 14


def test_z():

    a = Quaternion()
    assert a.z == 0

    a = Quaternion(1, 2, 3)
    assert a.z == 3

    a = Quaternion(4, 5, 6, 1)
    assert a.z == 6

    a = Quaternion(7, 8, 9)
    a.z = 10
    assert a.z == 10

    a = Quaternion(11, 12, 13)
    a.z = 14
    assert a.z == 14


def test_w():
    a = Quaternion()
    assert a.w == 1

    a = Quaternion(1, 2, 3)
    assert a.w == 1

    a = Quaternion(4, 5, 6, 1)
    assert a.w == 1

    a = Quaternion(7, 8, 9)
    a.w = 10
    assert a.w == 10

    a = Quaternion(11, 12, 13)
    a.w = 14
    assert a.w == 14


# PUBLIC STUFF
def test_set():
    a = Quaternion()
    assert a.x == 0
    assert a.y == 0
    assert a.z == 0
    assert a.w == 1

    a.set(x, y, z, w)
    assert a.x == x
    assert a.y == y
    assert a.z == z
    assert a.w == w


def test_clone():

    a = Quaternion().clone()
    assert a.x == 0
    assert a.y == 0
    assert a.z == 0
    assert a.w == 1

    b = a.set(x, y, z, w).clone()
    assert b.x == x
    assert b.y == y
    assert b.z == z
    assert b.w == w


def test_copy():
    a = Quaternion(x, y, z, w)
    b = Quaternion().copy(a)
    assert b.x == x
    assert b.y == y
    assert b.z == z
    assert b.w == w

    # ensure that it is a True copy
    a.x = 0
    a.y = -1
    a.z = 0
    a.w = -1
    assert b.x == x
    assert b.y == y


def test_set_from_euler_set_from_quaternion():
    angles = [Vector3(1, 0, 0), Vector3(0, 1, 0), Vector3(0, 0, 1)]

    # ensure euler conversion to/from Quaternion matches.
    for i in range(len(orders)):
        for j in range(len(angles)):
            eulers2 = Euler().set_from_quaternion(
                Quaternion().set_from_euler(
                    Euler(angles[j].x, angles[j].y, angles[j].z, orders[i])
                ),
                orders[i],
            )
            new_angle = Vector3(eulers2.x, eulers2.y, eulers2.z)
            assert new_angle.distance_to(angles[j]) < 0.001


def test_set_from_axis_angle():
    # TODO: find cases to validate.
    # assert True

    zero = Quaternion()

    a = Quaternion().set_from_axis_angle(Vector3(1, 0, 0), 0)
    assert a.equals(zero)
    a = Quaternion().set_from_axis_angle(Vector3(0, 1, 0), 0)
    assert a.equals(zero)
    a = Quaternion().set_from_axis_angle(Vector3(0, 0, 1), 0)
    assert a.equals(zero)

    b1 = Quaternion().set_from_axis_angle(Vector3(1, 0, 0), pi)
    assert not a.equals(b1)
    b2 = Quaternion().set_from_axis_angle(Vector3(1, 0, 0), -pi)
    assert not a.equals(b2)

    b1.multiply(b2)
    assert a.equals(b1)


def test_set_from_euler_set_from_rotation_matrix():
    # ensure euler conversion for Quaternion matches that of Matrix4
    for i in range(len(orders)):
        q = Quaternion().set_from_euler(change_euler_order(euler_angles, orders[i]))
        m = Matrix4().make_rotation_from_euler(
            change_euler_order(euler_angles, orders[i])
        )
        q2 = Quaternion().set_from_rotation_matrix(m)

        assert q_sub(q, q2).length() < 0.001


def test_set_from_rotation_matrix():
    # contrived examples purely to please the god of code coverage...
    # match conditions in various 'else [if]' blocks

    a = Quaternion()
    q = Quaternion(-9, -2, 3, -4).normalize()
    m = Matrix4().make_rotation_from_quaternion(q)
    expected = Vector4(
        0.8581163303210332,
        0.19069251784911848,
        -0.2860387767736777,
        0.38138503569823695,
    )

    a.set_from_rotation_matrix(m)
    assert abs(a.x - expected.x) <= eps, "m11 > m22 && m11 > m33: check x"
    assert abs(a.y - expected.y) <= eps, "m11 > m22 && m11 > m33: check y"
    assert abs(a.z - expected.z) <= eps, "m11 > m22 && m11 > m33: check z"
    assert abs(a.w - expected.w) <= eps, "m11 > m22 && m11 > m33: check w"

    q = Quaternion(-1, -2, 1, -1).normalize()
    m.make_rotation_from_quaternion(q)
    expected = Vector4(
        0.37796447300922714,
        0.7559289460184544,
        -0.37796447300922714,
        0.37796447300922714,
    )

    a.set_from_rotation_matrix(m)
    assert abs(a.x - expected.x) <= eps, "m22 > m33: check x"
    assert abs(a.y - expected.y) <= eps, "m22 > m33: check y"
    assert abs(a.z - expected.z) <= eps, "m22 > m33: check z"
    assert abs(a.w - expected.w) <= eps, "m22 > m33: check w"


def test_set_from_unit_vectors():
    a = Quaternion()
    b = Vector3(1, 0, 0)
    c = Vector3(0, 1, 0)
    expected = Quaternion(0, 0, sqrt(2) / 2, sqrt(2) / 2)

    a.set_from_unit_vectors(b, c)
    assert abs(a.x - expected.x) <= eps, "Check x"
    assert abs(a.y - expected.y) <= eps, "Check y"
    assert abs(a.z - expected.z) <= eps, "Check z"
    assert abs(a.w - expected.w) <= eps, "Check w"


def test_angle_to():
    a = Quaternion()
    b = Quaternion().set_from_euler(Euler(0, pi, 0))
    c = Quaternion().set_from_euler(Euler(0, pi * 2, 0))

    assert a.angle_to(a) == 0
    assert a.angle_to(b) == pi
    assert a.angle_to(c) == 0


def test_rotate_towards():
    a = Quaternion()
    b = Quaternion().set_from_euler(Euler(0, pi, 0))
    c = Quaternion()

    half_pi = pi * 0.5

    a.rotate_towards(b, 0)
    assert a.equals(a) is True

    a.rotate_towards(b, pi * 2)
    # test overshoot
    assert a.equals(b) is True

    a.set(0, 0, 0, 1)
    a.rotate_towards(b, half_pi)
    assert a.angle_to(c) - half_pi <= eps


def test_inverse_conjugate():
    a = Quaternion(x, y, z, w)

    # TODO: add better validation here.

    b = a.clone().conjugate()

    assert a.x == -b.x
    assert a.y == -b.y
    assert a.z == -b.z
    assert a.w == b.w


def test_dot():
    a = Quaternion()
    b = Quaternion()

    assert a.dot(b) == 1
    a = Quaternion(1, 2, 3, 1)
    b = Quaternion(3, 2, 1, 1)

    assert a.dot(b) == 11


def test_normalize_length_length_sq():
    a = Quaternion(x, y, z, w)

    assert a.length() != 1
    assert a.length_sq() != 1
    a.normalize()
    assert a.length() == 1
    assert a.length_sq() == 1

    a.set(0, 0, 0, 0)
    assert a.length_sq() == 0
    assert a.length() == 0
    a.normalize()
    assert a.length_sq() == 1
    assert a.length() == 1


def test_multiply_quaternions_multiply():
    angles = [Euler(1, 0, 0), Euler(0, 1, 0), Euler(0, 0, 1)]

    q1 = Quaternion().set_from_euler(
        change_euler_order(angles[0], Euler.RotationOrders.XYZ)
    )
    q2 = Quaternion().set_from_euler(
        change_euler_order(angles[1], Euler.RotationOrders.XYZ)
    )
    q3 = Quaternion().set_from_euler(
        change_euler_order(angles[2], Euler.RotationOrders.XYZ)
    )

    q = Quaternion().multiply_quaternions(q1, q2).multiply(q3)

    m1 = Matrix4().make_rotation_from_euler(
        change_euler_order(angles[0], Euler.RotationOrders.XYZ)
    )
    m2 = Matrix4().make_rotation_from_euler(
        change_euler_order(angles[1], Euler.RotationOrders.XYZ)
    )
    m3 = Matrix4().make_rotation_from_euler(
        change_euler_order(angles[2], Euler.RotationOrders.XYZ)
    )

    m = Matrix4().multiply_matrices(m1, m2).multiply(m3)

    q_from_m = Quaternion().set_from_rotation_matrix(m)

    assert q_sub(q, q_from_m).length() < 0.001


def test_premultiply():
    a = Quaternion(x, y, z, w)
    b = Quaternion(2 * x, -y, -2 * z, w)
    expected = Quaternion(42, -32, -2, 58)

    a.premultiply(b)
    assert abs(a.x - expected.x) <= eps, "Check x"
    assert abs(a.y - expected.y) <= eps, "Check y"
    assert abs(a.z - expected.z) <= eps, "Check z"
    assert abs(a.w - expected.w) <= eps, "Check w"


def test_slerp():
    a = Quaternion(x, y, z, w)
    b = Quaternion(-x, -y, -z, -w)

    c = a.clone().slerp(b, 0)
    d = a.clone().slerp(b, 1)

    assert a.equals(c), "Passed"
    assert b.equals(d), "Passed"

    sqrt1_2 = sqrt(1 / 2)

    e = Quaternion(1, 0, 0, 0)
    f = Quaternion(0, 0, 1, 0)
    expected = Quaternion(sqrt1_2, 0, sqrt1_2, 0)
    result = e.clone().slerp(f, 0.5)
    assert abs(result.x - expected.x) <= eps, "Check x"
    assert abs(result.y - expected.y) <= eps, "Check y"
    assert abs(result.z - expected.z) <= eps, "Check z"
    assert abs(result.w - expected.w) <= eps, "Check w"

    g = Quaternion(0, sqrt1_2, 0, sqrt1_2)
    h = Quaternion(0, -sqrt1_2, 0, sqrt1_2)
    expected = Quaternion(0, 0, 0, 1)
    result = g.clone().slerp(h, 0.5)

    assert abs(result.x - expected.x) <= eps, "Check x"
    assert abs(result.y - expected.y) <= eps, "Check y"
    assert abs(result.z - expected.z) <= eps, "Check z"
    assert abs(result.w - expected.w) <= eps, "Check w"


def test_equals():
    a = Quaternion(x, y, z, w)
    b = Quaternion(-x, -y, -z, -w)

    assert a.x != b.x
    assert a.y != b.y

    assert not a.equals(b)
    assert not b.equals(a)

    a.copy(b)
    assert a.x == b.x
    assert a.y == b.y

    assert a.equals(b)
    assert b.equals(a)


def test_from_array():
    a = Quaternion()
    a.from_array([x, y, z, w])
    assert a.x == x
    assert a.y == y
    assert a.z == z
    assert a.w == w

    a.from_array([None, x, y, z, w, None], 1)
    assert a.x == x
    assert a.y == y
    assert a.z == z
    assert a.w == w


def test_to_array():
    a = Quaternion(x, y, z, w)

    array = a.to_array()
    assert array[0] == x, "No array, no offset: check x"
    assert array[1] == y, "No array, no offset: check y"
    assert array[2] == z, "No array, no offset: check z"
    assert array[3] == w, "No array, no offset: check w"

    array = []
    a.to_array(array)
    assert array[0] == x, "With array, no offset: check x"
    assert array[1] == y, "With array, no offset: check y"
    assert array[2] == z, "With array, no offset: check z"
    assert array[3] == w, "With array, no offset: check w"

    array = []
    a.to_array(array, 1)
    assert array[0] is None, "With array and offset: check [0]"
    assert array[1] == x, "With array and offset: check x"
    assert array[2] == y, "With array and offset: check y"
    assert array[3] == z, "With array and offset: check z"
    assert array[4] == w, "With array and offset: check w"


# OTHERS
def test_multiply_vector3():
    angles = [Euler(1, 0, 0), Euler(0, 1, 0), Euler(0, 0, 1)]

    # ensure euler conversion for Quaternion matches that of Matrix4
    for i in range(len(orders)):
        for j in range(len(angles)):
            q = Quaternion().set_from_euler(change_euler_order(angles[j], orders[i]))
            m = Matrix4().make_rotation_from_euler(
                change_euler_order(angles[j], orders[i])
            )

            v0 = Vector3(1, 0, 0)
            qv = v0.clone().apply_quaternion(q)
            mv = v0.clone().apply_matrix4(m)

            assert qv.distance_to(mv) < 0.001
