from visvis2.linalg import (
    Vector3,
    Euler,
    Matrix4,
    Quaternion,
)

from utils import matrix_equals, euler_equals, quat_equals


x = 2
y = 3
z = 4
w = 5
euler_zero = Euler(0, 0, 0, Euler.RotationOrders.XYZ)
euler_axyz = Euler(1, 0, 0, Euler.RotationOrders.XYZ)
euler_azyx = Euler(0, 1, 0, Euler.RotationOrders.ZYX)


# INSTANCING
def test_instancing():
    a = Euler()
    assert a.equals(euler_zero)
    assert not a.equals(euler_axyz)
    assert not a.equals(euler_azyx)


# PROPERTIES STUFF
def test_x():
    a = Euler()
    assert a.x == 0

    a = Euler(1, 2, 3)
    assert a.x == 1

    a = Euler(4, 5, 6, Euler.RotationOrders.XYZ)
    assert a.x == 4

    a = Euler(7, 8, 9, Euler.RotationOrders.XYZ)
    a.x = 10
    assert a.x == 10

    a = Euler(11, 12, 13, Euler.RotationOrders.XYZ)
    a.x = 14
    assert a.x == 14


def test_y():
    a = Euler()
    assert a.y == 0

    a = Euler(1, 2, 3)
    assert a.y == 2

    a = Euler(4, 5, 6, Euler.RotationOrders.XYZ)
    assert a.y == 5

    a = Euler(7, 8, 9, Euler.RotationOrders.XYZ)
    a.y = 10
    assert a.y == 10

    a = Euler(11, 12, 13, Euler.RotationOrders.XYZ)
    a.y = 14
    assert a.y == 14


def test_z():
    a = Euler()
    assert a.z == 0

    a = Euler(1, 2, 3)
    assert a.z == 3

    a = Euler(4, 5, 6, Euler.RotationOrders.XYZ)
    assert a.z == 6

    a = Euler(7, 8, 9, Euler.RotationOrders.XYZ)
    a.z = 10
    assert a.z == 10

    a = Euler(11, 12, 13, Euler.RotationOrders.XYZ)
    a.z = 14
    assert a.z == 14


def test_order():
    a = Euler()
    assert a.order == Euler.DefaultOrder

    a = Euler(1, 2, 3)
    assert a.order == Euler.DefaultOrder

    a = Euler(4, 5, 6, Euler.RotationOrders.YZX)
    assert a.order == Euler.RotationOrders.YZX

    a = Euler(7, 8, 9, Euler.RotationOrders.YZX)
    a.order = Euler.RotationOrders.ZXY
    assert a.order == Euler.RotationOrders.ZXY

    a = Euler(11, 12, 13, Euler.RotationOrders.YZX)
    a.order = Euler.RotationOrders.ZXY
    assert a.order == Euler.RotationOrders.ZXY


# PUBLIC STUFF


def test_set_from_vector3_to_vector3():
    a = Euler()

    a.set(0, 1, 0, Euler.RotationOrders.ZYX)
    assert a.equals(euler_azyx)
    assert not a.equals(euler_axyz)
    assert not a.equals(euler_zero)

    vec = Vector3(0, 1, 0)

    b = Euler().set_from_vector3(vec, Euler.RotationOrders.ZYX)
    assert a.equals(b)

    c = b.to_vector3()
    assert c.equals(vec)


def test_clone_copy_equals():
    a = euler_axyz.clone()
    assert a.equals(euler_axyz)
    assert not a.equals(euler_zero)
    assert not a.equals(euler_azyx)

    a.copy(euler_azyx)
    assert a.equals(euler_azyx)
    assert not a.equals(euler_axyz)
    assert not a.equals(euler_zero)


def test_quaternion_set_from_euler_euler_from_quaternion():
    test_values = [euler_zero, euler_axyz, euler_azyx]
    for i in range(len(test_values)):
        v = test_values[i]
        q = Quaternion().set_from_euler(v)

        v2 = Euler().set_from_quaternion(q, v.order)
        q2 = Quaternion().set_from_euler(v2)
        assert quat_equals(q, q2)


def test_matrix4_set_from_euler_euler_from_rotation_matrix():
    test_values = [euler_zero, euler_axyz, euler_azyx]
    for i in range(len(test_values)):

        v = test_values[i]
        m = Matrix4().make_rotation_from_euler(v)

        v2 = Euler().set_from_rotation_matrix(m, v.order)
        m2 = Matrix4().make_rotation_from_euler(v2)
        assert matrix_equals(m, m2, 0.0001)


def test_reorder():
    test_values = [euler_zero, euler_axyz, euler_azyx]
    for i in range(len(test_values)):
        v = test_values[i]
        q = Quaternion().set_from_euler(v)

        v.reorder(Euler.RotationOrders.YZX)
        q2 = Quaternion().set_from_euler(v)
        assert quat_equals(q, q2)

        v.reorder(Euler.RotationOrders.ZXY)
        q3 = Quaternion().set_from_euler(v)
        assert quat_equals(q, q3)


def test_to_array():
    order = Euler.RotationOrders.YXZ
    a = Euler(x, y, z, order)

    array = a.to_array()
    assert array[0] == x, "No array, no offset: check x"
    assert array[1] == y, "No array, no offset: check y"
    assert array[2] == z, "No array, no offset: check z"
    assert array[3] == order, "No array, no offset: check order"

    array = []
    a.to_array(array)
    assert array[0] == x, "With array, no offset: check x"
    assert array[1] == y, "With array, no offset: check y"
    assert array[2] == z, "With array, no offset: check z"
    assert array[3] == order, "With array, no offset: check order"

    array = []
    a.to_array(array, 1)
    assert array[0] is None, "With array and offset: check [0]"
    assert array[1] == x, "With array and offset: check x"
    assert array[2] == y, "With array and offset: check y"
    assert array[3] == z, "With array and offset: check z"
    assert array[4] == order, "With array and offset: check order"


def test_from_array():
    a = Euler()
    array = [x, y, z]

    a.from_array(array)
    assert a.x == x, "No order: check x"
    assert a.y == y, "No order: check y"
    assert a.z == z, "No order: check z"
    assert a.order == Euler.RotationOrders.XYZ, "No order: check order"

    a = Euler()
    array = [x, y, z, Euler.RotationOrders.ZXY]
    a.from_array(array)
    assert a.x == x, "With order: check x"
    assert a.y == y, "With order: check y"
    assert a.z == z, "With order: check z"
    assert a.order == Euler.RotationOrders.ZXY, "With order: check order"


# OTHERS
def test_gimbal_local_quat():
    # known problematic quaternions
    q1 = Quaternion(
        0.5207769385244341, -0.4783214164122354, 0.520776938524434, 0.47832141641223547
    )
    # q2 = Quaternion(
    #     0.11284905712620674,
    #     0.6980437630368944,
    #     -0.11284905712620674,
    #     0.6980437630368944,
    # )

    euler_order = Euler.RotationOrders.ZYX

    # create Euler directly from a Quaternion
    e_via_q1 = Euler().set_from_quaternion(
        q1, euler_order
    )  # there is likely a bug here

    # create Euler from Quaternion via an intermediate Matrix4
    m_via_q1 = Matrix4().make_rotation_from_quaternion(q1)
    e_via_m_via_q1 = Euler().set_from_rotation_matrix(m_via_q1, euler_order)

    # the results here are different
    assert euler_equals(e_via_q1, e_via_m_via_q1)  # this result is correcy
