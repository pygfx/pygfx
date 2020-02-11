from math import pi, sqrt, acos

import pytest

from visvis2 import PerspectiveCamera
from visvis2.linalg import Vector3, Euler, Matrix3, Matrix4, Vector4, Quaternion, Spherical, Cylindrical


x = 2
y = 3
z = 4
w = 5
eps = 0.0001


# INSTANCING
def test_instancing():
    a = Vector3()
    assert a.x == 0
    assert a.y == 0
    assert a.z == 0

    a = Vector3(x, y, z)
    assert a.x == x
    assert a.y == y
    assert a.z == z


# PUBLIC STUFF
@pytest.mark.xfail(reason="todo")
def test_is_vector3():
    assert False


def test_set():
    a = Vector3()
    assert a.x == 0
    assert a.y == 0
    assert a.z == 0

    a.set(x, y, z)
    assert a.x == x
    assert a.y == y
    assert a.z == z


@pytest.mark.xfail(reason="todo")
def test_set_scalar():
    assert False


@pytest.mark.xfail(reason="todo")
def test_set_x():
    assert False


@pytest.mark.xfail(reason="todo")
def test_set_y():
    assert False


@pytest.mark.xfail(reason="todo")
def test_set_z():
    assert False


@pytest.mark.xfail(reason="todo")
def test_set_component():
    assert False


@pytest.mark.xfail(reason="todo")
def test_get_component():
    assert False


@pytest.mark.xfail(reason="todo")
def test_clone():
    assert False


def test_copy():
    a = Vector3(x, y, z)
    b = Vector3().copy(a)
    assert b.x == x
    assert b.y == y
    assert b.z == z

    # ensure that it is a True copy
    a.x = 0
    a.y = -1
    a.z = -2
    assert b.x == x
    assert b.y == y
    assert b.z == z


def test_add():
    a = Vector3(x, y, z)
    b = Vector3(-x, -y, -z)

    a.add(b)
    assert a.x == 0
    assert a.y == 0
    assert a.z == 0

    c = Vector3().add_vectors(b, b)
    assert c.x == -2 * x
    assert c.y == -2 * y
    assert c.z == -2 * z


@pytest.mark.xfail(reason="todo")
def test_add_scalar():
    assert False


@pytest.mark.xfail(reason="todo")
def test_add_vectors():
    assert False


def test_add_scaled_vector():
    a = Vector3(x, y, z)
    b = Vector3(2, 3, 4)
    s = 3

    a.add_scaled_vector(b, s)
    assert a.x == x + b.x * s, "Check x"
    assert a.y == y + b.y * s, "Check y"
    assert a.z == z + b.z * s, "Check z"


def test_sub():
    a = Vector3(x, y, z)
    b = Vector3(-x, -y, -z)

    a.sub(b)
    assert a.x == 2 * x
    assert a.y == 2 * y
    assert a.z == 2 * z

    c = Vector3().sub_vectors(a, a)
    assert c.x == 0
    assert c.y == 0
    assert c.z == 0


@pytest.mark.xfail(reason="todo")
def test_sub_scalar():
    assert False


@pytest.mark.xfail(reason="todo")
def test_sub_vectors():
    assert False


@pytest.mark.xfail(reason="todo")
def test_multiply():
    assert False


@pytest.mark.xfail(reason="todo")
def test_multiply_scalar():
    assert False


def test_multiply_vectors():
    a = Vector3(x, y, z)
    b = Vector3(2, 3, -5)

    c = Vector3().multiply_vectors(a, b)
    assert c.x == x * 2, "Check x"
    assert c.y == y * 3, "Check y"
    assert c.z == z * -5, "Check z"


def test_apply_euler():
    a = Vector3(x, y, z)
    euler = Euler(90, -45, 0)
    expected = Vector3(-2.352970120501014, -4.7441750936226645, 0.9779234597246458)

    a.apply_euler(euler)
    assert abs(a.x - expected.x) <= eps, "Check x"
    assert abs(a.y - expected.y) <= eps, "Check y"
    assert abs(a.z - expected.z) <= eps, "Check z"


def test_apply_axis_angle():
    a = Vector3(x, y, z)
    axis = Vector3(0, 1, 0)
    angle = pi / 4.0
    expected = Vector3(3 * sqrt(2), 3, sqrt(2))

    a.apply_axis_angle(axis, angle)
    assert abs(a.x - expected.x) <= eps, "Check x"
    assert abs(a.y - expected.y) <= eps, "Check y"
    assert abs(a.z - expected.z) <= eps, "Check z"


def test_apply_matrix3():
    a = Vector3(x, y, z)
    m = Matrix3().set(2, 3, 5, 7, 11, 13, 17, 19, 23)

    a.apply_matrix3(m)
    assert a.x == 33, "Check x"
    assert a.y == 99, "Check y"
    assert a.z == 183, "Check z"


def test_apply_matrix4():
    a = Vector3(x, y, z)
    b = Vector4(x, y, z, 1)

    m = Matrix4().make_rotation_x(pi)
    a.apply_matrix4(m)
    b.apply_matrix4(m)
    assert a.x == b.x / b.w
    assert a.y == b.y / b.w
    assert a.z == b.z / b.w

    m = Matrix4().make_translation(3, 2, 1)
    a.apply_matrix4(m)
    b.apply_matrix4(m)
    assert a.x == b.x / b.w
    assert a.y == b.y / b.w
    assert a.z == b.z / b.w

    m = Matrix4().set(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0)
    a.apply_matrix4(m)
    b.apply_matrix4(m)
    assert a.x == b.x / b.w
    assert a.y == b.y / b.w
    assert a.z == b.z / b.w


def test_apply_quaternion():
    a = Vector3(x, y, z)

    a.apply_quaternion(Quaternion())
    assert a.x == x, "Identity rotation: check x"
    assert a.y == y, "Identity rotation: check y"
    assert a.z == z, "Identity rotation: check z"

    a.apply_quaternion(Quaternion(x, y, z, w))
    assert a.x == 108, "Normal rotation: check x"
    assert a.y == 162, "Normal rotation: check y"
    assert a.z == 216, "Normal rotation: check z"


@pytest.mark.xfail(reason="todo")
def test_project():
    assert False


@pytest.mark.xfail(reason="todo")
def test_unproject():
    assert False


def test_transform_direction():
    a = Vector3(x, y, z)
    m = Matrix4()
    transformed = Vector3(0.3713906763541037, 0.5570860145311556, 0.7427813527082074)

    a.transform_direction(m)
    assert abs(a.x - transformed.x) <= eps, "Check x"
    assert abs(a.y - transformed.y) <= eps, "Check y"
    assert abs(a.z - transformed.z) <= eps, "Check z"


@pytest.mark.xfail(reason="todo")
def test_divide():
    assert False


@pytest.mark.xfail(reason="todo")
def test_divide_scalar():
    assert False


@pytest.mark.xfail(reason="todo")
def test_min():
    assert False


@pytest.mark.xfail(reason="todo")
def test_max():
    assert False


@pytest.mark.xfail(reason="todo")
def test_clamp():
    assert False


def test_clamp_scalar():
    a = Vector3(-0.01, 0.5, 1.5)
    clamped = Vector3(0.1, 0.5, 1.0)

    a.clamp_scalar(0.1, 1.0)
    assert abs(a.x - clamped.x) <= 0.001, "Check x"
    assert abs(a.y - clamped.y) <= 0.001, "Check y"
    assert abs(a.z - clamped.z) <= 0.001, "Check z"


@pytest.mark.xfail(reason="todo")
def test_clamp_length():
    assert False


@pytest.mark.xfail(reason="todo")
def test_floor():
    assert False


@pytest.mark.xfail(reason="todo")
def test_ceil():
    assert False


@pytest.mark.xfail(reason="todo")
def test_round():
    assert False


@pytest.mark.xfail(reason="todo")
def test_round_to_zero():
    assert False


def test_negate():
    a = Vector3(x, y, z)

    a.negate()
    assert a.x == -x
    assert a.y == -y
    assert a.z == -z


def test_dot():
    a = Vector3(x, y, z)
    b = Vector3(-x, -y, -z)
    c = Vector3()

    result = a.dot(b)
    assert result == (-x * x - y * y - z * z)

    result = a.dot(c)
    assert result == 0


@pytest.mark.xfail(reason="todo")
def test_length_sq():
    assert False


@pytest.mark.xfail(reason="todo")
def test_length():
    assert False


def test_manhattan_length():
    a = Vector3(x, 0, 0)
    b = Vector3(0, -y, 0)
    c = Vector3(0, 0, z)
    d = Vector3()

    assert a.manhattan_length() == x, "Positive x"
    assert b.manhattan_length() == y, "Negative y"
    assert c.manhattan_length() == z, "Positive z"
    assert d.manhattan_length() == 0, "Empty initialization"

    a.set(x, y, z)
    assert a.manhattan_length() == abs(x) + abs(y) + abs(
        z
    ), "All components"


def test_normalize():
    a = Vector3(x, 0, 0)
    b = Vector3(0, -y, 0)
    c = Vector3(0, 0, z)

    a.normalize()
    assert a.length() == 1
    assert a.x == 1

    b.normalize()
    assert b.length() == 1
    assert b.y == -1

    c.normalize()
    assert c.length() == 1
    assert c.z == 1


def test_set_length():
    a = Vector3(x, 0, 0)

    assert a.length() == x
    a.set_length(y)
    assert a.length() == y

    a = Vector3(0, 0, 0)
    assert a.length() == 0
    a.set_length(y)
    assert a.length() == 0
    with pytest.raises(TypeError):
        a.set_length()


@pytest.mark.xfail(reason="todo")
def test_lerp():
    assert False


@pytest.mark.xfail(reason="todo")
def test_lerp_vectors():
    assert False


def test_cross():
    a = Vector3(x, y, z)
    b = Vector3(2 * x, -y, 0.5 * z)
    crossed = Vector3(18, 12, -18)

    a.cross(b)
    assert abs(a.x - crossed.x) <= eps, "Check x"
    assert abs(a.y - crossed.y) <= eps, "Check y"
    assert abs(a.z - crossed.z) <= eps, "Check z"


def test_cross_vectors():
    a = Vector3(x, y, z)
    b = Vector3(x, -y, z)
    c = Vector3()
    crossed = Vector3(24, 0, -12)

    c.cross_vectors(a, b)
    assert abs(c.x - crossed.x) <= eps, "Check x"
    assert abs(c.y - crossed.y) <= eps, "Check y"
    assert abs(c.z - crossed.z) <= eps, "Check z"


def test_project_on_vector():
    a = Vector3(1, 0, 0)
    b = Vector3()
    normal = Vector3(10, 0, 0)

    assert b.copy(a).project_on_vector(normal).equals(Vector3(1, 0, 0))

    a.set(0, 1, 0)
    assert b.copy(a).project_on_vector(normal).equals(Vector3(0, 0, 0))

    a.set(0, 0, -1)
    assert b.copy(a).project_on_vector(normal).equals(Vector3(0, 0, 0))

    a.set(-1, 0, 0)
    assert b.copy(a).project_on_vector(normal).equals(Vector3(-1, 0, 0))


def test_project_on_plane():
    a = Vector3(1, 0, 0)
    b = Vector3()
    normal = Vector3(1, 0, 0)

    assert b.copy(a).project_on_plane(normal).equals(Vector3(0, 0, 0))

    a.set(0, 1, 0)
    assert b.copy(a).project_on_plane(normal).equals(Vector3(0, 1, 0))

    a.set(0, 0, -1)
    assert b.copy(a).project_on_plane(normal).equals(Vector3(0, 0, -1))

    a.set(-1, 0, 0)
    assert b.copy(a).project_on_plane(normal).equals(Vector3(0, 0, 0))


def test_reflect():
    a = Vector3()
    normal = Vector3(0, 1, 0)
    b = Vector3()

    a.set(0, -1, 0)
    assert b.copy(a).reflect(normal).equals(Vector3(0, 1, 0))

    a.set(1, -1, 0)
    assert b.copy(a).reflect(normal).equals(Vector3(1, 1, 0))

    a.set(1, -1, 0)
    normal.set(0, -1, 0)
    assert b.copy(a).reflect(normal).equals(Vector3(1, 1, 0))


def test_angle_to():
    a = Vector3(0, -0.18851655680720186, 0.9820700116639124)
    b = Vector3(0, 0.18851655680720186, -0.9820700116639124)

    assert a.angle_to(a) == 0
    assert a.angle_to(b) == pi

    x = Vector3(1, 0, 0)
    y = Vector3(0, 1, 0)
    z = Vector3(0, 0, 1)

    assert x.angle_to(y) == pi / 2
    assert x.angle_to(z) == pi / 2
    assert z.angle_to(x) == pi / 2

    assert abs(x.angle_to(Vector3(1, 1, 0)) - (pi / 4)) < 0.0000001


@pytest.mark.xfail(reason="todo")
def test_distance_to():
    assert False


@pytest.mark.xfail(reason="todo")
def test_distance_to_squared():
    assert False


@pytest.mark.xfail(reason="todo")
def test_manhattan_distance_to():
    assert False


def test_set_from_spherical():
    a = Vector3()
    phi = acos(-0.5)
    theta = sqrt(pi) * phi
    sph = Spherical(10, phi, theta)
    expected = Vector3(-4.677914006701843, -5, -7.288149322420796)

    a.set_from_spherical(sph)
    assert abs(a.x - expected.x) <= eps, "Check x"
    assert abs(a.y - expected.y) <= eps, "Check y"
    assert abs(a.z - expected.z) <= eps, "Check z"


def test_set_from_cylindrical():
    a = Vector3()
    cyl = Cylindrical(10, pi * 0.125, 20)
    expected = Vector3(3.826834323650898, 20, 9.238795325112868)

    a.set_from_cylindrical(cyl)
    assert abs(a.x - expected.x) <= eps, "Check x"
    assert abs(a.y - expected.y) <= eps, "Check y"
    assert abs(a.z - expected.z) <= eps, "Check z"


def test_set_from_matrix_position():
    a = Vector3()
    m = Matrix4().set(2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53)

    a.set_from_matrix_position(m)
    assert a.x == 7, "Check x"
    assert a.y == 19, "Check y"
    assert a.z == 37, "Check z"


def test_set_from_matrix_scale():
    a = Vector3()
    m = Matrix4().set(2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53)
    expected = Vector3(25.573423705088842, 31.921779399024736, 35.70714214271425)

    a.set_from_matrix_scale(m)
    assert abs(a.x - expected.x) <= eps, "Check x"
    assert abs(a.y - expected.y) <= eps, "Check y"
    assert abs(a.z - expected.z) <= eps, "Check z"


def test_set_from_matrix_column():
    a = Vector3()
    m = Matrix4().set(2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53)

    a.set_from_matrix_column(m, 0)
    assert a.x == 2, "Index 0: check x"
    assert a.y == 11, "Index 0: check y"
    assert a.z == 23, "Index 0: check z"

    a.set_from_matrix_column(m, 2)
    assert a.x == 5, "Index 2: check x"
    assert a.y == 17, "Index 2: check y"
    assert a.z == 31, "Index 2: check z"


def test_equals():
    a = Vector3(x, 0, z)
    b = Vector3(0, -y, 0)

    assert a.x != b.x
    assert a.y != b.y
    assert a.z != b.z

    assert not a.equals(b)
    assert not b.equals(a)

    a.copy(b)
    assert a.x == b.x
    assert a.y == b.y
    assert a.z == b.z

    assert a.equals(b)
    assert b.equals(a)


def test_from_array():
    a = Vector3()
    array = [1, 2, 3, 4, 5, 6]

    a.from_array(array)
    assert a.x == 1, "No offset: check x"
    assert a.y == 2, "No offset: check y"
    assert a.z == 3, "No offset: check z"

    a.from_array(array, 3)
    assert a.x == 4, "With offset: check x"
    assert a.y == 5, "With offset: check y"
    assert a.z == 6, "With offset: check z"


def test_to_array():
    a = Vector3(x, y, z)

    array = a.to_array()
    assert array[0], x == "No array,  no offset: check x"
    assert array[1], y == "No array,  no offset: check y"
    assert array[2], z == "No array,  no offset: check z"

    array = []
    a.to_array(array)
    assert array[0], x == "With array,  no offset: check x"
    assert array[1], y == "With array,  no offset: check y"
    assert array[2], z == "With array,  no offset: check z"

    array = []
    a.to_array(array, 1)
    assert array[0] is None, "With array and offset: check [0]"
    assert array[1] == x, "With array and offset: check x"
    assert array[2] == y, "With array and offset: check y"
    assert array[3] == z, "With array and offset: check z"


@pytest.mark.xfail(reason="todo")
def test_from_buffer_attribute():
    a = Vector3()
    attr = BufferAttribute(Float32Array([1, 2, 3, 4, 5, 6]), 3)

    a.from_buffer_attribute(attr, 0)
    assert a.x == 1, "Offset 0: check x"
    assert a.y == 2, "Offset 0: check y"
    assert a.z == 3, "Offset 0: check z"

    a.from_buffer_attribute(attr, 1)
    assert a.x == 4, "Offset 1: check x"
    assert a.y == 5, "Offset 1: check y"
    assert a.z == 6, "Offset 1: check z"


# TODO (Itee) refactor/split
def test_set_xset_yset_z():
    a = Vector3()
    assert a.x == 0
    assert a.y == 0
    assert a.z == 0

    a.set_x(x)
    a.set_y(y)
    a.set_z(z)

    assert a.x == x
    assert a.y == y
    assert a.z == z


def test_set_componentget_component():
    a = Vector3()
    assert a.x == 0
    assert a.y == 0
    assert a.z == 0

    a.set_component(0, 1)
    a.set_component(1, 2)
    a.set_component(2, 3)
    assert a.get_component(0) == 1
    assert a.get_component(1) == 2
    assert a.get_component(2) == 3


def test_set_componentget_componentexceptions():
    a = Vector3()

    with pytest.raises(IndexError):
        a.set_component(3, 0)

    with pytest.raises(IndexError):
        a.get_component(3)


def test_minmaxclamp():
    a = Vector3(x, y, z)
    b = Vector3(-x, -y, -z)
    c = Vector3()

    c.copy(a).min(b)
    assert c.x == -x
    assert c.y == -y
    assert c.z == -z

    c.copy(a).max(b)
    assert c.x == x
    assert c.y == y
    assert c.z == z

    c.set(-2 * x, 2 * y, -2 * z)
    c.clamp(b, a)
    assert c.x == -x
    assert c.y == y
    assert c.z == -z


def test_distance_todistance_to_squared():
    a = Vector3(x, 0, 0)
    b = Vector3(0, -y, 0)
    c = Vector3(0, 0, z)
    d = Vector3()

    assert a.distance_to(d) == x
    assert a.distance_to_squared(d) == x * x

    assert b.distance_to(d) == y
    assert b.distance_to_squared(d) == y * y

    assert c.distance_to(d) == z
    assert c.distance_to_squared(d) == z * z


def test_set_scalaradd_scalarsub_scalar():
    a = Vector3()
    s = 3

    a.set_scalar(s)
    assert a.x == s, "set_scalar: check x"
    assert a.y == s, "set_scalar: check y"
    assert a.z == s, "set_scalar: check z"

    a.add_scalar(s)
    assert a.x == 2 * s, "add_scalar: check x"
    assert a.y == 2 * s, "add_scalar: check y"
    assert a.z == 2 * s, "add_scalar: check z"

    a.sub_scalar(2 * s)
    assert a.x == 0, "sub_scalar: check x"
    assert a.y == 0, "sub_scalar: check y"
    assert a.z == 0, "sub_scalar: check z"


def test_multiplydivide():
    a = Vector3(x, y, z)
    b = Vector3(2 * x, 2 * y, 2 * z)
    c = Vector3(4 * x, 4 * y, 4 * z)

    a.multiply(b)
    assert a.x == x * b.x, "multiply: check x"
    assert a.y == y * b.y, "multiply: check y"
    assert a.z == z * b.z, "multiply: check z"

    b.divide(c)
    assert abs(b.x - 0.5) <= eps, "divide: check z"
    assert abs(b.y - 0.5) <= eps, "divide: check z"
    assert abs(b.z - 0.5) <= eps, "divide: check z"


def test_multiplydivide2():
    a = Vector3(x, y, z)
    b = Vector3(-x, -y, -z)

    a.multiply_scalar(-2)
    assert a.x == x * -2
    assert a.y == y * -2
    assert a.z == z * -2

    b.multiply_scalar(-2)
    assert b.x == 2 * x
    assert b.y == 2 * y
    assert b.z == 2 * z

    a.divide_scalar(-2)
    assert a.x == x
    assert a.y == y
    assert a.z == z

    b.divide_scalar(-2)
    assert b.x == -x
    assert b.y == -y
    assert b.z == -z


def test_projectunproject():
    a = Vector3(x, y, z)
    camera = PerspectiveCamera(75, 16 / 9, 0.1, 300.0)
    projected = Vector3(-0.36653213611158914, -0.9774190296309043, 1.0506835611870624)

    a.project(camera)
    assert abs(a.x - projected.x) <= eps, "project: check x"
    assert abs(a.y - projected.y) <= eps, "project: check y"
    assert abs(a.z - projected.z) <= eps, "project: check z"

    a.unproject(camera)
    assert abs(a.x - x) <= eps, "unproject: check x"
    assert abs(a.y - y) <= eps, "unproject: check y"
    assert abs(a.z - z) <= eps, "unproject: check z"


def test_lengthlength_sq():
    a = Vector3(x, 0, 0)
    b = Vector3(0, -y, 0)
    c = Vector3(0, 0, z)
    d = Vector3()

    assert a.length() == x
    assert a.length_sq() == x * x
    assert b.length() == y
    assert b.length_sq() == y * y
    assert c.length() == z
    assert c.length_sq() == z * z
    assert d.length() == 0
    assert d.length_sq() == 0

    a.set(x, y, z)
    assert a.length() == sqrt(x * x + y * y + z * z)
    assert a.length_sq() == (x * x + y * y + z * z)


def test_lerpclone():
    a = Vector3(x, 0, z)
    b = Vector3(0, -y, 0)

    assert a.lerp(a, 0).equals(a.lerp(a, 0.5))
    assert a.lerp(a, 0).equals(a.lerp(a, 1))

    assert a.clone().lerp(b, 0).equals(a)

    assert a.clone().lerp(b, 0.5).x == x * 0.5
    assert a.clone().lerp(b, 0.5).y == -y * 0.5
    assert a.clone().lerp(b, 0.5).z == z * 0.5

    assert a.clone().lerp(b, 1).equals(b)
