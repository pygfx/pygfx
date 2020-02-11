from math import pi, sqrt, acos

import pytest

from visvis2 import PerspectiveCamera
from visvis2.linalg import Vector3, Euler, Matrix3, Matrix4, Vector4, Quaternion, Spherical, Cylindrical


x = 2
y = 3
z = 4
w = 5
eps = 0.0001


def test_instancing():
    v1 = Vector3()
    assert v1.x == 0
    assert v1.y == 0
    assert v1.z == 0

    x, y, z = 1, 2, 3
    v2 = Vector3(x, y, z)
    assert v2.x == x
    assert v2.y == y
    assert v2.z == z


# INSTANCING
def test_Instancing():

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
def test_isVector3():

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
def test_setScalar():

    assert False


@pytest.mark.xfail(reason="todo")
def test_setX():

    assert False


@pytest.mark.xfail(reason="todo")
def test_setY():

    assert False


@pytest.mark.xfail(reason="todo")
def test_setZ():

    assert False


@pytest.mark.xfail(reason="todo")
def test_setComponent():

    assert False


@pytest.mark.xfail(reason="todo")
def test_getComponent():

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

    c = Vector3().addVectors(b, b)
    assert c.x == -2 * x
    assert c.y == -2 * y
    assert c.z == -2 * z


@pytest.mark.xfail(reason="todo")
def test_addScalar():

    assert False


@pytest.mark.xfail(reason="todo")
def test_addVectors():

    assert False


def test_addScaledVector():

    a = Vector3(x, y, z)
    b = Vector3(2, 3, 4)
    s = 3

    a.addScaledVector(b, s)
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

    c = Vector3().subVectors(a, a)
    assert c.x == 0
    assert c.y == 0
    assert c.z == 0


@pytest.mark.xfail(reason="todo")
def test_subScalar():

    assert False


@pytest.mark.xfail(reason="todo")
def test_subVectors():

    assert False


@pytest.mark.xfail(reason="todo")
def test_multiply():

    assert False


@pytest.mark.xfail(reason="todo")
def test_multiplyScalar():

    assert False


def test_multiplyVectors():

    a = Vector3(x, y, z)
    b = Vector3(2, 3, -5)

    c = Vector3().multiplyVectors(a, b)
    assert c.x == x * 2, "Check x"
    assert c.y == y * 3, "Check y"
    assert c.z == z * -5, "Check z"


def test_applyEuler():

    a = Vector3(x, y, z)
    euler = Euler(90, -45, 0)
    expected = Vector3(-2.352970120501014, -4.7441750936226645, 0.9779234597246458)

    a.applyEuler(euler)
    assert abs(a.x - expected.x) <= eps, "Check x"
    assert abs(a.y - expected.y) <= eps, "Check y"
    assert abs(a.z - expected.z) <= eps, "Check z"


def test_applyAxisAngle():

    a = Vector3(x, y, z)
    axis = Vector3(0, 1, 0)
    angle = pi / 4.0
    expected = Vector3(3 * sqrt(2), 3, sqrt(2))

    a.applyAxisAngle(axis, angle)
    assert abs(a.x - expected.x) <= eps, "Check x"
    assert abs(a.y - expected.y) <= eps, "Check y"
    assert abs(a.z - expected.z) <= eps, "Check z"


def test_applyMatrix3():

    a = Vector3(x, y, z)
    m = Matrix3().set(2, 3, 5, 7, 11, 13, 17, 19, 23)

    a.applyMatrix3(m)
    assert a.x == 33, "Check x"
    assert a.y == 99, "Check y"
    assert a.z == 183, "Check z"


def test_applyMatrix4():

    a = Vector3(x, y, z)
    b = Vector4(x, y, z, 1)

    m = Matrix4().makeRotationX(pi)
    a.applyMatrix4(m)
    b.applyMatrix4(m)
    assert a.x == b.x / b.w
    assert a.y == b.y / b.w
    assert a.z == b.z / b.w

    m = Matrix4().makeTranslation(3, 2, 1)
    a.applyMatrix4(m)
    b.applyMatrix4(m)
    assert a.x == b.x / b.w
    assert a.y == b.y / b.w
    assert a.z == b.z / b.w

    m = Matrix4().set(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0)
    a.applyMatrix4(m)
    b.applyMatrix4(m)
    assert a.x == b.x / b.w
    assert a.y == b.y / b.w
    assert a.z == b.z / b.w


def test_applyQuaternion():

    a = Vector3(x, y, z)

    a.applyQuaternion(Quaternion())
    assert a.x == x, "Identity rotation: check x"
    assert a.y == y, "Identity rotation: check y"
    assert a.z == z, "Identity rotation: check z"

    a.applyQuaternion(Quaternion(x, y, z, w))
    assert a.x == 108, "Normal rotation: check x"
    assert a.y == 162, "Normal rotation: check y"
    assert a.z == 216, "Normal rotation: check z"


@pytest.mark.xfail(reason="todo")
def test_project():

    assert False


@pytest.mark.xfail(reason="todo")
def test_unproject():

    assert False


def test_transformDirection():

    a = Vector3(x, y, z)
    m = Matrix4()
    transformed = Vector3(0.3713906763541037, 0.5570860145311556, 0.7427813527082074)

    a.transformDirection(m)
    assert abs(a.x - transformed.x) <= eps, "Check x"
    assert abs(a.y - transformed.y) <= eps, "Check y"
    assert abs(a.z - transformed.z) <= eps, "Check z"


@pytest.mark.xfail(reason="todo")
def test_divide():

    assert False


@pytest.mark.xfail(reason="todo")
def test_divideScalar():

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


def test_clampScalar():

    a = Vector3(-0.01, 0.5, 1.5)
    clamped = Vector3(0.1, 0.5, 1.0)

    a.clampScalar(0.1, 1.0)
    assert abs(a.x - clamped.x) <= 0.001, "Check x"
    assert abs(a.y - clamped.y) <= 0.001, "Check y"
    assert abs(a.z - clamped.z) <= 0.001, "Check z"


@pytest.mark.xfail(reason="todo")
def test_clampLength():

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
def test_roundToZero():

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
def test_lengthSq():

    assert False


@pytest.mark.xfail(reason="todo")
def test_length():

    assert False


def test_manhattanLength():

    a = Vector3(x, 0, 0)
    b = Vector3(0, -y, 0)
    c = Vector3(0, 0, z)
    d = Vector3()

    assert a.manhattanLength() == x, "Positive x"
    assert b.manhattanLength() == y, "Negative y"
    assert c.manhattanLength() == z, "Positive z"
    assert d.manhattanLength() == 0, "Empty initialization"

    a.set(x, y, z)
    assert a.manhattanLength() == abs(x) + abs(y) + abs(
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


def test_setLength():

    a = Vector3(x, 0, 0)

    assert a.length() == x
    a.setLength(y)
    assert a.length() == y

    a = Vector3(0, 0, 0)
    assert a.length() == 0
    a.setLength(y)
    assert a.length() == 0
    with pytest.raises(TypeError):
        a.setLength()


@pytest.mark.xfail(reason="todo")
def test_lerp():

    assert False


@pytest.mark.xfail(reason="todo")
def test_lerpVectors():

    assert False


def test_cross():

    a = Vector3(x, y, z)
    b = Vector3(2 * x, -y, 0.5 * z)
    crossed = Vector3(18, 12, -18)

    a.cross(b)
    assert abs(a.x - crossed.x) <= eps, "Check x"
    assert abs(a.y - crossed.y) <= eps, "Check y"
    assert abs(a.z - crossed.z) <= eps, "Check z"


def test_crossVectors():

    a = Vector3(x, y, z)
    b = Vector3(x, -y, z)
    c = Vector3()
    crossed = Vector3(24, 0, -12)

    c.crossVectors(a, b)
    assert abs(c.x - crossed.x) <= eps, "Check x"
    assert abs(c.y - crossed.y) <= eps, "Check y"
    assert abs(c.z - crossed.z) <= eps, "Check z"


def test_projectOnVector():

    a = Vector3(1, 0, 0)
    b = Vector3()
    normal = Vector3(10, 0, 0)

    assert b.copy(a).projectOnVector(normal).equals(Vector3(1, 0, 0))

    a.set(0, 1, 0)
    assert b.copy(a).projectOnVector(normal).equals(Vector3(0, 0, 0))

    a.set(0, 0, -1)
    assert b.copy(a).projectOnVector(normal).equals(Vector3(0, 0, 0))

    a.set(-1, 0, 0)
    assert b.copy(a).projectOnVector(normal).equals(Vector3(-1, 0, 0))


def test_projectOnPlane():

    a = Vector3(1, 0, 0)
    b = Vector3()
    normal = Vector3(1, 0, 0)

    assert b.copy(a).projectOnPlane(normal).equals(Vector3(0, 0, 0))

    a.set(0, 1, 0)
    assert b.copy(a).projectOnPlane(normal).equals(Vector3(0, 1, 0))

    a.set(0, 0, -1)
    assert b.copy(a).projectOnPlane(normal).equals(Vector3(0, 0, -1))

    a.set(-1, 0, 0)
    assert b.copy(a).projectOnPlane(normal).equals(Vector3(0, 0, 0))


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


def test_angleTo():

    a = Vector3(0, -0.18851655680720186, 0.9820700116639124)
    b = Vector3(0, 0.18851655680720186, -0.9820700116639124)

    assert a.angleTo(a) == 0
    assert a.angleTo(b) == pi

    x = Vector3(1, 0, 0)
    y = Vector3(0, 1, 0)
    z = Vector3(0, 0, 1)

    assert x.angleTo(y) == pi / 2
    assert x.angleTo(z) == pi / 2
    assert z.angleTo(x) == pi / 2

    assert abs(x.angleTo(Vector3(1, 1, 0)) - (pi / 4)) < 0.0000001


@pytest.mark.xfail(reason="todo")
def test_distanceTo():

    assert False


@pytest.mark.xfail(reason="todo")
def test_distanceToSquared():

    assert False


@pytest.mark.xfail(reason="todo")
def test_manhattanDistanceTo():

    assert False


def test_setFromSpherical():

    a = Vector3()
    phi = acos(-0.5)
    theta = sqrt(pi) * phi
    sph = Spherical(10, phi, theta)
    expected = Vector3(-4.677914006701843, -5, -7.288149322420796)

    a.setFromSpherical(sph)
    assert abs(a.x - expected.x) <= eps, "Check x"
    assert abs(a.y - expected.y) <= eps, "Check y"
    assert abs(a.z - expected.z) <= eps, "Check z"


def test_setFromCylindrical():

    a = Vector3()
    cyl = Cylindrical(10, pi * 0.125, 20)
    expected = Vector3(3.826834323650898, 20, 9.238795325112868)

    a.setFromCylindrical(cyl)
    assert abs(a.x - expected.x) <= eps, "Check x"
    assert abs(a.y - expected.y) <= eps, "Check y"
    assert abs(a.z - expected.z) <= eps, "Check z"


def test_setFromMatrixPosition():

    a = Vector3()
    m = Matrix4().set(2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53)

    a.setFromMatrixPosition(m)
    assert a.x == 7, "Check x"
    assert a.y == 19, "Check y"
    assert a.z == 37, "Check z"


def test_setFromMatrixScale():

    a = Vector3()
    m = Matrix4().set(2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53)
    expected = Vector3(25.573423705088842, 31.921779399024736, 35.70714214271425)

    a.setFromMatrixScale(m)
    assert abs(a.x - expected.x) <= eps, "Check x"
    assert abs(a.y - expected.y) <= eps, "Check y"
    assert abs(a.z - expected.z) <= eps, "Check z"


def test_setFromMatrixColumn():

    a = Vector3()
    m = Matrix4().set(2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53)

    a.setFromMatrixColumn(m, 0)
    assert a.x == 2, "Index 0: check x"
    assert a.y == 11, "Index 0: check y"
    assert a.z == 23, "Index 0: check z"

    a.setFromMatrixColumn(m, 2)
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


def test_fromArray():

    a = Vector3()
    array = [1, 2, 3, 4, 5, 6]

    a.fromArray(array)
    assert a.x == 1, "No offset: check x"
    assert a.y == 2, "No offset: check y"
    assert a.z == 3, "No offset: check z"

    a.fromArray(array, 3)
    assert a.x == 4, "With offset: check x"
    assert a.y == 5, "With offset: check y"
    assert a.z == 6, "With offset: check z"


def test_toArray():

    a = Vector3(x, y, z)

    array = a.toArray()
    assert array[0], x == "No array,  no offset: check x"
    assert array[1], y == "No array,  no offset: check y"
    assert array[2], z == "No array,  no offset: check z"

    array = []
    a.toArray(array)
    assert array[0], x == "With array,  no offset: check x"
    assert array[1], y == "With array,  no offset: check y"
    assert array[2], z == "With array,  no offset: check z"

    array = []
    a.toArray(array, 1)
    assert array[0] == None, "With array and offset: check [0]"
    assert array[1] == x, "With array and offset: check x"
    assert array[2] == y, "With array and offset: check y"
    assert array[3] == z, "With array and offset: check z"


@pytest.mark.xfail(reason="todo")
def test_fromBufferAttribute():
    a = Vector3()
    attr = BufferAttribute(Float32Array([1, 2, 3, 4, 5, 6]), 3)

    a.fromBufferAttribute(attr, 0)
    assert a.x == 1, "Offset 0: check x"
    assert a.y == 2, "Offset 0: check y"
    assert a.z == 3, "Offset 0: check z"

    a.fromBufferAttribute(attr, 1)
    assert a.x == 4, "Offset 1: check x"
    assert a.y == 5, "Offset 1: check y"
    assert a.z == 6, "Offset 1: check z"


# TODO (Itee) refactor/split
def test_setXsetYsetZ():

    a = Vector3()
    assert a.x == 0
    assert a.y == 0
    assert a.z == 0

    a.setX(x)
    a.setY(y)
    a.setZ(z)

    assert a.x == x
    assert a.y == y
    assert a.z == z


def test_setComponentgetComponent():

    a = Vector3()
    assert a.x == 0
    assert a.y == 0
    assert a.z == 0

    a.setComponent(0, 1)
    a.setComponent(1, 2)
    a.setComponent(2, 3)
    assert a.getComponent(0) == 1
    assert a.getComponent(1) == 2
    assert a.getComponent(2) == 3


def test_setComponentgetComponentexceptions():

    a = Vector3()

    with pytest.raises(IndexError):
        a.setComponent(3, 0)

    with pytest.raises(IndexError):
        a.getComponent(3)


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


def test_distanceTodistanceToSquared():

    a = Vector3(x, 0, 0)
    b = Vector3(0, -y, 0)
    c = Vector3(0, 0, z)
    d = Vector3()

    assert a.distanceTo(d) == x
    assert a.distanceToSquared(d) == x * x

    assert b.distanceTo(d) == y
    assert b.distanceToSquared(d) == y * y

    assert c.distanceTo(d) == z
    assert c.distanceToSquared(d) == z * z


def test_setScalaraddScalarsubScalar():

    a = Vector3()
    s = 3

    a.setScalar(s)
    assert a.x == s, "setScalar: check x"
    assert a.y == s, "setScalar: check y"
    assert a.z == s, "setScalar: check z"

    a.addScalar(s)
    assert a.x == 2 * s, "addScalar: check x"
    assert a.y == 2 * s, "addScalar: check y"
    assert a.z == 2 * s, "addScalar: check z"

    a.subScalar(2 * s)
    assert a.x == 0, "subScalar: check x"
    assert a.y == 0, "subScalar: check y"
    assert a.z == 0, "subScalar: check z"


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

    a.multiplyScalar(-2)
    assert a.x == x * -2
    assert a.y == y * -2
    assert a.z == z * -2

    b.multiplyScalar(-2)
    assert b.x == 2 * x
    assert b.y == 2 * y
    assert b.z == 2 * z

    a.divideScalar(-2)
    assert a.x == x
    assert a.y == y
    assert a.z == z

    b.divideScalar(-2)
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


def test_lengthlengthSq():

    a = Vector3(x, 0, 0)
    b = Vector3(0, -y, 0)
    c = Vector3(0, 0, z)
    d = Vector3()

    assert a.length() == x
    assert a.lengthSq() == x * x
    assert b.length() == y
    assert b.lengthSq() == y * y
    assert c.length() == z
    assert c.lengthSq() == z * z
    assert d.length() == 0
    assert d.lengthSq() == 0

    a.set(x, y, z)
    assert a.length() == sqrt(x * x + y * y + z * z)
    assert a.lengthSq() == (x * x + y * y + z * z)


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
