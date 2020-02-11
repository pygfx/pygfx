from math import acos, sin, cos

from .utils import clamp
from .quaternion import Quaternion


class Vector3:
    def __init__(self, x: float = 0, y: float = 0, z: float = 0) -> None:
        self.x = x
        self.y = y
        self.z = z

    def __repr__(self) -> str:
        return f"Vector3({self.x}, {self.y}, {self.z})"

    def set(self, x: float, y: float, z: float) -> "Vector3":
        self.x = x
        self.y = y
        self.z = z
        return self

    def setScalar(self, s: float) -> "Vector3":
        self.x = s
        self.y = s
        self.z = s
        return self

    def setX(self, x: float) -> "Vector3":
        self.x = x
        return self

    def setY(self, y: float) -> "Vector3":
        self.y = y
        return self

    def setZ(self, z: float) -> "Vector3":
        self.z = z
        return self

    def setComponent(self, index: int, value: float) -> "Vector3":
        if index == 0:
            self.x = value
        elif index == 1:
            self.y = value
        elif index == 2:
            self.z = value
        else:
            raise IndexError()
        return self

    def getComponent(self, index: int) -> float:
        if index == 0:
            return self.x
        elif index == 1:
            return self.y
        elif index == 2:
            return self.z
        else:
            raise IndexError()

    def clone(self) -> "Vector3":
        return Vector3(self.x, self.y, self.z)

    def copy(self, v: "Vector3") -> "Vector3":
        self.x = v.x
        self.y = v.y
        self.z = v.z
        return self

    def add(self, v: "Vector3") -> "Vector3":
        self.x += v.x
        self.y += v.y
        self.z += v.z
        return self

    def addScalar(self, s: float) -> "Vector3":
        self.x += s
        self.y += s
        self.z += s
        return self

    def addVectors(self, a: "Vector3", b: "Vector3") -> "Vector3":
        self.x = a.x + b.x
        self.y = a.y + b.y
        self.z = a.z + b.z
        return self

    def addScaledVector(self, v: "Vector3", s: float) -> "Vector3":
        self.x += v.x * s
        self.y += v.y * s
        self.z += v.z * s
        return self

    def sub(self, v: "Vector3") -> "Vector3":
        self.x -= v.x
        self.y -= v.y
        self.z -= v.z
        return self

    def subScalar(self, s: float) -> "Vector3":
        self.x -= s
        self.y -= s
        self.z -= s
        return self

    def subVectors(self, a: "Vector3", b: "Vector3") -> "Vector3":
        self.x = a.x - b.x
        self.y = a.y - b.y
        self.z = a.z - b.z
        return self

    def multiply(self, v: "Vector3") -> "Vector3":
        self.x *= v.x
        self.y *= v.y
        self.z *= v.z
        return self

    def multiplyScalar(self, s: float) -> "Vector3":
        self.x *= s
        self.y *= s
        self.z *= s
        return self

    def multiplyVectors(self, a: "Vector3", b: "Vector3") -> "Vector3":
        self.x = a.x * b.x
        self.y = a.y * b.y
        self.z = a.z * b.z
        return self

    def applyEuler(self, euler: "Euler") -> "Vector3":
        return self.applyQuaternion(_tmp_quaternion.setFromEuler(euler))

    def applyAxisAngle(self, axis: "Vector3", angle: float) -> "Vector3":
        return self.applyQuaternion(_tmp_quaternion.setFromAxisAngle(axis, angle))

    def applyMatrix3(self, m: "Matrix3") -> "Vector3":
        x, y, z = self.x, self.y, self.z
        e = m.elements

        self.x = e[0] * x + e[3] * y + e[6] * z
        self.y = e[1] * x + e[4] * y + e[7] * z
        self.z = e[2] * x + e[5] * y + e[8] * z
        return self

    def applyNormalMatrix(self, m: "Matrix3") -> "Vector3":
        return self.applyMatrix3(m).normalize()

    def applyMatrix4(self, m: "Matrix4") -> "Vector3":
        x, y, z = self.x, self.y, self.z
        e = m.elements

        w = 1 / (e[3] * x + e[7] * y + e[11] * z + e[15])
        self.x = (e[0] * x + e[4] * y + e[8] * z + e[12]) * w
        self.y = (e[1] * x + e[5] * y + e[9] * z + e[13]) * w
        self.z = (e[2] * x + e[6] * y + e[10] * z + e[14]) * w
        return self

    def applyQuaternion(self, q: "Quaternion") -> "Vector3":
        x = self.x
        y = self.y
        z = self.z
        qx = q.x
        qy = q.y
        qz = q.z
        qw = q.w

        # calculate quat * vector
        ix = qw * x + qy * z - qz * y
        iy = qw * y + qz * x - qx * z
        iz = qw * z + qx * y - qy * x
        iw = -qx * x - qy * y - qz * z

        # calculate result * inverse quat
        self.x = ix * qw + iw * -qx + iy * -qz - iz * -qy
        self.y = iy * qw + iw * -qy + iz * -qx - ix * -qz
        self.z = iz * qw + iw * -qz + ix * -qy - iy * -qx
        return self

    def project(self, camera) -> "Vector3":
        return self.applyMatrix4(camera.matrixWorldInverse).applyMatrix4(
            camera.projectionMatrix
        )

    def unproject(self, camera) -> "Vector3":
        return self.applyMatrix4(camera.projectionMatrixInverse).applyMatrix4(
            camera.matrixWorld
        )

    def transformDirection(self, m: "Matrix") -> "Vector3":
        # interpret self as directional vector
        # and apply affine transform in matrix4 m
        x = self.x
        y = self.y
        z = self.z
        e = m.elements

        self.x = e[0] * x + e[4] * y + e[8] * z
        self.y = e[1] * x + e[5] * y + e[9] * z
        self.z = e[2] * x + e[6] * y + e[10] * z
        return self.normalize()

    def divide(self, v: "Vector3") -> "Vector3":
        self.x /= v.x
        self.y /= v.y
        self.z /= v.z
        return self

    def divideScalar(self, s: float) -> "Vector3":
        self.x /= s
        self.y /= s
        self.z /= s
        return self

    def min(self, v: "Vector3") -> "Vector3":
        self.x = min(self.x, v.x)
        self.y = min(self.y, v.y)
        self.z = min(self.z, v.z)
        return self

    def max(self, v: "Vector3") -> "Vector3":
        self.x = max(self.x, v.x)
        self.y = max(self.y, v.y)
        self.z = max(self.z, v.z)
        return self

    def clamp(self, min: "Vector3", max: "Vector3") -> "Vector3":
        # assumes min < max, component-wise
        self.x = clamp(self.x, min.x, max.x)
        self.y = clamp(self.y, min.y, max.y)
        self.z = clamp(self.z, min.z, max.z)
        return self

    def clampScalar(self, min: float, max: float) -> "Vector3":
        self.x = clamp(self.x, min, max)
        self.y = clamp(self.y, min, max)
        self.z = clamp(self.z, min, max)

    def clampLength(self, min: float, max: float) -> "Vector3":
        l = self.length()
        return self.divideScalar(l or 1).multiplyScalar(clamp(l, min, max))

    def floor(self) -> "Vector3":
        self.x = floor(self.x)
        self.y = floor(self.y)
        self.z = floor(self.z)
        return self

    def ceil(self) -> "Vector3":
        self.x = ceil(self.x)
        self.y = ceil(self.y)
        self.z = ceil(self.z)
        return self

    def round(self) -> "Vector3":
        self.x = round(self.x)
        self.y = round(self.y)
        self.z = round(self.z)
        return self

    def roundToZero(self) -> "Vector3":
        self.x = ceil(self.x) if self.x < 0 else floor(self.x)
        self.y = ceil(self.y) if self.y < 0 else floor(self.y)
        self.z = ceil(self.z) if self.z < 0 else floor(self.z)
        return self

    def negate(self) -> "Vector3":
        self.x = -self.x
        self.y = -self.y
        self.z = -self.z
        return self

    def dot(self, v: "Vector3") -> float:
        return self.x * v.x + self.y * v.y + self.z * v.z

    def lengthSq(self) -> float:
        return self.x ** 2 + self.y ** 2 + self.z ** 2

    def length(self) -> float:
        return self.lengthSq() ** 0.5

    def manhattanLength(self) -> float:
        return abs(self.x) + abs(self.y) + abs(self.z)

    def normalize(self) -> "Vector3":
        return self.divideScalar(self.length() or 1)

    def setLength(self, l: float) -> "Vector3":
        return self.normalize().multiplyScalar(l)

    def lerp(self, v: "Vector3", a: float) -> "Vector3":
        self.x += (v.x - self.x) * a
        self.y += (v.y - self.y) * a
        self.z += (v.z - self.z) * a
        return self

    def lerpVectors(self, v1: "Vector3", v2: "Vector3", a: float) -> "Vector3":
        return self.subVectors(v2, v1).multiplyScalar(a).add(v1)

    def cross(self, v: "Vector3") -> "Vector3":
        return self.crossVectors(self, v)

    def crossVectors(self, a: "Vector3", b: "Vector3") -> "Vector3":
        ax = a.x
        ay = a.y
        az = a.z
        bx = b.x
        by = b.y
        bz = b.z

        self.x = ay * bz - az * by
        self.y = az * bx - ax * bz
        self.z = ax * by - ay * bx
        return self

    def projectOnVector(self, v: "Vector3") -> "Vector3":
        s = v.dot(self) / v.lengthSq()
        return self.copy(v).multiplyScalar(s)

    def projectOnPlane(self, n: "Vector3") -> "Vector3":
        _tmp_vector.copy(self).projectOnVector(n)
        return self.sub(_tmp_vector)

    def reflect(self, n: "Vector3") -> "Vector3":
        _tmp_vector.copy(n).multiplyScalar(2 * self.dot(n))
        return self.sub(_tmp_vector)

    def angleTo(self, v: "Vector3") -> float:
        denominator = (self.lengthSq() * v.lengthSq()) ** 0.5
        theta = self.dot(v) / denominator
        return acos(max(-1, min(1, theta)))

    def distanceTo(self, v: "Vector3") -> float:
        return self.distanceToSquared(v) ** 0.5

    def distanceToSquared(self, v: "Vector3") -> float:
        dx = self.x - v.x
        dy = self.y - v.y
        dz = self.z - v.z

        return dx * dx + dy * dy + dz * dz

    def manhattanDistanceTo(self, v: "Vector3") -> float:
        return abs(self.x - v.x) + abs(self.y - v.y) + abs(self.z - v.z)

    def setFromSpherical(self, s: "Spherical") -> "Vector3":
        return self.setFromSphericalCoords(s.radius, s.phi, s.theta)

    def setFromSphericalCoords(
        self, radius: float, phi: float, theta: float
    ) -> "Vector3":
        sin_phi_radius = sin(phi) * radius
        self.x = sin_phi_radius * sin(theta)
        self.y = cos(phi) * radius
        self.z = sin_phi_radius * cos(theta)
        return self

    def setFromCylindrical(self, c: "Cylindrical") -> "Vector3":
        return self.setFromCylindricalCoords(c.radius, c.theta, c.y)

    def setFromCylindricalCoords(
        self, radius: float, theta: float, y: float
    ) -> "Vector3":
        self.x = radius * sin(theta)
        self.y = y
        self.z = radius * cos(theta)
        return self

    def setFromMatrixPosition(self, m: "Matrix4") -> "Vector3":
        self.x = m.elements[12]
        self.y = m.elements[13]
        self.z = m.elements[14]
        return self

    def setFromMatrixScale(self, m: "Matrix4") -> "Vector3":
        sx = self.setFromMatrixColumn(m, 0).length()
        sy = self.setFromMatrixColumn(m, 1).length()
        sz = self.setFromMatrixColumn(m, 2).length()

        self.x = sx
        self.y = sy
        self.z = sz
        return self

    def setFromMatrixColumn(self, m: "Matrix4", i: int) -> "Vector3":
        return self.fromArray(m.elements, i * 4)

    def setFromMatrix3Column(self, m: "Matrix3", i: int) -> "Vector3":
        return self.fromArray(m.elements, i * 3)

    def equals(self, v: "Vector3") -> bool:
        return self.x == v.x and self.y == v.y and self.z == v.z

    def __eq__(self, other: "Vector3") -> bool:
        return isinstance(other, Vector3) and self.equals(other)

    def fromArray(self, array: list, offset: int = 0) -> "Vector3":
        self.x = array[offset]
        self.y = array[offset + 1]
        self.z = array[offset + 2]
        return self

    def toArray(self, array: list = None, offset: int = 0) -> list:
        if array is None:
            array = []

        padding = offset + 3 - len(array)
        if padding > 0:
            array.extend((None for _ in range(padding)))

        array[offset] = self.x
        array[offset + 1] = self.y
        array[offset + 2] = self.z
        return array

    def fromBufferAttribute(self, attribute, index: int) -> "Vector3":
        raise NotImplementedError()
        # self.x = attribute.getX(index)
        # self.y = attribute.getY(index)
        # self.z = attribute.getZ(index)
        # return self


_tmp_quaternion = Quaternion()
_tmp_vector = Vector3()
