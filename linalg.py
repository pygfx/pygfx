# Ported straight from three.js, original license at time of writing copied below:
# ---
# The MIT License
#
# Copyright © 2010-2020 three.js authors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
from enum import Enum
from math import acos, asin, atan2, ceil, floor, cos, sin


MACHINE_EPSILON = (
    7.0 / 3 - 4.0 / 3 - 1
)  # the difference between 1 and the smallest floating point number greater than 1


def clamp(x: float, left: float, right: float) -> float:
    return max(left, min(right, x))


class Matrix3:
    def __init__(self) -> None:
        self.elements = [
            1,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            1,
        ]


class Matrix4:
    def __init__(self) -> None:
        self.elements = [
            1,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            1,
        ]


class Spherical:
    def __init__(
        self, radius: float = 1.0, phi: float = 0.0, theta: float = 0.0
    ) -> None:
        self.radius = radius
        self.phi = phi
        self.theta = theta


class Cylindrical:
    def __init__(self, radius: float = 1.0, theta: float = 0.0, y: float = 0.0) -> None:
        self.radius = radius
        self.theta = theta
        self.y = y


class Euler:
    class RotationOrders(Enum):
        XYZ = "XYZ"
        YZX = "YZX"
        ZXY = "ZXY"
        XZY = "XZY"
        YXZ = "YXZ"
        ZYX = "ZYX"

    DefaultOrder = RotationOrders.XYZ

    def __init__(
        self, x: float = 0, y: float = 0, z: float = 0, order: RotationOrders = None
    ) -> None:
        self.x = x
        self.y = y
        self.z = z
        self.order = order if order is not None else self.DefaultOrder

    def __repr__(self) -> str:
        return f"Euler({self.x}, {self.y}, {self.z}, {self.order})"

    def set(
        self, x: float, y: float, z: float, order: RotationOrders = None
    ) -> "Euler":
        self.x = x
        self.y = y
        self.z = z
        self.order = order if order is not None else self.order
        return self

    def clone(self) -> "Euler":
        return Euler(self.x, self.y, self.z, self.order)

    def copy(self, euler: "Euler") -> "Euler":
        self.x = euler.x
        self.y = euler.y
        self.z = euler.z
        self.order = euler.order
        return self

    def setFromRotationMatrix(
        self, m: Matrix4, order: RotationOrders = None
    ) -> "Euler":
        # assumes the upper 3x3 of m is a pure rotation matrix (i.e, unscaled)
        te = m.elements
        m11 = te[0], m12 = te[4], m13 = te[8]
        m21 = te[1], m22 = te[5], m23 = te[9]
        m31 = te[2], m32 = te[6], m33 = te[10]

        order = order or self.order

        if order == RotationOrders.XYZ:
            self.y = asin(clamp(m13, -1, 1))
            if abs(m13) < 0.9999999:
                self.x = atan2(-m23, m33)
                self.z = atan2(-m12, m11)
            else:
                self.x = atan2(m32, m22)
                self.z = 0

        elif order == RotationOrders.YXZ:
            self.x = asin(-clamp(m23, -1, 1))
            if abs(m23) < 0.9999999:
                self.y = atan2(m13, m33)
                self.z = atan2(m21, m22)
            else:
                self.y = atan2(-m31, m11)
                self.z = 0

        elif order == RotationOrders.ZXY:
            self.x = asin(clamp(m32, -1, 1))
            if abs(m32) < 0.9999999:
                self.y = atan2(-m31, m33)
                self.z = atan2(-m12, m22)
            else:
                self.y = 0
                self.z = atan2(m21, m11)

        elif order == RotationOrders.ZYX:
            self.y = asin(-clamp(m31, -1, 1))
            if abs(m31) < 0.9999999:
                self.x = atan2(m32, m33)
                self.z = atan2(m21, m11)
            else:
                self.x = 0
                self.z = atan2(-m12, m22)

        elif order == RotationOrders.YZX:
            self.z = asin(clamp(m21, -1, 1))
            if abs(m21) < 0.9999999:
                self.x = atan2(-m23, m22)
                self.y = atan2(-m31, m11)
            else:
                self.x = 0
                self.y = atan2(m13, m33)

        elif order == RotationOrders.XZY:
            self.z = asin(-clamp(m12, -1, 1))
            if abs(m12) < 0.9999999:
                self.x = atan2(m32, m22)
                self.y = atan2(m13, m11)
            else:
                self.x = atan2(-m23, m33)
                self.y = 0

        else:
            raise ValueError(f"{order} not supported")

        self.order = order

        return self

    def setFromQuaternion(
        self, q: "Quaternion", order: RotationOrders = None
    ) -> "Euler":
        _tmp_matrix4.makeRotationFromQuaternion(q)
        return self.setFromRotationMatrix(_tmp_matrix4, order=order)

    def setFromVector3(self, v: "Vector3", order: RotationOrders = None) -> "Euler":
        return self.set(v.x, v.y, v.z, order=order)

    def reorder(self, newOrder: RotationOrders) -> "Euler":
        # warning: revolution info lost
        _tmp_quaternion.setFromEuler(self)
        return self.setFromQuaternion(_tmp_quaternion, order=newOrder)

    def equals(self, euler: "Euler") -> bool:
        return (
            euler.x == self.x
            and euler.y == self.y
            and euler.z == self.z
            and euler.order == self.order
        )

    def __eq__(self, other: "Euler") -> bool:
        return isinstance(other, Euler) and self.equals(other)

    def fromArray(self, array: list) -> "Euler":
        self.x = array[0]
        self.y = array[1]
        self.z = array[2]
        if len(array) >= 4:
            self.order = array[3]
        return self

    def toArray(self, array: list = None, offset: int = 0) -> list:
        if array is None:
            array = []
        array[offset] = self.x
        array[offset + 1] = self.y
        array[offset + 2] = self.z
        array[offset + 3] = self.order
        return array

    def toVector3(self, output: "Vector3" = None) -> "Vector3":
        if output is not None:
            return output.set(self.x, self.y, self.z)
        else:
            return Vector3(self.x, self.y, self.z)


class Quaternion:
    def __init__(self, x: float = 0, y: float = 0, z: float = 0, w: float = 1) -> None:
        self.x = x
        self.y = y
        self.z = z
        self.w = w

    def set(self, x: float, y: float, z: float, w: float) -> "Quaternion":
        self.x = x
        self.y = y
        self.z = z
        self.w = w
        return self

    def clone(self) -> "Quaternion":
        return Quaternion(self.x, self.y, self.z, self.w)

    def __repr__(self) -> str:
        return f"Quaternion({self.x}, {self.y}, {self.z}, {self.w})"

    def copy(self, quaternion: "Quaternion") -> "Quaternion":
        self.x = quaternion.x
        self.y = quaternion.y
        self.z = quaternion.z
        self.w = quaternion.w
        return self

    def setFromEuler(self, euler: "Euler") -> "Quaternion":
        x = euler.x
        y = euler.y
        z = euler.z
        order = euler.order

        c1 = cos(x / 2)
        c2 = cos(y / 2)
        c3 = cos(z / 2)
        s1 = sin(x / 2)
        s2 = sin(y / 2)
        s3 = sin(z / 2)

        if order == Euler.RotationOrders.XYZ:
            self.x = s1 * c2 * c3 + c1 * s2 * s3
            self.y = c1 * s2 * c3 - s1 * c2 * s3
            self.z = c1 * c2 * s3 + s1 * s2 * c3
            self.w = c1 * c2 * c3 - s1 * s2 * s3
        elif order == Euler.RotationOrders.YXZ:
            self.x = s1 * c2 * c3 + c1 * s2 * s3
            self.y = c1 * s2 * c3 - s1 * c2 * s3
            self.z = c1 * c2 * s3 - s1 * s2 * c3
            self.w = c1 * c2 * c3 + s1 * s2 * s3
        elif order == Euler.RotationOrders.ZXY:
            self.x = s1 * c2 * c3 - c1 * s2 * s3
            self.y = c1 * s2 * c3 + s1 * c2 * s3
            self.z = c1 * c2 * s3 + s1 * s2 * c3
            self.w = c1 * c2 * c3 - s1 * s2 * s3
        elif order == Euler.RotationOrders.ZYX:
            self.x = s1 * c2 * c3 - c1 * s2 * s3
            self.y = c1 * s2 * c3 + s1 * c2 * s3
            self.z = c1 * c2 * s3 - s1 * s2 * c3
            self.w = c1 * c2 * c3 + s1 * s2 * s3
        elif order == Euler.RotationOrders.YZX:
            self.x = s1 * c2 * c3 + c1 * s2 * s3
            self.y = c1 * s2 * c3 + s1 * c2 * s3
            self.z = c1 * c2 * s3 - s1 * s2 * c3
            self.w = c1 * c2 * c3 - s1 * s2 * s3
        elif order == Euler.RotationOrders.XZY:
            self.x = s1 * c2 * c3 - c1 * s2 * s3
            self.y = c1 * s2 * c3 - s1 * c2 * s3
            self.z = c1 * c2 * s3 + s1 * s2 * c3
            self.w = c1 * c2 * c3 + s1 * s2 * s3

        return self

    def setFromAxisAngle(self, axis: "Vector3", angle: float) -> "Quaternion":
        # assumes axis is normalized
        halfAngle = angle / 2
        s = sin(halfAngle)

        self.x = axis.x * s
        self.y = axis.y * s
        self.z = axis.z * s
        self.w = cos(halfAngle)

        return self

    def setFromRotationMatrix(self, m: Matrix4) -> "Quaternion":
        # assumes the upper 3x3 of m is a pure rotation matrix (i.e, unscaled)
        te = m.elements
        m11 = te[0]
        m12 = te[4]
        m13 = te[8]
        m21 = te[1]
        m22 = te[5]
        m23 = te[9]
        m31 = te[2]
        m32 = te[6]
        m33 = te[10]
        trace = m11 + m22 + m33

        if trace > 0:
            s = 0.5 / ((trace + 1.0) ** 0.5)
            self.w = 0.25 / s
            self.x = (m32 - m23) * s
            self.y = (m13 - m31) * s
            self.z = (m21 - m12) * s

        elif m11 > m22 and m11 > m33:
            s = 2.0 * ((1.0 + m11 - m22 - m33) ** 0.5)
            self.w = (m32 - m23) / s
            self.x = 0.25 * s
            self.y = (m12 + m21) / s
            self.z = (m13 + m31) / s

        elif m22 > m33:
            s = 2.0 * ((1.0 + m22 - m11 - m33) ** 0.5)
            self.w = (m13 - m31) / s
            self.x = (m12 + m21) / s
            self.y = 0.25 * s
            self.z = (m23 + m32) / s

        else:
            s = 2.0 * ((1.0 + m33 - m11 - m22) ** 0.5)
            self.w = (m21 - m12) / s
            self.x = (m13 + m31) / s
            self.y = (m23 + m32) / s
            self.z = 0.25 * s

        return self

    def setFromUnitVectors(self, vFrom: "Vector3", vTo: "Vector3") -> "Quaternion":
        # assumes direction vectors vFrom and vTo are normalized
        EPS = 0.000001
        r = vFrom.dot(vTo) + 1

        if r < EPS:
            r = 0
            if abs(vFrom.x) > abs(vFrom.z):
                self.x = -vFrom.y
                self.y = vFrom.x
                self.z = 0
                self.w = r
            else:
                self.x = 0
                self.y = -vFrom.z
                self.z = vFrom.y
                self.w = r
        else:
            self.x = vFrom.y * vTo.z - vFrom.z * vTo.y
            self.y = vFrom.z * vTo.x - vFrom.x * vTo.z
            self.z = vFrom.x * vTo.y - vFrom.y * vTo.x
            self.w = r

        return self.normalize()

    def angleTo(self, q: "Quaternion") -> float:
        return 2 * acos(abs(clamp(self.dot(q), -1, 1)))

    def rotateTowards(self, q: "Quaternion", step: float) -> "Quaternion":
        angle = self.angleTo(q)
        if angle == 0:
            return self
        t = min(1, step / angle)
        self.slerp(q, t)
        return self

    def inverse(self) -> "Quaternion":
        # quaternion is assumed to have unit length
        return self.conjugate()

    def conjugate(self) -> "Quaternion":
        self.x *= -1
        self.y *= -1
        self.z *= -1
        return self

    def dot(self, q: "Quaternion") -> float:
        return self.x * q.x + self.y * q.y + self.z * q.z + self.w * q.w

    def lengthSq(self) -> float:
        return self.x * self.x + self.y * self.y + self.z * self.z + self.w * self.w

    def length(self) -> float:
        return self.lengthSq() ** 0.5

    def normalize(self) -> "Quaternion":
        l = self.length()

        if l == 0:
            self.x = 0
            self.y = 0
            self.z = 0
            self.w = 1
        else:
            self.x /= l
            self.y /= l
            self.z /= l
            self.w /= l

        return self

    def multiply(self, q: "Quaternion") -> "Quaternion":
        return self.multiplyQuaternions(self, q)

    def premultiply(self, q: "Quaternion") -> "Quaternion":
        return self.multiplyQuaternions(q, self)

    def multiplyQuaternions(self, a: "Quaternion", b: "Quaternion") -> "Quaternion":
        qax = a.x
        qay = a.y
        qaz = a.z
        qaw = a.w
        qbx = b.x
        qby = b.y
        qbz = b.z
        qbw = b.w

        self.x = qax * qbw + qaw * qbx + qay * qbz - qaz * qby
        self.y = qay * qbw + qaw * qby + qaz * qbx - qax * qbz
        self.z = qaz * qbw + qaw * qbz + qax * qby - qay * qbx
        self.w = qaw * qbw - qax * qbx - qay * qby - qaz * qbz

        return self

    @staticmethod
    def slerp(
        qStart: "Quaternion", qEnd: "Quaternion", qTarget: "Quaternion", t: float
    ) -> "Quaternion":
        return qTarget.copy(qStart).slerp(qEnd, t)

    @staticmethod
    def slerpFlat(
        dst: list,
        dstOffset: int,
        src0: list,
        srcOffset0: int,
        src1: list,
        srcOffset1: int,
        t: float,
    ) -> None:
        x0 = src0[srcOffset0 + 0]
        y0 = src0[srcOffset0 + 1]
        z0 = src0[srcOffset0 + 2]
        w0 = src0[srcOffset0 + 3]
        x1 = src1[srcOffset1 + 0]
        y1 = src1[srcOffset1 + 1]
        z1 = src1[srcOffset1 + 2]
        w1 = src1[srcOffset1 + 3]

        if w0 != w1 or x0 != x1 or y0 != y1 or z0 != z1:
            s = 1 - t
            cos = x0 * x1 + y0 * y1 + z0 * z1 + w0 * w1
            dir = 1 if cos >= 0 else -1
            sqrSin = 1 - cos * cos

            # avoid numeric problems
            if sqrSin > MACHINE_EPSILON:
                sin = sqrt(sqrSin)
                len = atan2(sin, cos * dir)
                s = sin(s * len) / sin
                t = sin(t * len) / sin

            tDir = t * dir
            x0 = x0 * s + x1 * tDir
            y0 = y0 * s + y1 * tDir
            z0 = z0 * s + z1 * tDir
            w0 = w0 * s + w1 * tDir

            if s == 1 - t:
                f = 1 / ((x0 * x0 + y0 * y0 + z0 * z0 + w0 * w0) ** 0.5)

                x0 *= f
                y0 *= f
                z0 *= f
                w0 *= f

        dst[dstOffset] = x0
        dst[dstOffset + 1] = y0
        dst[dstOffset + 2] = z0
        dst[dstOffset + 3] = w0

    def slerp(self, qb: "Quaternion", t: float) -> "Quaternion":
        if t == 0:
            return self
        elif t == 1:
            return self.copy(qb)

        x = self.x
        y = self.y
        z = self.z
        w = self.w
        cosHalfTheta = w * qb.w + x * qb.x + y * qb.y + z * qb.z
        if cosHalfTheta < 0:
            self.w = -qb.w
            self.x = -qb.x
            self.y = -qb.y
            self.z = -qb.z
            cosHalfTheta = -cosHalfTheta
        else:
            self.copy(qb)

        if cosHalfTheta >= 1.0:
            self.w = w
            self.x = x
            self.y = y
            self.z = z
            return self

        sqrSinHalfTheta = 1.0 - cosHalfTheta * cosHalfTheta
        if sqrSinHalfTheta <= MACHINE_EPSILON:
            s = 1 - t
            self.w = s * w + t * self.w
            self.x = s * x + t * self.x
            self.y = s * y + t * self.y
            self.z = s * z + t * self.z
            self.normalize()
            return self

        sinHalfTheta = sqrSinHalfTheta ** 0.5
        halfTheta = atan2(sinHalfTheta, cosHalfTheta)
        ratioA = sin((1 - t) * halfTheta) / sinHalfTheta
        ratioB = sin(t * halfTheta) / sinHalfTheta

        self.w = w * ratioA + self.w * ratioB
        self.x = x * ratioA + self.x * ratioB
        self.y = y * ratioA + self.y * ratioB
        self.z = z * ratioA + self.z * ratioB
        return self

    def equals(self, quaternion: "Quaternion") -> bool:
        return (
            quaternion.x == self.x
            and quaternion.y == self.y
            and quaternion.z == self.z
            and quaternion.w == self.w
        )

    def __eq__(self, other: "Quaternion") -> bool:
        return isinstance(other, Quaternion) and self.equals(other)

    def fromArray(self, array: list, offset: int = 0) -> "Quaternion":
        self.x = array[offset]
        self.y = array[offset + 1]
        self.z = array[offset + 2]
        self.w = array[offset + 3]
        return self

    def toArray(self, array: list = None, offset: int = 0) -> list:
        if array is None:
            array = []
        array[offset] = self.x
        array[offset + 1] = self.y
        array[offset + 2] = self.z
        array[offset + 3] = self.w
        return array


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
        [self.x, self.y, self.z][index] = value
        return self

    def getComponent(self, index: int) -> float:
        return [self.x, self.y, self.z][index]

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

    def applyMatrix3(self, m: Matrix3) -> "Vector3":
        x, y, z = self.x, self.y, self.z
        e = m.elements

        self.x = e[0] * x + e[3] * y + e[6] * z
        self.y = e[1] * x + e[4] * y + e[7] * z
        self.z = e[2] * x + e[5] * y + e[8] * z
        return self

    def applyNormalMatrix(self, m: Matrix3) -> "Vector3":
        return self.applyMatrix3(m).normalize()

    def applyMatrix4(self, m: Matrix4) -> "Vector3":
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

    def transformDirection(self, m: Matrix4) -> "Vector3":
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

    def cross(self, v: "Vector3", w: "Vector3") -> "Vector3":
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
        denominator = (self.lenthSq() * v.lengthSq()) ** 0.5
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

    def setFromSpherical(self, s: Spherical) -> "Vector3":
        return self.setFromSphericalCoords(s.radius, s.phi, s.theta)

    def setFromSphericalCoords(
        self, radius: float, phi: float, theta: float
    ) -> "Vector3":
        sin_phi_radius = sin(phi) * radius
        self.x = sin_phi_radius * sin(theta)
        self.y = cos(phi) * radius
        self.z = sin_phi_radius * cos(theta)
        return self

    def setFromCylindrical(self, c: Cylindrical) -> "Vector3":
        return self.setFromCylindricalCoords(c.radius, c.theta, c.y)

    def setFromCylindricalCoords(
        self, radius: float, theta: float, y: float
    ) -> "Vector3":
        self.x = radius * sin(theta)
        self.y = y
        self.z = radius * cos(theta)
        return self

    def setFromMatrixPosition(self, m: Matrix4) -> "Vector3":
        self.x = m.elements[12]
        self.y = m.elements[13]
        self.z = m.elements[14]
        return self

    def setFromMatrixScale(self, m: Matrix4) -> "Vector3":
        sx = self.setFromMatrixColumn(m, 0).length()
        sy = self.setFromMatrixColumn(m, 1).length()
        sz = self.setFromMatrixColumn(m, 2).length()

        self.x = sx
        self.y = sy
        self.z = sz
        return self

    def setFromMatrixColumn(self, m: Matrix4, i: int) -> "Vector3":
        return self.fromArray(m.elements, i * 4)

    def setFromMatrix3Column(self, m: Matrix3, i: int) -> "Vector3":
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

        array[offset] = self.x
        array[offset + 1] = self.y
        array[offset + 2] = self.z
        return array

    def fromBufferAttribute(self, attribute, index: int) -> "Vector3":
        self.x = attribute.getX(index)
        self.y = attribute.getY(index)
        self.z = attribute.getZ(index)
        return self


_tmp_vector = Vector3()
_tmp_quaternion = Quaternion()
_tmp_matrix4 = Matrix4()
