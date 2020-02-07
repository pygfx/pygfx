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
        self, m: "Matrix4", order: RotationOrders = None
    ) -> "Euler":
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

        order = order or self.order

        if order == Euler.RotationOrders.XYZ:
            self.y = asin(clamp(m13, -1, 1))
            if abs(m13) < 0.9999999:
                self.x = atan2(-m23, m33)
                self.z = atan2(-m12, m11)
            else:
                self.x = atan2(m32, m22)
                self.z = 0

        elif order == Euler.RotationOrders.YXZ:
            self.x = asin(-clamp(m23, -1, 1))
            if abs(m23) < 0.9999999:
                self.y = atan2(m13, m33)
                self.z = atan2(m21, m22)
            else:
                self.y = atan2(-m31, m11)
                self.z = 0

        elif order == Euler.RotationOrders.ZXY:
            self.x = asin(clamp(m32, -1, 1))
            if abs(m32) < 0.9999999:
                self.y = atan2(-m31, m33)
                self.z = atan2(-m12, m22)
            else:
                self.y = 0
                self.z = atan2(m21, m11)

        elif order == Euler.RotationOrders.ZYX:
            self.y = asin(-clamp(m31, -1, 1))
            if abs(m31) < 0.9999999:
                self.x = atan2(m32, m33)
                self.z = atan2(m21, m11)
            else:
                self.x = 0
                self.z = atan2(-m12, m22)

        elif order == Euler.RotationOrders.YZX:
            self.z = asin(clamp(m21, -1, 1))
            if abs(m21) < 0.9999999:
                self.x = atan2(-m23, m22)
                self.y = atan2(-m31, m11)
            else:
                self.x = 0
                self.y = atan2(m13, m33)

        elif order == Euler.RotationOrders.XZY:
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
    def __init__(
        self,
        n11: float = 1,
        n12: float = 0,
        n13: float = 0,
        n14: float = 0,
        n21: float = 0,
        n22: float = 1,
        n23: float = 0,
        n24: float = 0,
        n31: float = 0,
        n32: float = 0,
        n33: float = 1,
        n34: float = 0,
        n41: float = 0,
        n42: float = 0,
        n43: float = 0,
        n44: float = 1,
    ) -> None:
        self.elements = [
            n11,
            n12,
            n13,
            n14,
            n21,
            n22,
            n23,
            n24,
            n31,
            n32,
            n33,
            n34,
            n41,
            n42,
            n43,
            n44,
        ]

    def __repr__(self) -> str:
        return f"Matrix4({self.elements[0]}, {self.elements[1]}, {self.elements[2]}, {self.elements[3]}, {self.elements[4]}, {self.elements[5]}, {self.elements[6]}, {self.elements[7]}, {self.elements[8]}, {self.elements[9]}, {self.elements[10]}, {self.elements[11]}, {self.elements[12]}, {self.elements[13]}, {self.elements[14]}, {self.elements[15]})"

    def set(
        self,
        n11: float,
        n12: float,
        n13: float,
        n14: float,
        n21: float,
        n22: float,
        n23: float,
        n24: float,
        n31: float,
        n32: float,
        n33: float,
        n34: float,
        n41: float,
        n42: float,
        n43: float,
        n44: float,
    ) -> "Matrix4":
        te = self.elements

        te[0] = n11
        te[4] = n12
        te[8] = n13
        te[12] = n14
        te[1] = n21
        te[5] = n22
        te[9] = n23
        te[13] = n24
        te[2] = n31
        te[6] = n32
        te[10] = n33
        te[14] = n34
        te[3] = n41
        te[7] = n42
        te[11] = n43
        te[15] = n44

        return self

    def identity(self) -> "Matrix4":
        self.set(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1)
        return self

    def clone(self) -> "Matrix4":
        return Matrix4().fromArray(self.elements)

    def copy(self, m: "Matrix4") -> "Matrix4":
        te = self.elements
        me = m.elements

        te[0] = me[0]
        te[1] = me[1]
        te[2] = me[2]
        te[3] = me[3]
        te[4] = me[4]
        te[5] = me[5]
        te[6] = me[6]
        te[7] = me[7]
        te[8] = me[8]
        te[9] = me[9]
        te[10] = me[10]
        te[11] = me[11]
        te[12] = me[12]
        te[13] = me[13]
        te[14] = me[14]
        te[15] = me[15]

        return self

    def copyPosition(self, m: "Matrix4") -> "Matrix4":
        te = self.elements
        me = m.elements

        te[12] = me[12]
        te[13] = me[13]
        te[14] = me[14]

        return self

    def extractBasis(
        self, xAxis: "Vector3", yAxis: "Vector3", zAxis: "Vector3"
    ) -> "Matrix4":
        xAxis.setFromMatrixColumn(self, 0)
        yAxis.setFromMatrixColumn(self, 1)
        zAxis.setFromMatrixColumn(self, 2)
        return self

    def makeBasis(
        self, xAxis: "Vector3", yAxis: "Vector3", zAxis: "Vector3"
    ) -> "Matrix4":
        self.set(
            xAxis.x,
            yAxis.x,
            zAxis.x,
            0,
            xAxis.y,
            yAxis.y,
            zAxis.y,
            0,
            xAxis.z,
            yAxis.z,
            zAxis.z,
            0,
            0,
            0,
            0,
            1,
        )

        return self

    def extractRotation(self, m: "Matrix4") -> "Matrix4":
        # this method does not support reflection matrices
        te = self.elements
        me = m.elements

        scaleX = 1 / _tmp_vector.setFromMatrixColumn(m, 0).length()
        scaleY = 1 / _tmp_vector.setFromMatrixColumn(m, 1).length()
        scaleZ = 1 / _tmp_vector.setFromMatrixColumn(m, 2).length()

        te[0] = me[0] * scaleX
        te[1] = me[1] * scaleX
        te[2] = me[2] * scaleX
        te[3] = 0
        te[4] = me[4] * scaleY
        te[5] = me[5] * scaleY
        te[6] = me[6] * scaleY
        te[7] = 0
        te[8] = me[8] * scaleZ
        te[9] = me[9] * scaleZ
        te[10] = me[10] * scaleZ
        te[11] = 0
        te[12] = 0
        te[13] = 0
        te[14] = 0
        te[15] = 1

        return self

    def makeRotationFromEuler(self, euler: "Euler") -> "Matrix4":
        te = self.elements
        x = euler.x
        y = euler.y
        z = euler.z
        a = cos(x)
        b = sin(x)
        c = cos(y)
        d = sin(y)
        e = cos(z)
        f = sin(z)

        if euler.order == Euler.RotationOrders.XYZ:
            ae = a * e
            af = a * f
            be = b * e
            bf = b * f

            te[0] = c * e
            te[4] = -c * f
            te[8] = d

            te[1] = af + be * d
            te[5] = ae - bf * d
            te[9] = -b * c

            te[2] = bf - ae * d
            te[6] = be + af * d
            te[10] = a * c
        elif euler.order == Euler.RotationOrders.YXZ:
            ce = c * e
            cf = c * f
            de = d * e
            df = d * f

            te[0] = ce + df * b
            te[4] = de * b - cf
            te[8] = a * d

            te[1] = a * f
            te[5] = a * e
            te[9] = -b

            te[2] = cf * b - de
            te[6] = df + ce * b
            te[10] = a * c
        elif euler.order == Euler.RotationOrders.ZXY:
            ce = c * e
            cf = c * f
            de = d * e
            df = d * f

            te[0] = ce - df * b
            te[4] = -a * f
            te[8] = de + cf * b

            te[1] = cf + de * b
            te[5] = a * e
            te[9] = df - ce * b

            te[2] = -a * d
            te[6] = b
            te[10] = a * c
        elif euler.order == Euler.RotationOrders.ZYX:
            ae = a * e
            af = a * f
            be = b * e
            bf = b * f

            te[0] = c * e
            te[4] = be * d - af
            te[8] = ae * d + bf

            te[1] = c * f
            te[5] = bf * d + ae
            te[9] = af * d - be

            te[2] = -d
            te[6] = b * c
            te[10] = a * c
        elif euler.order == Euler.RotationOrders.YZX:
            ac = a * c
            ad = a * d
            bc = b * c
            bd = b * d

            te[0] = c * e
            te[4] = bd - ac * f
            te[8] = bc * f + ad

            te[1] = f
            te[5] = a * e
            te[9] = -b * e

            te[2] = -d * e
            te[6] = ad * f + bc
            te[10] = ac - bd * f
        elif euler.order == Euler.RotationOrders.XZY:
            ac = a * c
            ad = a * d
            bc = b * c
            bd = b * d

            te[0] = c * e
            te[4] = -f
            te[8] = d * e

            te[1] = ac * f + bd
            te[5] = a * e
            te[9] = ad * f - bc

            te[2] = bc * f - ad
            te[6] = b * e
            te[10] = bd * f + ac

        # bottom row
        te[3] = 0
        te[7] = 0
        te[11] = 0

        # last column
        te[12] = 0
        te[13] = 0
        te[14] = 0
        te[15] = 1

        return self

    def makeRotationFromQuaternion(self, q: "Quaternion") -> "Matrix4":
        return self.compose(_zero, q, _one)

    def lookAt(self, eye: "Vector3", target: "Vector3", up: "Vector3") -> "Matrix4":
        te = self.elements

        _tmp_vector.subVectors(eye, target)

        if _tmp_vector.lengthSq() == 0:
            # eye and target are in the same position
            _tmp_vector.z = 1

        _tmp_vector.normalize()
        _tmp_vector2.crossVectors(up, _tmp_vector)

        if _tmp_vector2.lengthSq() == 0:
            # up and z are parallel
            if abs(up.z) == 1:
                _tmp_vector.x += 0.0001
            else:
                _tmp_vector.z += 0.0001

            _tmp_vector.normalize()
            _tmp_vector2.crossVectors(up, _tmp_vector)

        _tmp_vector2.normalize()
        _tmp_vector3.crossVectors(_tmp_vector, _tmp_vector2)

        te[0] = _tmp_vector2.x
        te[4] = _tmp_vector3.x
        te[8] = _tmp_vector.x
        te[1] = _tmp_vector2.y
        te[5] = _tmp_vector3.y
        te[9] = _tmp_vector.y
        te[2] = _tmp_vector2.z
        te[6] = _tmp_vector3.z
        te[10] = _tmp_vector.z

        return self

    def multiply(self, m: "Matrix4", n: "Matrix4") -> "Matrix4":
        return self.multiplyMatrices(self, m)

    def premultiply(self, m: "Matrix4") -> "Matrix4":
        return self.multiplyMatrices(m, self)

    def multiplyMatrices(self, a: "Matrix4", b: "Matrix4") -> "Matrix4":
        ae = a.elements
        be = b.elements
        te = self.elements

        a11 = ae[0]
        a12 = ae[4]
        a13 = ae[8]
        a14 = ae[12]
        a21 = ae[1]
        a22 = ae[5]
        a23 = ae[9]
        a24 = ae[13]
        a31 = ae[2]
        a32 = ae[6]
        a33 = ae[10]
        a34 = ae[14]
        a41 = ae[3]
        a42 = ae[7]
        a43 = ae[11]
        a44 = ae[15]

        b11 = be[0]
        b12 = be[4]
        b13 = be[8]
        b14 = be[12]
        b21 = be[1]
        b22 = be[5]
        b23 = be[9]
        b24 = be[13]
        b31 = be[2]
        b32 = be[6]
        b33 = be[10]
        b34 = be[14]
        b41 = be[3]
        b42 = be[7]
        b43 = be[11]
        b44 = be[15]

        te[0] = a11 * b11 + a12 * b21 + a13 * b31 + a14 * b41
        te[4] = a11 * b12 + a12 * b22 + a13 * b32 + a14 * b42
        te[8] = a11 * b13 + a12 * b23 + a13 * b33 + a14 * b43
        te[12] = a11 * b14 + a12 * b24 + a13 * b34 + a14 * b44

        te[1] = a21 * b11 + a22 * b21 + a23 * b31 + a24 * b41
        te[5] = a21 * b12 + a22 * b22 + a23 * b32 + a24 * b42
        te[9] = a21 * b13 + a22 * b23 + a23 * b33 + a24 * b43
        te[13] = a21 * b14 + a22 * b24 + a23 * b34 + a24 * b44

        te[2] = a31 * b11 + a32 * b21 + a33 * b31 + a34 * b41
        te[6] = a31 * b12 + a32 * b22 + a33 * b32 + a34 * b42
        te[10] = a31 * b13 + a32 * b23 + a33 * b33 + a34 * b43
        te[14] = a31 * b14 + a32 * b24 + a33 * b34 + a34 * b44

        te[3] = a41 * b11 + a42 * b21 + a43 * b31 + a44 * b41
        te[7] = a41 * b12 + a42 * b22 + a43 * b32 + a44 * b42
        te[11] = a41 * b13 + a42 * b23 + a43 * b33 + a44 * b43
        te[15] = a41 * b14 + a42 * b24 + a43 * b34 + a44 * b44

        return self

    def multiplyScalar(self, s: float) -> "Matrix4":
        te = self.elements

        te[0] *= s
        te[4] *= s
        te[8] *= s
        te[12] *= s
        te[1] *= s
        te[5] *= s
        te[9] *= s
        te[13] *= s
        te[2] *= s
        te[6] *= s
        te[10] *= s
        te[14] *= s
        te[3] *= s
        te[7] *= s
        te[11] *= s
        te[15] *= s

        return self

    def determinant(self) -> float:
        te = self.elements

        n11 = te[0]
        n12 = te[4]
        n13 = te[8]
        n14 = te[12]
        n21 = te[1]
        n22 = te[5]
        n23 = te[9]
        n24 = te[13]
        n31 = te[2]
        n32 = te[6]
        n33 = te[10]
        n34 = te[14]
        n41 = te[3]
        n42 = te[7]
        n43 = te[11]
        n44 = te[15]

        # TODO: make this more efficient
        # (based on http:#www.euclideanspace.com/maths/algebra/matrix/functions/inverse/fourD/index.htm)

        return (
            n41
            * (
                +n14 * n23 * n32
                - n13 * n24 * n32
                - n14 * n22 * n33
                + n12 * n24 * n33
                + n13 * n22 * n34
                - n12 * n23 * n34
            )
            + n42
            * (
                +n11 * n23 * n34
                - n11 * n24 * n33
                + n14 * n21 * n33
                - n13 * n21 * n34
                + n13 * n24 * n31
                - n14 * n23 * n31
            )
            + n43
            * (
                +n11 * n24 * n32
                - n11 * n22 * n34
                - n14 * n21 * n32
                + n12 * n21 * n34
                + n14 * n22 * n31
                - n12 * n24 * n31
            )
            + n44
            * (
                -n13 * n22 * n31
                - n11 * n23 * n32
                + n11 * n22 * n33
                + n13 * n21 * n32
                - n12 * n21 * n33
                + n12 * n23 * n31
            )
        )

    def transpose(self) -> "Matrix4":
        te = self.elements
        tmp = None

        tmp = te[1]
        te[1] = te[4]
        te[4] = tmp

        tmp = te[2]
        te[2] = te[8]
        te[8] = tmp

        tmp = te[6]
        te[6] = te[9]
        te[9] = tmp

        tmp = te[3]
        te[3] = te[12]
        te[12] = tmp

        tmp = te[7]
        te[7] = te[13]
        te[13] = tmp

        tmp = te[11]
        te[11] = te[14]
        te[14] = tmp

        return self

    def setPosition(self, v: "Vector3") -> "Matrix4":
        te = self.elements
        te[12] = x.x
        te[13] = x.y
        te[14] = x.z
        return self

    def setPositionXYZ(self, x: float, y: float, z: float) -> "Matrix4":
        te = self.elements
        te[12] = x
        te[13] = y
        te[14] = z
        return self

    def getInverse(self, m: "Matrix4") -> "Matrix4":
        te = self.elements
        me = m.elements

        n11 = me[0]
        n21 = me[1]
        n31 = me[2]
        n41 = me[3]
        n12 = me[4]
        n22 = me[5]
        n32 = me[6]
        n42 = me[7]
        n13 = me[8]
        n23 = me[9]
        n33 = me[10]
        n43 = me[11]
        n14 = me[12]
        n24 = me[13]
        n34 = me[14]
        n44 = me[15]

        t11 = (
            n23 * n34 * n42
            - n24 * n33 * n42
            + n24 * n32 * n43
            - n22 * n34 * n43
            - n23 * n32 * n44
            + n22 * n33 * n44,
        )
        t12 = (
            n14 * n33 * n42
            - n13 * n34 * n42
            - n14 * n32 * n43
            + n12 * n34 * n43
            + n13 * n32 * n44
            - n12 * n33 * n44,
        )
        t13 = (
            n13 * n24 * n42
            - n14 * n23 * n42
            + n14 * n22 * n43
            - n12 * n24 * n43
            - n13 * n22 * n44
            + n12 * n23 * n44,
        )
        t14 = (
            n14 * n23 * n32
            - n13 * n24 * n32
            - n14 * n22 * n33
            + n12 * n24 * n33
            + n13 * n22 * n34
            - n12 * n23 * n34
        )

        det = n11 * t11 + n21 * t12 + n31 * t13 + n41 * t14
        if det == 0:
            raise RuntimeError("matrix determinant is zero, cannot invert")

        detInv = 1 / det

        te[0] = t11 * detInv
        te[1] = (
            n24 * n33 * n41
            - n23 * n34 * n41
            - n24 * n31 * n43
            + n21 * n34 * n43
            + n23 * n31 * n44
            - n21 * n33 * n44
        ) * detInv
        te[2] = (
            n22 * n34 * n41
            - n24 * n32 * n41
            + n24 * n31 * n42
            - n21 * n34 * n42
            - n22 * n31 * n44
            + n21 * n32 * n44
        ) * detInv
        te[3] = (
            n23 * n32 * n41
            - n22 * n33 * n41
            - n23 * n31 * n42
            + n21 * n33 * n42
            + n22 * n31 * n43
            - n21 * n32 * n43
        ) * detInv

        te[4] = t12 * detInv
        te[5] = (
            n13 * n34 * n41
            - n14 * n33 * n41
            + n14 * n31 * n43
            - n11 * n34 * n43
            - n13 * n31 * n44
            + n11 * n33 * n44
        ) * detInv
        te[6] = (
            n14 * n32 * n41
            - n12 * n34 * n41
            - n14 * n31 * n42
            + n11 * n34 * n42
            + n12 * n31 * n44
            - n11 * n32 * n44
        ) * detInv
        te[7] = (
            n12 * n33 * n41
            - n13 * n32 * n41
            + n13 * n31 * n42
            - n11 * n33 * n42
            - n12 * n31 * n43
            + n11 * n32 * n43
        ) * detInv

        te[8] = t13 * detInv
        te[9] = (
            n14 * n23 * n41
            - n13 * n24 * n41
            - n14 * n21 * n43
            + n11 * n24 * n43
            + n13 * n21 * n44
            - n11 * n23 * n44
        ) * detInv
        te[10] = (
            n12 * n24 * n41
            - n14 * n22 * n41
            + n14 * n21 * n42
            - n11 * n24 * n42
            - n12 * n21 * n44
            + n11 * n22 * n44
        ) * detInv
        te[11] = (
            n13 * n22 * n41
            - n12 * n23 * n41
            - n13 * n21 * n42
            + n11 * n23 * n42
            + n12 * n21 * n43
            - n11 * n22 * n43
        ) * detInv

        te[12] = t14 * detInv
        te[13] = (
            n13 * n24 * n31
            - n14 * n23 * n31
            + n14 * n21 * n33
            - n11 * n24 * n33
            - n13 * n21 * n34
            + n11 * n23 * n34
        ) * detInv
        te[14] = (
            n14 * n22 * n31
            - n12 * n24 * n31
            - n14 * n21 * n32
            + n11 * n24 * n32
            + n12 * n21 * n34
            - n11 * n22 * n34
        ) * detInv
        te[15] = (
            n12 * n23 * n31
            - n13 * n22 * n31
            + n13 * n21 * n32
            - n11 * n23 * n32
            - n12 * n21 * n33
            + n11 * n22 * n33
        ) * detInv

        return self

    def scale(self, v: "Vector3") -> "Matrix4":
        te = self.elements
        x = v.x
        y = v.y
        z = v.z

        te[0] *= x
        te[4] *= y
        te[8] *= z
        te[1] *= x
        te[5] *= y
        te[9] *= z
        te[2] *= x
        te[6] *= y
        te[10] *= z
        te[3] *= x
        te[7] *= y
        te[11] *= z

        return self

    def getMaxScaleOnAxis(self) -> float:
        te = self.elements

        scaleXSq = te[0] * te[0] + te[1] * te[1] + te[2] * te[2]
        scaleYSq = te[4] * te[4] + te[5] * te[5] + te[6] * te[6]
        scaleZSq = te[8] * te[8] + te[9] * te[9] + te[10] * te[10]

        return max(scaleXSq, scaleYSq, scaleZSq) ** 0.5

    def makeTranslation(self, x: float, y: float, z: float) -> "Matrix4":
        self.set(1, 0, 0, x, 0, 1, 0, y, 0, 0, 1, z, 0, 0, 0, 1)
        return self

    def makeRotationX(self, theta: float) -> "Matrix4":
        c = cos(theta)
        s = sin(theta)
        self.set(1, 0, 0, 0, 0, c, -s, 0, 0, s, c, 0, 0, 0, 0, 1)
        return self

    def makeRotationY(self, theta: float) -> "Matrix4":
        c = cos(theta)
        s = sin(theta)
        self.set(c, 0, s, 0, 0, 1, 0, 0, -s, 0, c, 0, 0, 0, 0, 1)
        return self

    def makeRotationZ(self, theta: float) -> "Matrix4":
        c = cos(theta)
        s = sin(theta)
        self.set(c, -s, 0, 0, s, c, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1)
        return self

    def makeRotationAxis(self, axis: "Vector3", angle: float) -> "Matrix4":
        # Based on http:#www.gamedev.net/reference/articles/article1199.asp
        c = cos(angle)
        s = sin(angle)
        t = 1 - c
        x = axis.x
        y = axis.y
        z = axis.z
        tx = t * x
        ty = t * y
        self.set(
            tx * x + c,
            tx * y - s * z,
            tx * z + s * y,
            0,
            tx * y + s * z,
            ty * y + c,
            ty * z - s * x,
            0,
            tx * z - s * y,
            ty * z + s * x,
            t * z * z + c,
            0,
            0,
            0,
            0,
            1,
        )
        return self

    def makeScale(self, x: float, y: float, z: float) -> "Matrix4":
        self.set(x, 0, 0, 0, 0, y, 0, 0, 0, 0, z, 0, 0, 0, 0, 1)
        return self

    def makeShear(self, x: float, y: float, z: float) -> "Matrix4":
        self.set(1, y, z, 0, x, 1, z, 0, x, y, 1, 0, 0, 0, 0, 1)
        return self

    def compose(
        self, position: "Vector3", quaternion: "Quaternion", scale: "Vector3"
    ) -> "Matrix4":
        te = self.elements

        x = quaternion.x
        y = quaternion.y
        z = quaternion.z
        w = quaternion.w
        x2 = x + x
        y2 = y + y
        z2 = z + z
        xx = x * x2
        xy = x * y2
        xz = x * z2
        yy = y * y2
        yz = y * z2
        zz = z * z2
        wx = w * x2
        wy = w * y2
        wz = w * z2

        sx = scale.x
        sy = scale.y
        sz = scale.z

        te[0] = (1 - (yy + zz)) * sx
        te[1] = (xy + wz) * sx
        te[2] = (xz - wy) * sx
        te[3] = 0

        te[4] = (xy - wz) * sy
        te[5] = (1 - (xx + zz)) * sy
        te[6] = (yz + wx) * sy
        te[7] = 0

        te[8] = (xz + wy) * sz
        te[9] = (yz - wx) * sz
        te[10] = (1 - (xx + yy)) * sz
        te[11] = 0

        te[12] = position.x
        te[13] = position.y
        te[14] = position.z
        te[15] = 1

        return self

    def decompose(
        self, position: "Vector3", quaternion: "Quaternion", scale: "Vector3"
    ) -> "Matrix4":
        te = self.elements

        sx = _tmp_vector.set(te[0], te[1], te[2]).length()
        sy = _tmp_vector.set(te[4], te[5], te[6]).length()
        sz = _tmp_vector.set(te[8], te[9], te[10]).length()

        # if determine is negative, we need to invert one scale
        det = self.determinant()
        if det < 0:
            sx = -sx

        position.x = te[12]
        position.y = te[13]
        position.z = te[14]

        # scale the rotation part
        _tmp_matrix4.copy(self)

        invSX = 1 / sx
        invSY = 1 / sy
        invSZ = 1 / sz

        _tmp_matrix4.elements[0] *= invSX
        _tmp_matrix4.elements[1] *= invSX
        _tmp_matrix4.elements[2] *= invSX

        _tmp_matrix4.elements[4] *= invSY
        _tmp_matrix4.elements[5] *= invSY
        _tmp_matrix4.elements[6] *= invSY

        _tmp_matrix4.elements[8] *= invSZ
        _tmp_matrix4.elements[9] *= invSZ
        _tmp_matrix4.elements[10] *= invSZ

        quaternion.setFromRotationMatrix(_tmp_matrix4)

        scale.x = sx
        scale.y = sy
        scale.z = sz

        return self

    def makePerspective(
        self,
        left: float,
        right: float,
        top: float,
        bottom: float,
        near: float,
        far: float,
    ) -> "Matrix4":
        te = self.elements
        x = 2 * near / (right - left)
        y = 2 * near / (top - bottom)

        a = (right + left) / (right - left)
        b = (top + bottom) / (top - bottom)
        c = -(far + near) / (far - near)
        d = -2 * far * near / (far - near)

        te[0] = x
        te[4] = 0
        te[8] = a
        te[12] = 0
        te[1] = 0
        te[5] = y
        te[9] = b
        te[13] = 0
        te[2] = 0
        te[6] = 0
        te[10] = c
        te[14] = d
        te[3] = 0
        te[7] = 0
        te[11] = -1
        te[15] = 0

        return self

    def makeOrthographic(
        self,
        left: float,
        right: float,
        top: float,
        bottom: float,
        near: float,
        far: float,
    ) -> "Matrix4":
        te = self.elements
        w = 1.0 / (right - left)
        h = 1.0 / (top - bottom)
        p = 1.0 / (far - near)

        x = (right + left) * w
        y = (top + bottom) * h
        z = (far + near) * p

        te[0] = 2 * w
        te[4] = 0
        te[8] = 0
        te[12] = -x
        te[1] = 0
        te[5] = 2 * h
        te[9] = 0
        te[13] = -y
        te[2] = 0
        te[6] = 0
        te[10] = -2 * p
        te[14] = -z
        te[3] = 0
        te[7] = 0
        te[11] = 0
        te[15] = 1

        return self

    def equals(self, matrix: "Matrix4") -> bool:
        te = self.elements
        me = matrix.elements
        for i in range(16):
            if te[i] != me[i]:
                return False
        return True

    def __eq__(self, other: "Matrix4") -> bool:
        return isinstance(other, Matrix4) and self.equals(other)

    def fromArray(self, array: list, offset: int = 0) -> "Matrix4":
        for i in range(16):
            self.elements[i] = array[i + offset]
        return self

    def toArray(self, array: list = None, offset: int = 0) -> list:
        if array is None:
            array = []
        for i in range(16):
            array[i + offset] = self.elements[i]
        return array


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
        if index == 0:
            self.x = value
        elif index == 1:
            self.y = value
        elif index == 2:
            self.z = value
        return self

    def getComponent(self, index: int) -> float:
        if index == 0:
            return self.x
        elif index == 1:
            return self.y
        elif index == 2:
            return self.z

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
_tmp_vector2 = Vector3()
_tmp_vector3 = Vector3()
_tmp_quaternion = Quaternion()
_tmp_matrix4 = Matrix4()

_zero = Vector3(0, 0, 0)
_one = Vector3(1, 1, 1)
