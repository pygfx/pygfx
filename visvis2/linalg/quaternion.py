
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

    def setFromRotationMatrix(self, m: "Matrix4") -> "Quaternion":
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
