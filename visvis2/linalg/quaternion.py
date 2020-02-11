from math import cos, sin, acos, atan2

from .utils import clamp, MACHINE_EPSILON


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

    def set_from_euler(self, euler: "Euler") -> "Quaternion":
        from .euler import Euler

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

    def set_from_axis_angle(self, axis: "Vector3", angle: float) -> "Quaternion":
        # assumes axis is normalized
        half_angle = angle / 2
        s = sin(half_angle)

        self.x = axis.x * s
        self.y = axis.y * s
        self.z = axis.z * s
        self.w = cos(half_angle)

        return self

    def set_from_rotation_matrix(self, m: "Matrix4") -> "Quaternion":
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

    def set_from_unit_vectors(self, v_from: "Vector3", v_to: "Vector3") -> "Quaternion":
        # assumes direction vectors v_from and v_to are normalized
        eps = 0.000001
        r = v_from.dot(v_to) + 1

        if r < eps:
            r = 0
            if abs(v_from.x) > abs(v_from.z):
                self.x = -v_from.y
                self.y = v_from.x
                self.z = 0
                self.w = r
            else:
                self.x = 0
                self.y = -v_from.z
                self.z = v_from.y
                self.w = r
        else:
            self.x = v_from.y * v_to.z - v_from.z * v_to.y
            self.y = v_from.z * v_to.x - v_from.x * v_to.z
            self.z = v_from.x * v_to.y - v_from.y * v_to.x
            self.w = r

        return self.normalize()

    def angle_to(self, q: "Quaternion") -> float:
        return 2 * acos(abs(clamp(self.dot(q), -1, 1)))

    def rotate_towards(self, q: "Quaternion", step: float) -> "Quaternion":
        angle = self.angle_to(q)
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

    def length_sq(self) -> float:
        return self.x * self.x + self.y * self.y + self.z * self.z + self.w * self.w

    def length(self) -> float:
        return self.length_sq() ** 0.5

    def normalize(self) -> "Quaternion":
        length = self.length()

        if length == 0:
            self.x = 0
            self.y = 0
            self.z = 0
            self.w = 1
        else:
            self.x /= length
            self.y /= length
            self.z /= length
            self.w /= length

        return self

    def multiply(self, q: "Quaternion") -> "Quaternion":
        return self.multiply_quaternions(self, q)

    def premultiply(self, q: "Quaternion") -> "Quaternion":
        return self.multiply_quaternions(q, self)

    def multiply_quaternions(self, a: "Quaternion", b: "Quaternion") -> "Quaternion":
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
        cos_half_theta = w * qb.w + x * qb.x + y * qb.y + z * qb.z
        if cos_half_theta < 0:
            self.w = -qb.w
            self.x = -qb.x
            self.y = -qb.y
            self.z = -qb.z
            cos_half_theta = -cos_half_theta
        else:
            self.copy(qb)

        if cos_half_theta >= 1.0:
            self.w = w
            self.x = x
            self.y = y
            self.z = z
            return self

        sqr_sin_half_theta = 1.0 - cos_half_theta * cos_half_theta
        if sqr_sin_half_theta <= MACHINE_EPSILON:
            s = 1 - t
            self.w = s * w + t * self.w
            self.x = s * x + t * self.x
            self.y = s * y + t * self.y
            self.z = s * z + t * self.z
            self.normalize()
            return self

        sin_half_theta = sqr_sin_half_theta ** 0.5
        half_theta = atan2(sin_half_theta, cos_half_theta)
        ratio_a = sin((1 - t) * half_theta) / sin_half_theta
        ratio_b = sin(t * half_theta) / sin_half_theta

        self.w = w * ratio_a + self.w * ratio_b
        self.x = x * ratio_a + self.x * ratio_b
        self.y = y * ratio_a + self.y * ratio_b
        self.z = z * ratio_a + self.z * ratio_b
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

    def from_array(self, array: list, offset: int = 0) -> "Quaternion":
        self.x = array[offset]
        self.y = array[offset + 1]
        self.z = array[offset + 2]
        self.w = array[offset + 3]
        return self

    def to_array(self, array: list = None, offset: int = 0) -> list:
        if array is None:
            array = []

        padding = offset + 4 - len(array)
        if padding > 0:
            array.extend((None for _ in range(padding)))

        array[offset] = self.x
        array[offset + 1] = self.y
        array[offset + 2] = self.z
        array[offset + 3] = self.w
        return array
