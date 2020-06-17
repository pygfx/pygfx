from enum import Enum
from math import asin, atan2

from .utils import clamp
from .matrix4 import Matrix4
from .quaternion import Quaternion


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

    def set_from_rotation_matrix(
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

    def set_from_quaternion(
        self, q: "Quaternion", order: RotationOrders = None
    ) -> "Euler":
        _tmp_matrix4.make_rotation_from_quaternion(q)
        return self.set_from_rotation_matrix(_tmp_matrix4, order=order)

    def set_from_vector3(self, v: "Vector3", order: RotationOrders = None) -> "Euler":
        return self.set(v.x, v.y, v.z, order=order)

    def reorder(self, new_order: RotationOrders) -> "Euler":
        # warning: revolution info lost
        _tmp_quaternion.set_from_euler(self)
        return self.set_from_quaternion(_tmp_quaternion, order=new_order)

    def equals(self, euler: "Euler") -> bool:
        return (
            euler.x == self.x
            and euler.y == self.y
            and euler.z == self.z
            and euler.order == self.order
        )

    def __eq__(self, other: "Euler") -> bool:
        return isinstance(other, Euler) and self.equals(other)

    def from_array(self, array: list) -> "Euler":
        self.x = array[0]
        self.y = array[1]
        self.z = array[2]
        if len(array) >= 4:
            self.order = array[3]
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
        array[offset + 3] = self.order
        return array

    def to_vector3(self, output: "Vector3" = None) -> "Vector3":
        from .vector3 import Vector3

        if output is not None:
            return output.set(self.x, self.y, self.z)
        else:
            return Vector3(self.x, self.y, self.z)


_tmp_matrix4 = Matrix4()
_tmp_quaternion = Quaternion()
