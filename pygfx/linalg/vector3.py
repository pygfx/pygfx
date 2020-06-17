from math import acos, sin, cos, floor, ceil

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

    def set_scalar(self, s: float) -> "Vector3":
        self.x = s
        self.y = s
        self.z = s
        return self

    def set_x(self, x: float) -> "Vector3":
        self.x = x
        return self

    def set_y(self, y: float) -> "Vector3":
        self.y = y
        return self

    def set_z(self, z: float) -> "Vector3":
        self.z = z
        return self

    def set_component(self, index: int, value: float) -> "Vector3":
        if index == 0:
            self.x = value
        elif index == 1:
            self.y = value
        elif index == 2:
            self.z = value
        else:
            raise IndexError()
        return self

    def get_component(self, index: int) -> float:
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

    def add_scalar(self, s: float) -> "Vector3":
        self.x += s
        self.y += s
        self.z += s
        return self

    def add_vectors(self, a: "Vector3", b: "Vector3") -> "Vector3":
        self.x = a.x + b.x
        self.y = a.y + b.y
        self.z = a.z + b.z
        return self

    def add_scaled_vector(self, v: "Vector3", s: float) -> "Vector3":
        self.x += v.x * s
        self.y += v.y * s
        self.z += v.z * s
        return self

    def sub(self, v: "Vector3") -> "Vector3":
        self.x -= v.x
        self.y -= v.y
        self.z -= v.z
        return self

    def sub_scalar(self, s: float) -> "Vector3":
        self.x -= s
        self.y -= s
        self.z -= s
        return self

    def sub_vectors(self, a: "Vector3", b: "Vector3") -> "Vector3":
        self.x = a.x - b.x
        self.y = a.y - b.y
        self.z = a.z - b.z
        return self

    def multiply(self, v: "Vector3") -> "Vector3":
        self.x *= v.x
        self.y *= v.y
        self.z *= v.z
        return self

    def multiply_scalar(self, s: float) -> "Vector3":
        self.x *= s
        self.y *= s
        self.z *= s
        return self

    def multiply_vectors(self, a: "Vector3", b: "Vector3") -> "Vector3":
        self.x = a.x * b.x
        self.y = a.y * b.y
        self.z = a.z * b.z
        return self

    def apply_euler(self, euler: "Euler") -> "Vector3":
        return self.apply_quaternion(_tmp_quaternion.set_from_euler(euler))

    def apply_axis_angle(self, axis: "Vector3", angle: float) -> "Vector3":
        return self.apply_quaternion(_tmp_quaternion.set_from_axis_angle(axis, angle))

    def apply_matrix3(self, m: "Matrix3") -> "Vector3":
        x, y, z = self.x, self.y, self.z
        e = m.elements

        self.x = e[0] * x + e[3] * y + e[6] * z
        self.y = e[1] * x + e[4] * y + e[7] * z
        self.z = e[2] * x + e[5] * y + e[8] * z
        return self

    def apply_normal_matrix(self, m: "Matrix3") -> "Vector3":
        return self.apply_matrix3(m).normalize()

    def apply_matrix4(self, m: "Matrix4") -> "Vector3":
        x, y, z = self.x, self.y, self.z
        e = m.elements

        denom = e[3] * x + e[7] * y + e[11] * z + e[15]
        if denom == 0:
            w = float("Inf")
        elif denom == -0:
            w = float("-Inf")
        else:
            w = 1 / denom
        self.x = (e[0] * x + e[4] * y + e[8] * z + e[12]) * w
        self.y = (e[1] * x + e[5] * y + e[9] * z + e[13]) * w
        self.z = (e[2] * x + e[6] * y + e[10] * z + e[14]) * w
        return self

    def apply_quaternion(self, q: "Quaternion") -> "Vector3":
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
        return self.apply_matrix4(camera.matrix_world_inverse).apply_matrix4(
            camera.projection_matrix
        )

    def unproject(self, camera) -> "Vector3":
        return self.apply_matrix4(camera.projection_matrix_inverse).apply_matrix4(
            camera.matrix_world
        )

    def transform_direction(self, m: "Matrix") -> "Vector3":
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

    def divide_scalar(self, s: float) -> "Vector3":
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

    def clamp_scalar(self, min: float, max: float) -> "Vector3":
        self.x = clamp(self.x, min, max)
        self.y = clamp(self.y, min, max)
        self.z = clamp(self.z, min, max)

    def clamp_length(self, min: float, max: float) -> "Vector3":
        length = self.length()
        return self.divide_scalar(length or 1).multiply_scalar(clamp(length, min, max))

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

    def round_to_zero(self) -> "Vector3":
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

    def length_sq(self) -> float:
        return self.x ** 2 + self.y ** 2 + self.z ** 2

    def length(self) -> float:
        return self.length_sq() ** 0.5

    def manhattan_length(self) -> float:
        return abs(self.x) + abs(self.y) + abs(self.z)

    def normalize(self) -> "Vector3":
        return self.divide_scalar(self.length() or 1)

    def set_length(self, length: float) -> "Vector3":
        return self.normalize().multiply_scalar(length)

    def lerp(self, v: "Vector3", a: float) -> "Vector3":
        self.x += (v.x - self.x) * a
        self.y += (v.y - self.y) * a
        self.z += (v.z - self.z) * a
        return self

    def lerp_vectors(self, v1: "Vector3", v2: "Vector3", a: float) -> "Vector3":
        return self.sub_vectors(v2, v1).multiply_scalar(a).add(v1)

    def cross(self, v: "Vector3") -> "Vector3":
        return self.cross_vectors(self, v)

    def cross_vectors(self, a: "Vector3", b: "Vector3") -> "Vector3":
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

    def project_on_vector(self, v: "Vector3") -> "Vector3":
        s = v.dot(self) / v.length_sq()
        return self.copy(v).multiply_scalar(s)

    def project_on_plane(self, n: "Vector3") -> "Vector3":
        _tmp_vector.copy(self).project_on_vector(n)
        return self.sub(_tmp_vector)

    def reflect(self, n: "Vector3") -> "Vector3":
        _tmp_vector.copy(n).multiply_scalar(2 * self.dot(n))
        return self.sub(_tmp_vector)

    def angle_to(self, v: "Vector3") -> float:
        denominator = (self.length_sq() * v.length_sq()) ** 0.5
        theta = self.dot(v) / denominator
        return acos(max(-1, min(1, theta)))

    def distance_to(self, v: "Vector3") -> float:
        return self.distance_to_squared(v) ** 0.5

    def distance_to_squared(self, v: "Vector3") -> float:
        dx = self.x - v.x
        dy = self.y - v.y
        dz = self.z - v.z

        return dx * dx + dy * dy + dz * dz

    def manhattan_distance_to(self, v: "Vector3") -> float:
        return abs(self.x - v.x) + abs(self.y - v.y) + abs(self.z - v.z)

    def set_from_spherical(self, s: "Spherical") -> "Vector3":
        return self.set_from_spherical_coords(s.radius, s.phi, s.theta)

    def set_from_spherical_coords(
        self, radius: float, phi: float, theta: float
    ) -> "Vector3":
        sin_phi_radius = sin(phi) * radius
        self.x = sin_phi_radius * sin(theta)
        self.y = cos(phi) * radius
        self.z = sin_phi_radius * cos(theta)
        return self

    def set_from_cylindrical(self, c: "Cylindrical") -> "Vector3":
        return self.set_from_cylindrical_coords(c.radius, c.theta, c.y)

    def set_from_cylindrical_coords(
        self, radius: float, theta: float, y: float
    ) -> "Vector3":
        self.x = radius * sin(theta)
        self.y = y
        self.z = radius * cos(theta)
        return self

    def set_from_matrix_position(self, m: "Matrix4") -> "Vector3":
        self.x = m.elements[12]
        self.y = m.elements[13]
        self.z = m.elements[14]
        return self

    def set_from_matrix_scale(self, m: "Matrix4") -> "Vector3":
        sx = self.set_from_matrix_column(m, 0).length()
        sy = self.set_from_matrix_column(m, 1).length()
        sz = self.set_from_matrix_column(m, 2).length()

        self.x = sx
        self.y = sy
        self.z = sz
        return self

    def set_from_matrix_column(self, m: "Matrix4", i: int) -> "Vector3":
        return self.from_array(m.elements, i * 4)

    def set_from_matrix3_column(self, m: "Matrix3", i: int) -> "Vector3":
        return self.from_array(m.elements, i * 3)

    def equals(self, v: "Vector3") -> bool:
        return self.x == v.x and self.y == v.y and self.z == v.z

    def __eq__(self, other: "Vector3") -> bool:
        return isinstance(other, Vector3) and self.equals(other)

    def from_array(self, array: list, offset: int = 0) -> "Vector3":
        self.x = array[offset]
        self.y = array[offset + 1]
        self.z = array[offset + 2]
        return self

    def to_array(self, array: list = None, offset: int = 0) -> list:
        if array is None:
            array = []

        padding = offset + 3 - len(array)
        if padding > 0:
            array.extend((None for _ in range(padding)))

        array[offset] = self.x
        array[offset + 1] = self.y
        array[offset + 2] = self.z
        return array

    def from_buffer_attribute(self, attribute, index: int) -> "Vector3":
        raise NotImplementedError()
        # self.x = attribute.getX(index)
        # self.y = attribute.getY(index)
        # self.z = attribute.getZ(index)
        # return self


_tmp_quaternion = Quaternion()
_tmp_vector = Vector3()
