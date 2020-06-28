from math import cos, sin

from .vector3 import Vector3


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
        return Matrix4().from_array(self.elements)

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

    def copy_position(self, m: "Matrix4") -> "Matrix4":
        te = self.elements
        me = m.elements

        te[12] = me[12]
        te[13] = me[13]
        te[14] = me[14]

        return self

    def extract_basis(
        self, x_axis: "Vector3", y_axis: "Vector3", z_axis: "Vector3"
    ) -> "Matrix4":
        x_axis.set_from_matrix_column(self, 0)
        y_axis.set_from_matrix_column(self, 1)
        z_axis.set_from_matrix_column(self, 2)
        return self

    def make_basis(
        self, x_axis: "Vector3", y_axis: "Vector3", z_axis: "Vector3"
    ) -> "Matrix4":
        self.set(
            x_axis.x,
            y_axis.x,
            z_axis.x,
            0,
            x_axis.y,
            y_axis.y,
            z_axis.y,
            0,
            x_axis.z,
            y_axis.z,
            z_axis.z,
            0,
            0,
            0,
            0,
            1,
        )

        return self

    def extract_rotation(self, m: "Matrix4") -> "Matrix4":
        # this method does not support reflection matrices
        te = self.elements
        me = m.elements

        scale_x = 1 / _tmp_vector.set_from_matrix_column(m, 0).length()
        scale_y = 1 / _tmp_vector.set_from_matrix_column(m, 1).length()
        scale_z = 1 / _tmp_vector.set_from_matrix_column(m, 2).length()

        te[0] = me[0] * scale_x
        te[1] = me[1] * scale_x
        te[2] = me[2] * scale_x
        te[3] = 0
        te[4] = me[4] * scale_y
        te[5] = me[5] * scale_y
        te[6] = me[6] * scale_y
        te[7] = 0
        te[8] = me[8] * scale_z
        te[9] = me[9] * scale_z
        te[10] = me[10] * scale_z
        te[11] = 0
        te[12] = 0
        te[13] = 0
        te[14] = 0
        te[15] = 1

        return self

    def make_rotation_from_euler(self, euler: "Euler") -> "Matrix4":
        from .euler import Euler

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

    def make_rotation_from_quaternion(self, q: "Quaternion") -> "Matrix4":
        return self.compose(_zero, q, _one)

    def look_at(self, eye: "Vector3", target: "Vector3", up: "Vector3") -> "Matrix4":
        te = self.elements

        _tmp_vector.sub_vectors(eye, target)

        if _tmp_vector.length_sq() == 0:
            # eye and target are in the same position
            _tmp_vector.z = 1

        _tmp_vector.normalize()
        _tmp_vector2.cross_vectors(up, _tmp_vector)

        if _tmp_vector2.length_sq() == 0:
            # up and z are parallel
            if abs(up.z) == 1:
                _tmp_vector.x += 0.0001
            else:
                _tmp_vector.z += 0.0001

            _tmp_vector.normalize()
            _tmp_vector2.cross_vectors(up, _tmp_vector)

        _tmp_vector2.normalize()
        _tmp_vector3.cross_vectors(_tmp_vector, _tmp_vector2)

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

    def multiply(self, m: "Matrix4") -> "Matrix4":
        return self.multiply_matrices(self, m)

    def premultiply(self, m: "Matrix4") -> "Matrix4":
        return self.multiply_matrices(m, self)

    def multiply_matrices(self, a: "Matrix4", b: "Matrix4") -> "Matrix4":
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

    def multiply_scalar(self, s: float) -> "Matrix4":
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

    def set_position(self, v: "Vector3") -> "Matrix4":
        te = self.elements
        te[12] = v.x
        te[13] = v.y
        te[14] = v.z
        return self

    def set_position_xyz(self, x: float, y: float, z: float) -> "Matrix4":
        te = self.elements
        te[12] = x
        te[13] = y
        te[14] = z
        return self

    def get_inverse(self, m: "Matrix4", throw_on_degenerate: bool = True) -> "Matrix4":
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
            + n22 * n33 * n44
        )
        t12 = (
            n14 * n33 * n42
            - n13 * n34 * n42
            - n14 * n32 * n43
            + n12 * n34 * n43
            + n13 * n32 * n44
            - n12 * n33 * n44
        )
        t13 = (
            n13 * n24 * n42
            - n14 * n23 * n42
            + n14 * n22 * n43
            - n12 * n24 * n43
            - n13 * n22 * n44
            + n12 * n23 * n44
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
            if throw_on_degenerate:
                raise ValueError("matrix determinant is zero, cannot invert")
            else:
                return self.identity()

        det_inv = 1 / det

        te[0] = t11 * det_inv
        te[1] = (
            n24 * n33 * n41
            - n23 * n34 * n41
            - n24 * n31 * n43
            + n21 * n34 * n43
            + n23 * n31 * n44
            - n21 * n33 * n44
        ) * det_inv
        te[2] = (
            n22 * n34 * n41
            - n24 * n32 * n41
            + n24 * n31 * n42
            - n21 * n34 * n42
            - n22 * n31 * n44
            + n21 * n32 * n44
        ) * det_inv
        te[3] = (
            n23 * n32 * n41
            - n22 * n33 * n41
            - n23 * n31 * n42
            + n21 * n33 * n42
            + n22 * n31 * n43
            - n21 * n32 * n43
        ) * det_inv

        te[4] = t12 * det_inv
        te[5] = (
            n13 * n34 * n41
            - n14 * n33 * n41
            + n14 * n31 * n43
            - n11 * n34 * n43
            - n13 * n31 * n44
            + n11 * n33 * n44
        ) * det_inv
        te[6] = (
            n14 * n32 * n41
            - n12 * n34 * n41
            - n14 * n31 * n42
            + n11 * n34 * n42
            + n12 * n31 * n44
            - n11 * n32 * n44
        ) * det_inv
        te[7] = (
            n12 * n33 * n41
            - n13 * n32 * n41
            + n13 * n31 * n42
            - n11 * n33 * n42
            - n12 * n31 * n43
            + n11 * n32 * n43
        ) * det_inv

        te[8] = t13 * det_inv
        te[9] = (
            n14 * n23 * n41
            - n13 * n24 * n41
            - n14 * n21 * n43
            + n11 * n24 * n43
            + n13 * n21 * n44
            - n11 * n23 * n44
        ) * det_inv
        te[10] = (
            n12 * n24 * n41
            - n14 * n22 * n41
            + n14 * n21 * n42
            - n11 * n24 * n42
            - n12 * n21 * n44
            + n11 * n22 * n44
        ) * det_inv
        te[11] = (
            n13 * n22 * n41
            - n12 * n23 * n41
            - n13 * n21 * n42
            + n11 * n23 * n42
            + n12 * n21 * n43
            - n11 * n22 * n43
        ) * det_inv

        te[12] = t14 * det_inv
        te[13] = (
            n13 * n24 * n31
            - n14 * n23 * n31
            + n14 * n21 * n33
            - n11 * n24 * n33
            - n13 * n21 * n34
            + n11 * n23 * n34
        ) * det_inv
        te[14] = (
            n14 * n22 * n31
            - n12 * n24 * n31
            - n14 * n21 * n32
            + n11 * n24 * n32
            + n12 * n21 * n34
            - n11 * n22 * n34
        ) * det_inv
        te[15] = (
            n12 * n23 * n31
            - n13 * n22 * n31
            + n13 * n21 * n32
            - n11 * n23 * n32
            - n12 * n21 * n33
            + n11 * n22 * n33
        ) * det_inv

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

    def get_max_scale_on_axis(self) -> float:
        te = self.elements

        scale_x_sq = te[0] * te[0] + te[1] * te[1] + te[2] * te[2]
        scale_y_sq = te[4] * te[4] + te[5] * te[5] + te[6] * te[6]
        scale_z_sq = te[8] * te[8] + te[9] * te[9] + te[10] * te[10]

        return max(scale_x_sq, scale_y_sq, scale_z_sq) ** 0.5

    def make_translation(self, x: float, y: float, z: float) -> "Matrix4":
        self.set(1, 0, 0, x, 0, 1, 0, y, 0, 0, 1, z, 0, 0, 0, 1)
        return self

    def make_rotation_x(self, theta: float) -> "Matrix4":
        c = cos(theta)
        s = sin(theta)
        self.set(1, 0, 0, 0, 0, c, -s, 0, 0, s, c, 0, 0, 0, 0, 1)
        return self

    def make_rotation_y(self, theta: float) -> "Matrix4":
        c = cos(theta)
        s = sin(theta)
        self.set(c, 0, s, 0, 0, 1, 0, 0, -s, 0, c, 0, 0, 0, 0, 1)
        return self

    def make_rotation_z(self, theta: float) -> "Matrix4":
        c = cos(theta)
        s = sin(theta)
        self.set(c, -s, 0, 0, s, c, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1)
        return self

    def make_rotation_axis(self, axis: "Vector3", angle: float) -> "Matrix4":
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

    def make_scale(self, x: float, y: float, z: float) -> "Matrix4":
        self.set(x, 0, 0, 0, 0, y, 0, 0, 0, 0, z, 0, 0, 0, 0, 1)
        return self

    def make_shear(self, x: float, y: float, z: float) -> "Matrix4":
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

        inv_sx = 1 / sx
        inv_sy = 1 / sy
        inv_sz = 1 / sz

        _tmp_matrix4.elements[0] *= inv_sx
        _tmp_matrix4.elements[1] *= inv_sx
        _tmp_matrix4.elements[2] *= inv_sx

        _tmp_matrix4.elements[4] *= inv_sy
        _tmp_matrix4.elements[5] *= inv_sy
        _tmp_matrix4.elements[6] *= inv_sy

        _tmp_matrix4.elements[8] *= inv_sz
        _tmp_matrix4.elements[9] *= inv_sz
        _tmp_matrix4.elements[10] *= inv_sz

        quaternion.set_from_rotation_matrix(_tmp_matrix4)

        scale.x = sx
        scale.y = sy
        scale.z = sz

        return self

    def make_perspective(
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

    def make_orthographic(
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

    def from_array(self, array: list, offset: int = 0) -> "Matrix4":
        for i in range(16):
            self.elements[i] = array[i + offset]
        return self

    def to_array(self, array: list = None, offset: int = 0) -> list:
        if array is None:
            array = []
        padding = offset + 16 - len(array)
        if padding > 0:
            array.extend((None for _ in range(padding)))
        for i in range(16):
            array[i + offset] = self.elements[i]
        return array


_zero = Vector3(0, 0, 0)
_one = Vector3(1, 1, 1)
_tmp_vector = Vector3()
_tmp_vector2 = Vector3()
_tmp_vector3 = Vector3()
_tmp_matrix4 = Matrix4()
