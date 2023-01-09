from math import pi, sqrt, radians

import pytest

from pygfx.linalg import (
    Vector3,
    Euler,
    Matrix4,
    Quaternion,
)

from .utils import matrix_equals, euler_equals  # this import works: WHY?


eps = 0.0001


# INSTANCING
def test_instancing():
    a = Matrix4()
    assert a.determinant() == 1

    b = Matrix4().set(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)
    assert b.elements[0] == 0
    assert b.elements[1] == 4
    assert b.elements[2] == 8
    assert b.elements[3] == 12
    assert b.elements[4] == 1
    assert b.elements[5] == 5
    assert b.elements[6] == 9
    assert b.elements[7] == 13
    assert b.elements[8] == 2
    assert b.elements[9] == 6
    assert b.elements[10] == 10
    assert b.elements[11] == 14
    assert b.elements[12] == 3
    assert b.elements[13] == 7
    assert b.elements[14] == 11
    assert b.elements[15] == 15

    assert not matrix_equals(a, b)


# PUBLIC STUFF
def test_set():
    b = Matrix4()
    assert b.determinant() == 1

    b.set(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)
    assert b.elements[0] == 0
    assert b.elements[1] == 4
    assert b.elements[2] == 8
    assert b.elements[3] == 12
    assert b.elements[4] == 1
    assert b.elements[5] == 5
    assert b.elements[6] == 9
    assert b.elements[7] == 13
    assert b.elements[8] == 2
    assert b.elements[9] == 6
    assert b.elements[10] == 10
    assert b.elements[11] == 14
    assert b.elements[12] == 3
    assert b.elements[13] == 7
    assert b.elements[14] == 11
    assert b.elements[15] == 15


def test_identity():
    b = Matrix4().set(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)
    assert b.elements[0] == 0
    assert b.elements[1] == 4
    assert b.elements[2] == 8
    assert b.elements[3] == 12
    assert b.elements[4] == 1
    assert b.elements[5] == 5
    assert b.elements[6] == 9
    assert b.elements[7] == 13
    assert b.elements[8] == 2
    assert b.elements[9] == 6
    assert b.elements[10] == 10
    assert b.elements[11] == 14
    assert b.elements[12] == 3
    assert b.elements[13] == 7
    assert b.elements[14] == 11
    assert b.elements[15] == 15

    a = Matrix4()
    assert not matrix_equals(a, b)

    b.identity()
    assert matrix_equals(a, b)


def test_clone():
    a = Matrix4().set(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)
    b = a.clone()

    assert matrix_equals(a, b)

    # ensure that it is a True copy
    a.elements[0] = 2
    assert not matrix_equals(a, b)


def test_copy():
    a = Matrix4().set(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)
    b = Matrix4().copy(a)

    assert matrix_equals(a, b)

    # ensure that it is a True copy
    a.elements[0] = 2
    assert not matrix_equals(a, b)


def test_copy_position():
    a = Matrix4().set(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)
    b = Matrix4().set(1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 16)

    assert not matrix_equals(a, b), "a and b initially not equal"

    b.copy_position(a)
    assert matrix_equals(a, b), "a and b equal after copy_position()"


def test_make_basisextract_basis():
    identity_basis = [Vector3(1, 0, 0), Vector3(0, 1, 0), Vector3(0, 0, 1)]
    a = Matrix4().make_basis(identity_basis[0], identity_basis[1], identity_basis[2])
    identity = Matrix4()
    assert matrix_equals(a, identity)

    test_bases = [[Vector3(0, 1, 0), Vector3(-1, 0, 0), Vector3(0, 0, 1)]]
    for i in range(len(test_bases)):
        test_basis = test_bases[i]
        b = Matrix4().make_basis(test_basis[0], test_basis[1], test_basis[2])
        out_basis = [Vector3(), Vector3(), Vector3()]
        b.extract_basis(out_basis[0], out_basis[1], out_basis[2])
        # check what goes in, is what comes out.
        for j in range(len(out_basis)):
            assert out_basis[j].equals(test_basis[j])

        # get the basis out the hard war
        for j in range(len(identity_basis)):
            out_basis[j].copy(identity_basis[j])
            out_basis[j].apply_matrix4(b)

        # did the multiply method of basis extraction work?
        for j in range(len(out_basis)):
            assert out_basis[j].equals(test_basis[j])


def test_make_rotation_from_eulerextract_rotation():
    test_values = [
        Euler(0, 0, 0, Euler.RotationOrders.XYZ),
        Euler(1, 0, 0, Euler.RotationOrders.XYZ),
        Euler(0, 1, 0, Euler.RotationOrders.ZYX),
        Euler(0, 0, 0.5, Euler.RotationOrders.YZX),
        Euler(0, 0, -0.5, Euler.RotationOrders.YZX),
    ]

    for i in range(len(test_values)):
        v = test_values[i]

        m = Matrix4().make_rotation_from_euler(v)

        v2 = Euler().set_from_rotation_matrix(m, v.order)
        m2 = Matrix4().make_rotation_from_euler(v2)

        assert matrix_equals(m, m2, eps), (
            "make_rotation_from_euler #"
            + i
            + ": original and Euler-derived matrices are equal"
        )
        assert euler_equals(v, v2, eps), (
            "make_rotation_from_euler #"
            + i
            + ": original and matrix-derived Eulers are equal"
        )

        m3 = Matrix4().extract_rotation(m2)
        v3 = Euler().set_from_rotation_matrix(m3, v.order)

        assert matrix_equals(m, m3, eps), (
            "extract_rotation #" + i + ": original and extracted matrices are equal"
        )
        assert euler_equals(v, v3, eps), (
            "extract_rotation #" + i + ": original and extracted Eulers are equal"
        )


def test_look_at():
    a = Matrix4()
    expected = Matrix4().identity()
    eye = Vector3(0, 0, 0)
    target = Vector3(0, 1, -1)
    up = Vector3(0, 1, 0)

    a.look_at(eye, target, up)
    rotation = Euler().set_from_rotation_matrix(a)
    assert rotation.x * (180 / pi) == 45, "Check the rotation"

    # eye and target are in the same position
    eye.copy(target)
    a.look_at(eye, target, up)
    assert matrix_equals(a, expected), "Check the result for eye == target"

    # up and z are parallel
    eye.set(0, 1, 0)
    target.set(0, 0, 0)
    a.look_at(eye, target, up)
    expected.set(1, 0, 0, 0, 0, 0.0001, 1, 0, 0, -1, 0.0001, 0, 0, 0, 0, 1)
    assert matrix_equals(a, expected), "Check the result for when up and z are parallel"


def test_multiply():
    lhs = Matrix4().set(2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53)
    rhs = Matrix4().set(
        59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131
    )

    lhs.multiply(rhs)

    assert lhs.elements[0] == 1585
    assert lhs.elements[1] == 5318
    assert lhs.elements[2] == 10514
    assert lhs.elements[3] == 15894
    assert lhs.elements[4] == 1655
    assert lhs.elements[5] == 5562
    assert lhs.elements[6] == 11006
    assert lhs.elements[7] == 16634
    assert lhs.elements[8] == 1787
    assert lhs.elements[9] == 5980
    assert lhs.elements[10] == 11840
    assert lhs.elements[11] == 17888
    assert lhs.elements[12] == 1861
    assert lhs.elements[13] == 6246
    assert lhs.elements[14] == 12378
    assert lhs.elements[15] == 18710


def test_premultiply():
    lhs = Matrix4().set(2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53)
    rhs = Matrix4().set(
        59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131
    )

    rhs.premultiply(lhs)

    assert rhs.elements[0] == 1585
    assert rhs.elements[1] == 5318
    assert rhs.elements[2] == 10514
    assert rhs.elements[3] == 15894
    assert rhs.elements[4] == 1655
    assert rhs.elements[5] == 5562
    assert rhs.elements[6] == 11006
    assert rhs.elements[7] == 16634
    assert rhs.elements[8] == 1787
    assert rhs.elements[9] == 5980
    assert rhs.elements[10] == 11840
    assert rhs.elements[11] == 17888
    assert rhs.elements[12] == 1861
    assert rhs.elements[13] == 6246
    assert rhs.elements[14] == 12378
    assert rhs.elements[15] == 18710


def test_multiply_matrices():
    # Reference:
    #
    # #!/usr/bin/env python
    # from __future__ import print_function
    # import numpy as np
    # print(
    #     np.dot(
    #         np.reshape([2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53], (4, 4)),
    #         np.reshape([59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131], (4, 4))
    #     )
    # )
    #
    # [[ 1585  1655  1787  1861]
    #  [ 5318  5562  5980  6246]
    #  [10514 11006 11840 12378]
    #  [15894 16634 17888 18710]]
    lhs = Matrix4().set(2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53)
    rhs = Matrix4().set(
        59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131
    )
    ans = Matrix4()

    ans.multiply_matrices(lhs, rhs)

    assert ans.elements[0] == 1585
    assert ans.elements[1] == 5318
    assert ans.elements[2] == 10514
    assert ans.elements[3] == 15894
    assert ans.elements[4] == 1655
    assert ans.elements[5] == 5562
    assert ans.elements[6] == 11006
    assert ans.elements[7] == 16634
    assert ans.elements[8] == 1787
    assert ans.elements[9] == 5980
    assert ans.elements[10] == 11840
    assert ans.elements[11] == 17888
    assert ans.elements[12] == 1861
    assert ans.elements[13] == 6246
    assert ans.elements[14] == 12378
    assert ans.elements[15] == 18710


def test_multiply_scalar():
    b = Matrix4().set(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)
    assert b.elements[0] == 0
    assert b.elements[1] == 4
    assert b.elements[2] == 8
    assert b.elements[3] == 12
    assert b.elements[4] == 1
    assert b.elements[5] == 5
    assert b.elements[6] == 9
    assert b.elements[7] == 13
    assert b.elements[8] == 2
    assert b.elements[9] == 6
    assert b.elements[10] == 10
    assert b.elements[11] == 14
    assert b.elements[12] == 3
    assert b.elements[13] == 7
    assert b.elements[14] == 11
    assert b.elements[15] == 15

    b.multiply_scalar(2)
    assert b.elements[0] == 0 * 2
    assert b.elements[1] == 4 * 2
    assert b.elements[2] == 8 * 2
    assert b.elements[3] == 12 * 2
    assert b.elements[4] == 1 * 2
    assert b.elements[5] == 5 * 2
    assert b.elements[6] == 9 * 2
    assert b.elements[7] == 13 * 2
    assert b.elements[8] == 2 * 2
    assert b.elements[9] == 6 * 2
    assert b.elements[10] == 10 * 2
    assert b.elements[11] == 14 * 2
    assert b.elements[12] == 3 * 2
    assert b.elements[13] == 7 * 2
    assert b.elements[14] == 11 * 2
    assert b.elements[15] == 15 * 2


def test_determinant():
    a = Matrix4()
    assert a.determinant() == 1

    a.elements[0] = 2
    assert a.determinant() == 2

    a.elements[0] = 0
    assert a.determinant() == 0

    # calculated via http://www.euclideanspace.com/maths/algebra/matrix/functions/determinant/fourD/index.htm
    a.set(2, 3, 4, 5, -1, -21, -3, -4, 6, 7, 8, 10, -8, -9, -10, -12)
    assert a.determinant() == 76


def test_transpose():
    a = Matrix4()
    b = a.clone().transpose()
    assert matrix_equals(a, b)

    b = Matrix4().set(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)
    c = b.clone().transpose()
    assert not matrix_equals(b, c)
    c.transpose()
    assert matrix_equals(b, c)


def test_set_position():
    a = Matrix4().set(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)
    b = Vector3(-1, -2, -3)
    c = Matrix4().set(0, 1, 2, -1, 4, 5, 6, -2, 8, 9, 10, -3, 12, 13, 14, 15)

    a.set_position(b)
    assert matrix_equals(a, c)


def test_get_inverse():
    identity = Matrix4()

    a = Matrix4()
    b = Matrix4().set(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    c = Matrix4().set(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

    assert not matrix_equals(a, b)
    b.get_inverse(a, False)
    assert matrix_equals(b, Matrix4())

    with pytest.raises(ValueError):
        b.get_inverse(c, True)

    test_matrices = [
        Matrix4().make_rotation_x(0.3),
        Matrix4().make_rotation_x(-0.3),
        Matrix4().make_rotation_y(0.3),
        Matrix4().make_rotation_y(-0.3),
        Matrix4().make_rotation_z(0.3),
        Matrix4().make_rotation_z(-0.3),
        Matrix4().make_scale(1, 2, 3),
        Matrix4().make_scale(1 / 8, 1 / 2, 1 / 3),
        Matrix4().make_perspective(-1, 1, 1, -1, 1, 1000),
        Matrix4().make_perspective(-16, 16, 9, -9, 0.1, 10000),
        Matrix4().make_translation(1, 2, 3),
    ]

    for i in range(len(test_matrices)):
        m = test_matrices[i]

        m_inverse = Matrix4().get_inverse(m)
        m_self_inverse = m.clone()
        m_self_inverse.get_inverse(m_self_inverse)

        # self-inverse should the same as inverse
        assert matrix_equals(m_self_inverse, m_inverse)

        # the determinant of the inverse should be the reciprocal
        assert abs(m.determinant() * m_inverse.determinant() - 1) < 0.0001

        m_product = Matrix4().multiply_matrices(m, m_inverse)

        # the determinant of the identity matrix is 1
        assert abs(m_product.determinant() - 1) < 0.0001
        assert matrix_equals(m_product, identity)


def test_scale():
    a = Matrix4().set(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)
    b = Vector3(2, 3, 4)
    c = Matrix4().set(2, 6, 12, 4, 10, 18, 28, 8, 18, 30, 44, 12, 26, 42, 60, 16)

    a.scale(b)
    assert matrix_equals(a, c)


def test_get_max_scale_on_axis():
    a = Matrix4().set(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)
    expected = sqrt(3 * 3 + 7 * 7 + 11 * 11)

    assert abs(a.get_max_scale_on_axis() - expected) <= eps, "Check result"


def test_make_translation():
    a = Matrix4()
    b = Vector3(2, 3, 4)
    c = Matrix4().set(1, 0, 0, 2, 0, 1, 0, 3, 0, 0, 1, 4, 0, 0, 0, 1)

    a.make_translation(b.x, b.y, b.z)
    assert matrix_equals(a, c)


def test_make_rotation_x():
    a = Matrix4()
    b = sqrt(3) / 2
    c = Matrix4().set(1, 0, 0, 0, 0, b, -0.5, 0, 0, 0.5, b, 0, 0, 0, 0, 1)

    a.make_rotation_x(pi / 6)
    assert matrix_equals(a, c)


def test_make_rotation_y():

    a = Matrix4()
    b = sqrt(3) / 2
    c = Matrix4().set(b, 0, 0.5, 0, 0, 1, 0, 0, -0.5, 0, b, 0, 0, 0, 0, 1)

    a.make_rotation_y(pi / 6)
    assert matrix_equals(a, c)


def test_make_rotation_z():

    a = Matrix4()
    b = sqrt(3) / 2
    c = Matrix4().set(b, -0.5, 0, 0, 0.5, b, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1)

    a.make_rotation_z(pi / 6)
    assert matrix_equals(a, c)


def test_make_rotation_axis():
    axis = Vector3(1.5, 0.0, 1.0).normalize()
    rads = radians(45)
    a = Matrix4().make_rotation_axis(axis, rads)

    expected = Matrix4().set(
        0.9098790095958609,
        -0.39223227027636803,
        0.13518148560620882,
        0,
        0.39223227027636803,
        0.7071067811865476,
        -0.588348405414552,
        0,
        0.13518148560620882,
        0.588348405414552,
        0.7972277715906868,
        0,
        0,
        0,
        0,
        1,
    )

    assert matrix_equals(a, expected), "Check numeric result"


def test_make_scale():
    a = Matrix4()
    c = Matrix4().set(2, 0, 0, 0, 0, 3, 0, 0, 0, 0, 4, 0, 0, 0, 0, 1)

    a.make_scale(2, 3, 4)
    assert matrix_equals(a, c)


def test_make_shear():
    a = Matrix4()
    c = Matrix4().set(1, 3, 4, 0, 2, 1, 4, 0, 2, 3, 1, 0, 0, 0, 0, 1)

    a.make_shear(2, 3, 4)
    assert matrix_equals(a, c)


def test_composedecompose():
    t_values = [
        Vector3(),
        Vector3(3, 0, 0),
        Vector3(0, 4, 0),
        Vector3(0, 0, 5),
        Vector3(-6, 0, 0),
        Vector3(0, -7, 0),
        Vector3(0, 0, -8),
        Vector3(-2, 5, -9),
        Vector3(-2, -5, -9),
    ]

    s_values = [
        Vector3(1, 1, 1),
        Vector3(2, 2, 2),
        Vector3(1, -1, 1),
        Vector3(-1, 1, 1),
        Vector3(1, 1, -1),
        Vector3(2, -2, 1),
        Vector3(-1, 2, -2),
        Vector3(-1, -1, -1),
        Vector3(-2, -2, -2),
    ]

    r_values = [
        Quaternion(),
        Quaternion().set_from_euler(Euler(1, 1, 0)),
        Quaternion().set_from_euler(Euler(1, -1, 1)),
        Quaternion(0, 0.9238795292366128, 0, 0.38268342717215614),
    ]

    for ti in range(len(t_values)):
        for si in range(len(s_values)):
            for ri in range(len(r_values)):
                t = t_values[ti]
                s = s_values[si]
                r = r_values[ri]

                m = Matrix4().compose(t, r, s)
                t2 = Vector3()
                r2 = Quaternion()
                s2 = Vector3()

                m.decompose(t2, r2, s2)

                m2 = Matrix4().compose(t2, r2, s2)

                ##
                # debug code
                # matrixIsSame = matrix_equals( m, m2 )
                #             if ( ! matrixIsSame ) {
                #     console.log( t, s, r )
                #                     console.log( t2, s2, r2 )
                #                     console.log( m, m2 )
                #                 }
                ##

                assert matrix_equals(m, m2)


def test_make_perspective():
    a = Matrix4().make_perspective(-1, 1, -1, 1, 1, 100)
    expected = Matrix4().set(
        1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -101 / 99, -200 / 99, 0, 0, -1, 0
    )
    assert matrix_equals(a, expected), "Check result"


def test_make_orthographic():
    a = Matrix4().make_orthographic(-1, 1, -1, 1, 1, 100)
    expected = Matrix4().set(
        1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -2 / 99, -101 / 99, 0, 0, 0, 1
    )

    assert matrix_equals(a, expected), "Check result"


def test_equals():
    a = Matrix4().set(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)
    b = Matrix4().set(0, -1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)

    assert not a.equals(b), "Check that a does not equal b"
    assert not b.equals(a), "Check that b does not equal a"

    a.copy(b)
    assert a.equals(b), "Check that a equals b after copy()"
    assert b.equals(a), "Check that b equals a after copy()"


def test_from_array():
    a = Matrix4()
    b = Matrix4().set(1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15, 4, 8, 12, 16)

    a.from_array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
    assert a.equals(b)


def test_to_array():
    a = Matrix4().set(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)
    no_offset = [1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15, 4, 8, 12, 16]
    with_offset = [None, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15, 4, 8, 12, 16]

    array = a.to_array()
    assert array == no_offset, "No array, no offset"

    array = []
    a.to_array(array)
    assert array == no_offset, "With array, no offset"

    array = []
    a.to_array(array, 1)
    assert array == with_offset, "With array, with offset"
