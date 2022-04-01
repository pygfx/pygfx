"""

Doelen:
* Beschrijven van points, directions, lines, and planes, en operaties erop.
* Beschrijven van scalor, rotator, translator, en een composite transform die alles bevat.
*
"""

import numpy as np


# %% Functional API

# These functions have rather verbose names, but it makes things
# explicit. Each function accepts either singletons or arrays of
# "things", and uses Numpy's broadcasting to just make it work. Each
# function also accepts an out argument.
#
# This API is for internal use and for power-users that want to
# vectorize operations on large sets of things.
# TODO: check shapes of input


def vector_add_vector(v1, v2, out=None):
    if out is None:
        out = np.empty_like(v1 if v1.size > v2.size else v2)
    out[:] = v1 + v2
    return out


def vector_mul_vector(v1, v2, out=None):
    if out is None:
        out = np.empty_like(v1 if v1.size > v2.size else v2)
    out[:] = v1 * v2
    return out


def vector_mul_scalar(v, s, out=None):
    if out is None:
        out = np.empty_like(v)
    out[:] = v * s
    return out


def point_add_vector(p, v, out=None):
    # Implementation is the same as vector_add_vector
    return vector_add_vector(p, v, out=out)


def quaternion_mul_quaternion(qa, qb):

    qax = qa[0]
    qay = qa[1]
    qaz = qa[2]
    qaw = qa[3]
    qbx = qb[0]
    qby = qb[1]
    qbz = qb[2]
    qbw = qb[3]

    qc = np.zeros_like(qa if qa.size > qb.size else qb)

    qc[0] = qax * qbw + qaw * qbx + qay * qbz - qaz * qby
    qc[1] = qay * qbw + qaw * qby + qaz * qbx - qax * qbz
    qc[2] = qaz * qbw + qaw * qbz + qax * qby - qay * qbx
    qc[3] = qaw * qbw - qax * qbx - qay * qby - qaz * qbz

    return qc


class LinalgBase:
    __slots__ = ["_val"]

    def __len__(self):
        return self._n

    def __getitem__(self, index):
        return self._val[index]

    def __iter__(self):
        return self._val.__iter__()

    @property
    def __array_interface__(self):
        # Numpy can wrap our memory in an array without copying
        return self._val.__array_interface__


# %% Object API

# In this API each "thing" is represented as one object.
# These objects are array-like and iterable to make them easy to
# convert to native Python/Numpy objects. The objects support mul
# and add where applicable, and have methods specific to the type of
# object.
#
# This API should make any linalg work much easier and safer, partly
# because semantics matters here: a point is not the same as a vector.


class LinalgObject(LinalgBase):
    pass


class Point(LinalgObject):
    """A representation of a location in 3D Euclidean space."""

    _n = 3

    def __init__(self, x, y, z):
        self._val = np.array([x, y, z], np.float64)

    def __repr__(self):
        x, y, z = self._val
        return f"<Point (x, y, z): {x:0.5g}, {y:0.5g}, {z:0.5g}>"

    def __add__(self, vector):
        if isinstance(vector, LinalgBase):
            if not isinstance(vector, Vector):
                raise TypeError("Can only add a Vector to a Point")
            arr = vector._val
        else:
            arr = np.asanyarray(vector)
        assert arr.shape == (3,)
        new = Point(0, 0, 0)
        point_add_vector(self._val, arr, out=new._val)
        return new

    def line_to_point(self, point):
        pass

    def line_to_plane(self, plane):
        pass


class Line:
    """Representation of a line in 3D Euclidean space."""

    _n = 8  # point + a vector

    def intersection_with_plane(self, plane):
        pass


class Plane:
    pass


class Cylinder:
    pass


class Sphere:
    pass


# %% Transformations


class Transform(LinalgBase):
    """Transforms can be applied to an object to transform it."""

    pass


class Vector(Transform):
    """Translate an object in xyz.
    TODO: maybe call this Translator or something because the term Vector is a bit generic.
    """

    _n = 3

    def __init__(self, dx, dy, dz):
        self._val = np.array([dx, dy, dz], np.float64)

    def __repr__(self):
        dx, dy, dz = self._val
        return f"<Vector (dx, dy, dz): {dx:0.5g}, {dy:0.5g}, {dz:0.5g}>"

    def __add__(self, vector):
        if isinstance(vector, LinalgBase):
            if not isinstance(vector, Vector):
                raise TypeError("Can only add a Vector to a Vector")
            arr = vector._val
        else:
            arr = np.asanyarray(vector)
        assert arr.shape == (4,)
        new = Vector(0, 0, 0)
        vector_add_vector(self._val, arr, out=new._val)
        return new

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            new = Vector(0, 0, 0)
            vector_mul_scalar(self._val, other, out=new._val)
            return new
        else:
            if isinstance(other, LinalgBase):
                if not isinstance(other, Scalor):
                    raise TypeError(
                        "Can only multiply a Vector with a scalar or Scalor."
                    )
                arr = other._val
            else:
                arr = np.asanyarray(other)
            assert arr.shape == (3,)
            new = Vector(0, 0, 0)
            vector_mul_vector(self._val, arr, out=new._val)
            return new

    def __rmul__(self, other):
        return Vector.__mul__(self, other)


class Rotor(Transform):
    """Rotate an object."""

    _n = 4  # a quaternion, but this is an implementation detail

    def __init__(self):
        self._val = np.array([0, 0, 0, 0], np.float64)

    def __repr__(self):
        x, y, z, w = self._val
        return f"<Rotor (quaternion): {x:0.5g}, {y:0.5g}, {z:0.5g}, {w:0.5g}>"

    def __mul__(self, other):
        if isinstance(other, (float, int)):
            new = Rotor()
            quaternion_mul_scalar(self._val, other, out=new._val)
            return new
        else:
            if isinstance(other, LinalgBase):
                if not isinstance(other, Rotor):
                    raise TypeError(
                        "Can only multiply a Rotor with a Rotor or a scalar."
                    )
                arr = other._val
            else:
                arr = np.asanyarray(other)
            assert arr.shape == (4,)
            new = Rotor(0, 0, 0)
            quaternion_mul_quaternion(self._val, arr, out=new._val)
            return new

    def __rmul__(self, other):
        return self.__mul__(self)


class Scalor(Transform):
    """Scale an object in xyz. May be non-uniform (stretch)."""

    _n = 3

    def __init__(self, sx, sy, sz):
        self._val = np.array([sx, sy, sz], np.float64)

    def __repr__(self):
        sx, sy, sz = self._val
        return f"<Scalor (sx, sy, sz): {sx:0.5g}, {sy:0.5g}, {sz:0.5g}>"

    def __mul__(self, other):
        if isinstance(other, (float, int)):
            new = Scalor()
            vector_mul_scalar(self._val, other, out=new._val)
            return new
        else:
            if isinstance(other, LinalgBase):
                if not isinstance(other, Scalor):
                    raise TypeError(
                        "Can only multiply a Scalor with a scalar or Scalor."
                    )
                arr = other._val
            else:
                arr = np.asanyarray(other)
            assert arr.shape == (3,)
            new = Scalor(0, 0, 0)
            vector_mul_vector(self._val, arr, out=new._val)
            return new


if __name__ == "__main__":
    p = Point(2, 3, 4)
    v = Vector(1, 0, 2)

    print(p)
    print(v)
    print(tuple(p))
    print(np.array(p))

    print()
    print(p + v)
    print(p + v * 2)
    print(p + v * Scalor(*v))
