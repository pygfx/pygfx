import numpy as np
import pylinalg as pla
from datetime import datetime

from typing import List


class cached:
    """Cache for computed properties.

    This descriptor implements a minimal timestamp-based cache for computed
    properties. The value of the property is computed using ``update_fn`` and
    the result is cached until ``obj.last_modified`` advances. At this point the
    value of the computed property is recomputed upon the next read.



    Example
    -------

    class Foobar:
        def __init__(self, x):
            self._value = 0
            self.last_modified = datetime.now()

        @property
        def value(self):
            return self._value

        @value.setter
        def value(self, new_value):
            self._value = new_value
            self.last_modified = datetime.now()

        @property
        @cached
        def double(self):
            print("Computed value of double..")
            return 2*x

    """

    def __init__(self, update_fn=None) -> None:
        self.update_fn = update_fn
        self.last_updated = None
        self.cached_value = None

    def __get__(self, obj: "AffineBase", objtype=None):
        if obj is None:
            return self

        if self.last_updated is None or obj.last_modified > self.last_updated:
            self.cached_value = self.update_fn(obj)
            self.last_updated = obj.last_modified

        return self.cached_value

    def __call__(self, obj):
        """Helper to allow interoperability with @property."""
        return self.__get__(obj)


class AffineBase:
    last_modified: datetime
    matrix: np.ndarray

    @property
    @cached
    def inverse_matrix(self):
        return np.linalg.inv(self.matrix)

    @property
    @cached
    def position(self) -> np.ndarray:
        pos, _, _ = pla.matrix_decompose(self.matrix)
        return pos

    @property
    @cached
    def rotation(self) -> np.ndarray:
        _, rot, _ = pla.matrix_decompose(self.matrix)
        return rot

    def __array__(self, dtype=None):
        return self.matrix.astype(dtype)

    @property
    @cached
    def scale(self) -> np.ndarray:
        _, _, scale = pla.matrix_decompose(self.matrix)
        return scale


class AffineTransform(AffineBase):
    """An affine tranformation"""

    def __init__(
        self, matrix=None, /, *, position=None, rotation=None, scale=None
    ) -> None:
        super().__init__()
        self.last_modified = datetime.now()

        if matrix is None:
            matrix = np.eye(4, dtype=float)

        self.matrix = np.asarray(matrix)

        if position is not None:
            self.position = position

        if rotation is not None:
            self.rotation = rotation

        if scale is not None:
            self.scale = scale

    def flag_update(self):
        self.last_modified = datetime.now()

    @AffineBase.position.setter
    def position(self, value):
        self.matrix[:-1, -1] = value
        self.last_modified = datetime.now()

    @AffineBase.rotation.setter
    def rotation(self, value):
        position = self.position
        scale = self.scale

        pla.matrix_make_transform(position, value, scale, out=self.matrix)
        self.last_modified = datetime.now()

    @AffineBase.scale.setter
    def scale(self, value):
        position = self.position
        rotation = self.rotation

        pla.matrix_make_transform(position, rotation, value, out=self.matrix)
        self.last_modified = datetime.now()

    def __matmul__(self, other):
        if isinstance(other, AffineTransform):
            return AffineTransform(self.matrix @ other.matrix)

        return np.asarray(self) @ other

    def look_at(self, target) -> None:
        rotation = pla.matrix_make_look_at(self.position, target, (0, 1, 0))
        rotation = pla.matrix_to_quaternion(rotation)
        self.rotation = pla.quaternion_multiply(rotation, self.rotation)


class ChainedTransform(AffineBase):
    def __init__(self, transform_sequence: List[AffineTransform]) -> None:
        super().__init__()
        self.sequence = transform_sequence

    @property
    def last_modified(self):
        candidate = self.sequence[0].last_modified
        for transform in self.sequence:
            if transform.last_modified > candidate:
                candidate = transform.last_modified

        return candidate

    @property
    @cached
    def matrix(self):
        result = np.eye(4, dtype=float)
        for transform in self.sequence:
            np.matmul(result, transform, out=result)

        return result

    def __matmul__(self, other):
        if isinstance(other, ChainedTransform):
            return ChainedTransform(self.sequence + other.sequence)
        elif isinstance(other, AffineTransform):
            return ChainedTransform(self.sequence + [other])
        else:
            return np.asarray(self) @ other


class LinkedTransform(AffineBase):
    def __init__(
        self,
        linked_transform: AffineTransform,
        *,
        before: AffineBase = None,
        after: AffineBase = None,
    ) -> None:
        super().__init__()

        self.linked = linked_transform

        if before is not None:
            self.before = before
        else:
            self.before = AffineTransform()

        if after is not None:
            self.after = after
        else:
            self.after = AffineTransform()

    @property
    def last_modified(self):
        value = self.linked.last_modified

        if value < self.before.last_modified:
            value = self.before.last_modified

        if value < self.after.last_modified:
            value = self.after.last_modified

        return value

    @property
    def matrix(self):
        return (self.before @ self.linked @ self.after).matrix

    @matrix.setter
    def matrix(self, value):
        new_link = (
            self.before.inverse_matrix @ np.asarray(value) @ self.after.inverse_matrix
        )
        self.linked.matrix = new_link
        self.linked.last_modified = datetime.now()

    @AffineBase.position.setter
    def position(self, value):
        total_transform = pla.matrix_make_transform(value, self.rotation, self.scale)
        self.matrix = total_transform

    @AffineBase.rotation.setter
    def rotation(self, value):
        total_transform = pla.matrix_make_transform(self.position, value, self.scale)
        self.matrix = total_transform

    @AffineBase.scale.setter
    def scale(self, value):
        total_transform = pla.matrix_make_transform(self.position, self.rotation, value)
        self.matrix = total_transform
