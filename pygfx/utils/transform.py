import numpy as np
import pylinalg as pla
from datetime import datetime

from typing import List

OUTDATED = datetime(1900, 1, 1)  # some date in the past


class AffineBase:
    last_modified: datetime
    matrix: np.ndarray
    position: np.ndarray
    rotation: np.ndarray
    scale: np.ndarray

    def __init__(self) -> None:
        self._components_age = OUTDATED
        self._matrix: np.ndarray = None
        self._position: np.ndarray = None
        self._rotation: np.ndarray = None
        self._scale: np.ndarray = None

        self._inverse_age = OUTDATED
        self._inverse_matrix: np.ndarray = None

    def _update_inverse(self):
        if self.last_modified < self._inverse_age:
            return

        self._inverse_matrix = np.linalg.inv(self.matrix)

    @property
    def inverse_matrix(self):
        self._update_inverse()
        return self._inverse_matrix

    def __array__(self, dtype=None):
        return self.matrix.astype(dtype)


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

    @property
    def position(self) -> np.ndarray:
        self._update_cache()
        return self._position

    @position.setter
    def position(self, value):
        self.matrix[:-1, -1] = value
        self.last_modified = datetime.now()

    @property
    def rotation(self) -> np.ndarray:
        self._update_cache()
        return self._rotation

    @rotation.setter
    def rotation(self, value):
        position = self.position
        scale = self.scale

        pla.matrix_make_transform(position, value, scale, out=self.matrix)
        self.last_modified = datetime.now()

    @property
    def scale(self) -> np.ndarray:
        self._update_cache()
        return self._scale

    @scale.setter
    def scale(self, value):
        position = self.position
        rotation = self.rotation

        pla.matrix_make_transform(position, rotation, value, out=self.matrix)
        self.last_modified = datetime.now()

    def _update_cache(self):
        if self.last_modified < self._components_age:
            return

        if self._position is None:
            self._position = np.zeros(3, dtype=float)
            self._rotation = np.zeros(4, dtype=float)
            self._scale = np.zeros(3, dtype=float)

        pla.matrix_decompose(
            self.matrix, out=(self._position, self._rotation, self._scale)
        )
        self._components_age = datetime.now()

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

    def _update_cache(self):
        if self._components_age > self.last_modified:
            return

        if self._matrix is None:
            self._matrix = np.ones((4, 4), dtype=float)
            self._position = np.zeros(3, dtype=float)
            self._rotation = np.zeros(4, dtype=float)
            self._scale = np.zeros(3, dtype=float)

        self._matrix = np.eye(4, dtype=float)
        for transform in self.sequence:
            np.matmul(self._matrix, transform, out=self._matrix)

        pla.matrix_decompose(
            self._matrix, out=(self._position, self._rotation, self._scale)
        )
        self._components_age = datetime.now()

    @property
    def matrix(self):
        self._update_cache()
        return self._matrix

    @property
    def position(self):
        self._update_cache()
        return self._position

    @property
    def rotation(self):
        self._update_cache()
        return self._rotation

    @property
    def scale(self):
        self._update_cache()
        return self._scale

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

        self.before = AffineTransform()
        self.after = AffineTransform()
        self.linked = linked_transform

        if before is not None:
            self.before = before

        if after is not None:
            self.after = after

    @property
    def last_modified(self):
        value = self.linked.last_modified

        if value < self.before.last_modified:
            value = self.before.last_modified

        if value < self.after.last_modified:
            value = self.after.last_modified

        return value

    def _update_cache(self):
        if self._components_age > self.last_modified:
            return

        if self._matrix is None:
            self._matrix = np.ones((4, 4), dtype=float)
            self._position = np.zeros(3, dtype=float)
            self._rotation = np.zeros(4, dtype=float)
            self._scale = np.zeros(3, dtype=float)

        self._matrix = (self.before @ self.linked @ self.after).matrix
        pla.matrix_decompose(
            self._matrix, out=(self._position, self._rotation, self._scale)
        )
        self._components_age = datetime.now()

    @property
    def matrix(self):
        self._update_cache()
        return self._matrix

    @matrix.setter
    def matrix(self, value):
        new_link = (
            self.before.inverse_matrix @ np.asarray(value) @ self.after.inverse_matrix
        )
        self.linked.matrix = new_link
        self.linked.last_modified = datetime.now()

    @property
    def position(self):
        self._update_cache()
        return self._position

    @position.setter
    def position(self, value):
        total_transform = pla.matrix_make_transform(value, self.rotation, self.scale)
        self.matrix = total_transform

    @property
    def rotation(self):
        self._update_cache()
        return self._rotation

    @rotation.setter
    def rotation(self, value):
        total_transform = pla.matrix_make_transform(self.position, value, self.scale)
        self.matrix = total_transform

    @property
    def scale(self):
        self._update_cache()
        return self._scale

    @scale.setter
    def scale(self, value):
        total_transform = pla.matrix_make_transform(self.position, self.rotation, value)
        self.matrix = total_transform
