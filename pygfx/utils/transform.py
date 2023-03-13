import numpy as np
import pylinalg as pla
from datetime import datetime


class AffineTransform:
    """An affine tranformation"""

    def __init__(
        self, matrix=None, /, *, position=None, rotation=None, scale=None
    ) -> None:
        self.last_updated = datetime.now()

        if matrix is None:
            matrix = np.eye(4, dtype=float)

        self.matrix = np.asarray(matrix)
        self._cache_time = datetime(1900, 1, 1)  # some date in the past
        self._position = None
        self._rotation = None
        self._scale = None

        if position is not None:
            self.position = position

        if rotation is not None:
            self.rotation = rotation

        if scale is not None:
            self.scale = scale

    def flag_update(self):
        self.last_updated = datetime.now()

    @property
    def position(self) -> np.ndarray:
        self._update_cache()
        return self._position

    @position.setter
    def position(self, value):
        self.matrix[:-1, -1] = value
        self.last_updated = datetime.now()

    @property
    def rotation(self) -> np.ndarray:
        self._update_cache()
        return self._rotation

    @rotation.setter
    def rotation(self, value):
        position = self.position
        scale = self.scale

        pla.matrix_make_transform(position, value, scale, out=self.matrix)
        self.last_updated = datetime.now()

    @property
    def scale(self) -> np.ndarray:
        self._update_cache()
        return self._scale

    @scale.setter
    def scale(self, value):
        position = self.position
        rotation = self.rotation

        pla.matrix_make_transform(position, rotation, value, out=self.matrix)
        self.last_updated = datetime.now()

    def _update_cache(self):
        if self.last_updated < self._cache_time:
            return

        if self._position is None:
            self._position = np.zeros(3, dtype=float)
            self._rotation = np.zeros(4, dtype=float)
            self._scale = np.zeros(3, dtype=float)

        pla.matrix_decompose(
            self.matrix, out=(self._position, self._rotation, self._scale)
        )
        self._cache_time = datetime.now()

    def __array__(self, dtype=None):
        return self.matrix.astype(dtype)

    def __matmul__(self, other):
        if isinstance(other, AffineTransform):
            return AffineTransform(self.matrix @ other.matrix)

        return np.asarray(self) @ other

    def look_at(self, target) -> None:
        rotation = pla.matrix_make_look_at(self.position, target, (0, 1, 0))
        rotation = pla.matrix_to_quaternion(rotation)
        self.rotation = pla.quaternion_multiply(rotation, self.rotation)
