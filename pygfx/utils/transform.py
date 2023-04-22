import numpy as np
import pylinalg as la
from time import perf_counter_ns
import weakref
import functools

from typing import List, Tuple, Union


class cached:  # noqa: N801
    """Cache for computed properties.

    This descriptor implements a minimal timestamp-based cache for computed
    properties. The value of the property is computed using ``compute_fn`` and
    the result is cached until ``obj.last_modified`` advances. At this point the
    value of the computed property is recomputed upon the next read.

    """

    def __init__(self, compute_fn=None) -> None:
        self.compute_fn = compute_fn
        self.name = None

    def __set_name__(self, clazz, name) -> None:
        self.name = f"_{name}_cache"

    def __get__(self, instance: "AffineBase", clazz=None) -> np.ndarray:
        if instance is None:
            return self

        if not hasattr(instance, self.name):
            cache = (instance.last_modified, self.compute_fn(instance))
            setattr(instance, self.name, cache)
        else:
            cache = getattr(instance, self.name)

        if instance.last_modified > cache[0]:
            cache = (instance.last_modified, self.compute_fn(instance))
            setattr(instance, self.name, cache)

        return cache[1]


class callback:  # noqa: N801
    """Weakref support for AffineTransform callbacks.

    This decorator replaces the instance reference of a class method (self) with
    a weakref to the instance. It also dynamically resolves this weakref
    so that users don't have to.

    The use-case for it is to avoid the circular reference that happens when a
    WorldObject registers a callback to keep it's local transform in sync with
    its world transform. This speeds up garbage collection as WorldObjects will
    get removed once the last reference is deleted instead of waiting for GC's
    cycle detection phase.

    """

    def __init__(self, callback_fn) -> None:
        self.callback_fn = callback_fn

    def __get__(self, instance, clazz=None):
        if instance is None:
            return self

        weak_instance = weakref.ref(instance)

        @functools.wraps(self.callback_fn)
        def inner(*args, **kwargs):
            if weak_instance() is None:
                return

            return self.callback_fn(weak_instance(), *args, **kwargs)

        return inner


class AffineBase:
    last_modified: int

    def __init__(self, forward_is_minus_z=False):
        self.update_callbacks = {}
        self.forward_is_minus_z = int(forward_is_minus_z)

    @property
    def matrix(self):
        raise NotImplementedError()

    @cached
    def inverse_matrix(self) -> np.ndarray:
        return np.linalg.inv(self.matrix)

    @cached
    def _decomposed(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return la.matrix_decompose(self.matrix)

    @cached
    def directions(self):
        # Note: forward_is_minus_z indicates the camera frame
        directions = (
            2 * self.forward_is_minus_z - 1,
            1,
            1 - 2 * self.forward_is_minus_z
        )

        axes = np.diag(directions)
        return (*la.vector_apply_matrix(axes, self.matrix),)

    def flag_update(self):
        for callback in self.update_callbacks.values():
            callback(self)

    def on_update(self, callback):
        callback_id = id(callback)
        self.update_callbacks[callback_id] = callback

        return callback_id

    def remove_callback(self, ref):
        """Ref can be the callback function itself or the callback ID returned by `on_update`."""

        if isinstance(ref, int):
            callback_id = ref
        else:
            callback_id = id(ref)

        self.update_callbacks.pop(callback_id, None)

    @property
    def position(self) -> np.ndarray:
        return self._decomposed[0]

    @property
    def rotation(self) -> np.ndarray:
        return self._decomposed[1]

    @property
    def scale(self) -> np.ndarray:
        return self._decomposed[2]

    @property
    def x(self) -> float:
        return self.position[0]

    @property
    def y(self) -> float:
        return self.position[1]

    @property
    def z(self) -> float:
        return self.position[2]

    @property
    def scale_x(self) -> float:
        return self.scale[0]

    @property
    def scale_y(self) -> float:
        return self.scale[1]

    @property
    def scale_z(self) -> float:
        return self.scale[2]

    @position.setter
    def position(self, value):
        self.matrix = la.matrix_make_transform(value, self.rotation, self.scale)

    @rotation.setter
    def rotation(self, value):
        self.matrix = la.matrix_make_transform(self.position, value, self.scale)

    @scale.setter
    def scale(self, value):
        self.matrix = la.matrix_make_transform(self.position, self.rotation, value)

    @x.setter
    def x(self, value):
        _, y, z = self.position
        self.position = (value, y, z)

    @y.setter
    def y(self, value):
        x, _, z = self.position
        self.position = (x, value, z)

    @z.setter
    def z(self, value):
        x, y, _ = self.position
        self.position = (x, y, value)

    @scale_x.setter
    def scale_x(self, value):
        _, y, z = self.scale
        self.scale = (value, y, z)

    @scale_y.setter
    def scale_y(self, value):
        x, _, z = self.scale
        self.scale = (x, value, z)

    @scale_z.setter
    def scale_z(self, value):
        x, y, _ = self.scale
        self.scale = (x, y, value)

    def __array__(self, dtype=None):
        return self.matrix.astype(dtype)

    @property
    def right(self):
        return self.directions[0]

    @property
    def up(self):
        return self.directions[1]

    @property
    def forward(self):
        return self.directions[2]


class AffineTransform(AffineBase):
    """An affine tranformation"""

    def __init__(
        self,
        matrix=None,
        /,
        *,
        position=None,
        rotation=None,
        scale=None,
        forward_is_minus_z=False,
    ) -> None:
        super().__init__(forward_is_minus_z)
        self.last_modified = perf_counter_ns()

        if matrix is None:
            matrix = np.eye(4, dtype=float)

        self.untracked_matrix = np.asarray(matrix)

        if position is not None:
            self.position = position

        if rotation is not None:
            self.rotation = rotation

        if scale is not None:
            self.scale = scale

    def flag_update(self):
        self.last_modified = perf_counter_ns()
        super().flag_update()

    @property
    def matrix(self):
        view_array = self.untracked_matrix.view()
        view_array.flags.writeable = False
        return view_array

    @matrix.setter
    def matrix(self, value):
        self.untracked_matrix[:] = value
        self.flag_update()

    def __matmul__(self, other):
        if isinstance(other, AffineTransform):
            return AffineTransform(
                self.matrix @ other.matrix, forward_is_minus_z=self.forward_is_minus_z
            )

        return np.asarray(self) @ other

    def __array__(self, dtype=None):
        return self.untracked_matrix.astype(dtype, copy=False)


class RecursiveTransform(AffineBase):
    """A transform that may be preceeded by another transform."""

    def __init__(
        self,
        matrix: Union[np.ndarray, AffineBase],
        /,
        *,
        parent=None,
        forward_is_minus_z=False,
    ) -> None:
        super().__init__(forward_is_minus_z)
        self._parent = None
        self.own = None
        self._last_modified = perf_counter_ns()

        if isinstance(matrix, AffineBase):
            self.own = matrix
        else:
            self.own = AffineTransform(matrix, forward_is_minus_z=forward_is_minus_z)

        if parent is None:
            self._parent = AffineTransform()
        else:
            self._parent = parent

        self.own.on_update(self.update_pipe)
        self.parent.on_update(self.update_pipe)

    @property
    def last_modified(self):
        return max(
            self.own.last_modified, self._parent.last_modified, self._last_modified
        )

    def flag_update(self):
        self._last_modified = perf_counter_ns()
        super().flag_update()

    @callback
    def update_pipe(self, other: AffineBase):
        self.flag_update()

    @property
    def parent(self) -> AffineBase:
        return self._parent

    @parent.setter
    def parent(self, value):
        self.parent.remove_callback(self.update_pipe)

        if value is None:
            self._parent = AffineTransform()
        else:
            self._parent = value

        self.parent.on_update(self.update_pipe)
        self.flag_update()

    @cached
    def _matrix(self):
        return self._parent.matrix @ self.own.matrix

    @property
    def matrix(self):
        return self._matrix

    @matrix.setter
    def matrix(self, value):
        self.own.matrix = self._parent.inverse_matrix @ value

    def __matmul__(self, other):
        if isinstance(other, AffineBase):
            return RecursiveTransform(other, parent=self)
        else:
            return np.asarray(self) @ other
