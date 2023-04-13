import numpy as np
import pylinalg as la
from time import perf_counter_ns
import weakref
import functools

from typing import List, Tuple


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

    @property
    def matrix(self):
        raise NotImplementedError()

    @cached
    def inverse_matrix(self) -> np.ndarray:
        return np.linalg.inv(self.matrix)

    @cached
    def _decomposed(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return la.matrix_decompose(self.matrix)

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
        update_callback=None,
    ) -> None:
        super().__init__()
        self.last_modified = perf_counter_ns()
        self.update_callbacks = []

        if update_callback is not None:
            self.update_callbacks.append(update_callback)

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

        for callback in self.update_callbacks:
            callback(self)

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
            return AffineTransform(self.matrix @ other.matrix)

        return np.asarray(self) @ other

    def __array__(self, dtype=None):
        return self.untracked_matrix.astype(dtype, copy=False)


class ChainedTransform(AffineBase):
    def __init__(
        self, transform_sequence: List[AffineTransform], *, settable_index:int=None
    ) -> None:
        super().__init__()
        self.sequence = transform_sequence
        self.settable:AffineTransform = None
        self.before:AffineBase = None
        self.after:AffineBase = None

        if settable_index is not None:
            idx = settable_index
            before, after = self.sequence[:idx], self.sequence[idx:]
            target, after = after[0], after[1:]
            
            self.settable = target

            if len(before) > 0:
                self.before = ChainedTransform(before)
            else:
                self.before = AffineTransform()          
            
            if len(after) > 0:
                self.after = ChainedTransform(after)
            else:
                self.after = AffineTransform()

    @property
    def last_modified(self):
        return max(x.last_modified for x in self.sequence)

    @cached
    def _matrix(self):
        return la.matrix_combine(self.sequence)

    @property
    def matrix(self):
        return self._matrix

    @matrix.setter
    def matrix(self, value):
        if self.settable is None:
            raise AttributeError(
                "This ChainedTransform doesn't use `settable_index` and "
                "thus can't set properties."
            )
        
        new_value = self.before.inverse_matrix @ value @ self.after.inverse_matrix
        self.settable.matrix = new_value

    def __matmul__(self, other):
        if isinstance(other, ChainedTransform):
            return ChainedTransform(self.sequence + other.sequence)
        elif isinstance(other, AffineTransform):
            return ChainedTransform(self.sequence + [other])
        else:
            return np.asarray(self) @ other


class EmbeddedTransform(AffineBase):
    def __init__(
        self,
        total_transform: AffineTransform = None,
        *,
        before: AffineBase = None,
        after: AffineBase = None,
    ) -> None:
        if total_transform is None:
            total_transform = AffineTransform()
        self._total = total_transform

        self._before = before
        self._after = after
        self._last_modified = 0

    def flag_update(self):
        self._last_modified = perf_counter_ns()

    @property
    def last_modified(self):
        return max(
            (
                self._last_modified,
                self._total.last_modified,
                self.before.last_modified,
                self.after.last_modified,
            )
        )

    @cached
    def _matrix(self):
        return self._before.inverse_matrix @ self._total @ self._after.inverse_matrix

    @property
    def matrix(self):
        return self._matrix

    @matrix.setter
    def matrix(self, value: np.ndarray):
        # preserve buffer
        self._total.untracked_matrix[:] = (
            self._before.matrix @ np.asarray(value) @ self._after.matrix
        )
        self._total.flag_update()

    @property
    def before(self) -> AffineBase:
        if self._before is None:
            self._before = AffineTransform()

        return self._before

    @before.setter
    def before(self, value: AffineBase) -> None:
        self._before = value
        self.flag_update()

    @property
    def after(self) -> AffineBase:
        if self._after is None:
            self._after = AffineTransform()

        return self._after

    def __matmul__(self, other):
        return np.asarray(self) @ other
