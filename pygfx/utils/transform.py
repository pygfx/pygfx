from __future__ import annotations

import numpy as np
import pylinalg as la
from time import perf_counter_ns

from typing import Any, Tuple, Union


PRECISION_EPSILON = 1e-7


class cached:  # noqa: N801
    """Cache for computed properties.

    This descriptor implements a minimal timestamp-based cache for computed
    properties. The value of the property is computed using ``compute_fn`` and
    the result is cached until ``obj.last_modified`` advances. At this point the
    value of the computed property is recomputed upon the next read.

    """

    __slots__ = ("compute_fn", "name")

    def __init__(self, compute_fn=None) -> None:
        self.compute_fn = compute_fn
        self.name = None

    def __set_name__(self, clazz, name) -> None:
        self.name = f"cache_{name}_cache"

    def __get__(self, instance: AffineBase, clazz=None) -> Any:
        if instance is None:
            return self

        last_modified = instance.last_modified
        cache = getattr(instance, self.name, None)

        if cache is None or last_modified > cache[0]:
            cache = (last_modified, self.compute_fn(instance))
            setattr(instance, self.name, cache)

        return cache[1]


class AffineBase:
    """Base class for affine transformations.

    .. warning::
        In-place updates of slices of properties, e.g. ``transform.position[1] =
        42`` have no effect due to limitations of the python programming
        language and our decision to have the properties return pure numpy arrays.
        Where possible these arrays are flagged as read-only.

    This class implements basic getters and setters for the various properties
    of an affine transformation used in pygfx. If you are looking for
    `obj.local.<something>` or `obj.world.<something>` it is probably defined
    here.

    It also implements a basic caching mechanism that keeps computed properties
    around until the underlying transform changes. This makes use of the
    `@cached` descriptor defined above. The descriptor expects a `last_modified`
    property on the consuming class which is used as a monotonously increasing
    timestamp/counter to indicate if and when a cached value has become invalid.
    Once invalid, cached values are updated upon the next read/get meaning that
    they are updated lazily.

    Parameters
    ----------
    reference_up : ndarray, [3]
        The direction of the reference_up vector expressed in the target frame.
        It indicates neutral tilt and is used by the axis properties (right, up,
        forward) to maintain a common level of rotation around an axis when it
        is updated by it's setter. By default, it points along the Y-axis.
    is_camera_space : bool
        If True, the transform represents a camera space which means that it's
        ``forward`` and ``right`` directions are inverted.

    Notes
    -----
    Subclasses need to define and implement ``last_modified`` for the caching
    mechanism to work correctly. Check out existing subclasses for an example of
    how this might look like.

    All properties are **expressed in the target frame**, i.e., they use the
    target's basis, unless otherwise specified.

    """

    __slots__ = (
        "_reference_up",
        "_reference_up_view",
        "_scaling_signs",
        "_scaling_signs_view",
        "cache__decomposed_cache",
        "cache__decomposed_position_cache",
        "cache__directions_cache",
        "cache__euler_cache",
        "cache__inverse_matrix_cache",
        "cache__rotation_matrix_cache",
        "is_camera_space",
    )

    last_modified: int

    def __init__(self, /, *, reference_up=(0, 1, 0), is_camera_space=False):
        self.is_camera_space = int(is_camera_space)

        self._reference_up = la.vec_normalize(reference_up, dtype=float)
        self._reference_up_view = self._reference_up.view()
        self._reference_up_view.flags.writeable = False

        self._scaling_signs = np.asarray([1, 1, 1], dtype=float)
        self._scaling_signs_view = self._scaling_signs.view()
        self._scaling_signs_view.flags.writeable = False

    def flag_update(self):
        """Signal that this transform has updated."""
        raise NotImplementedError()

    @property
    def reference_up(self) -> np.ndarray:
        """The zero-tilt reference vector used for the direction setters."""
        return self._reference_up_view

    @reference_up.setter
    def reference_up(self, value):
        la.vec_normalize(value, out=self._reference_up)

    @property
    def matrix(self) -> np.ndarray:
        """Affine matrix describing this transform.

        ``vec_target = matrix @ vec_source``.

        """
        raise NotImplementedError()

    @cached
    def _inverse_matrix(self) -> np.ndarray:
        mat = la.mat_inverse(self.matrix)
        mat.flags.writeable = False
        return mat

    @property
    def scaling_signs(self) -> np.ndarray:
        """Property used to track and preserve the scale factor signs
        over matrix decomposition operations."""
        return self._scaling_signs_view

    @cached
    def _decomposed_position(self) -> np.ndarray:
        position = la.mat_decompose_translation(self.matrix)
        position.flags.writeable = False
        return position

    @cached
    def _decomposed(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        try:
            decomposed = la.mat_decompose(self.matrix, scaling_signs=self.scaling_signs)
        except ValueError:
            # the matrix has been set manually
            # and so there is no scaling component to preserve
            # any decomposed scaling is acceptable
            decomposed = la.mat_decompose(self.matrix)

        decomposed[0].flags.writeable = False
        decomposed[1].flags.writeable = False
        decomposed[2].flags.writeable = False

        return decomposed

    @cached
    def _directions(self):
        # Note: forward_is_minus_z indicates the camera frame
        directions = (2 * self.is_camera_space - 1, 1, 1 - 2 * self.is_camera_space)

        axes_source = np.diag(directions)
        axes_target = la.vec_transform(axes_source, self.matrix)
        origin_target = la.vec_transform((0, 0, 0), self.matrix)

        directions = axes_target - origin_target
        directions.flags.writeable = False
        return directions

    @cached
    def _rotation_matrix(self):
        rotation = la.mat_from_quat(self._decomposed[1])
        rotation.flags.writeable = False
        return rotation

    @cached
    def _euler(self):
        euler = la.quat_to_euler(self._decomposed[1])
        euler.flags.writeable = False
        return euler

    @property
    def inverse_matrix(self) -> np.ndarray:
        """Inverse of this transform as affine matrix.

        ``vec_source = inverse_matrix @ vec_target``

        """
        return self._inverse_matrix

    @property
    def position(self) -> np.ndarray:
        """The origin of source."""
        return self._decomposed_position

    @property
    def rotation(self) -> np.ndarray:
        """The orientation of source as a quaternion."""
        return self._decomposed[1]

    @property
    def rotation_matrix(self) -> np.ndarray:
        """The orientation of source as a rotation matrix."""
        return self._rotation_matrix

    @property
    def euler(self) -> np.ndarray:
        """The orientation of source as XYZ euler angles."""
        return self._euler

    @property
    def euler_x(self) -> float:
        """The X component of source's orientation as XYZ euler angles."""
        return self._euler[0]

    @property
    def euler_y(self) -> float:
        """The Y component of source's orientation as XYZ euler angles."""
        return self._euler[1]

    @property
    def euler_z(self) -> float:
        """The Z component of source's orientation as XYZ euler angles."""
        return self._euler[2]

    @property
    def scale(self) -> np.ndarray:
        """The scale of source."""
        return self._decomposed[2]

    @property
    def x(self) -> float:
        """The X component of source's position."""
        return self.position[0]

    @property
    def y(self) -> float:
        """The Y component of source's position."""
        return self.position[1]

    @property
    def z(self) -> float:
        """The Z component of source's position."""
        return self.position[2]

    @property
    def scale_x(self) -> float:
        """The X component of source's scale."""
        return self.scale[0]

    @property
    def scale_y(self) -> float:
        """The Y component of source's scale."""
        return self.scale[1]

    @property
    def scale_z(self) -> float:
        """The Z component of source's scale."""
        return self.scale[2]

    @property
    def right(self) -> np.ndarray:
        """The right direction of source."""
        return self._directions[0]

    @property
    def up(self) -> np.ndarray:
        """The up direction of source."""
        return self._directions[1]

    @property
    def forward(self) -> np.ndarray:
        """The forward direction of source."""
        return self._directions[2]

    @position.setter
    def position(self, value):
        self.matrix = la.mat_compose(value, self.rotation, self.scale)

    @rotation.setter
    def rotation(self, value):
        self.matrix = la.mat_compose(self.position, value, self.scale)

    @euler.setter
    def euler(self, value):
        self.rotation = la.quat_from_euler(value)

    @rotation_matrix.setter
    def rotation_matrix(self, value):
        self.rotation = la.quat_from_mat(value)

    @euler_x.setter
    def euler_x(self, value):
        self.euler = (value, 0, 0)

    @euler_y.setter
    def euler_y(self, value):
        self.euler = (0, value, 0)

    @euler_z.setter
    def euler_z(self, value):
        self.euler = (0, 0, value)

    @scale.setter
    def scale(self, value):
        m = la.mat_compose(self.position, self.rotation, value)
        np.sign(value, out=self._scaling_signs)
        self.matrix = m

    @property
    def components(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self._decomposed

    @components.setter
    def components(self, value: Tuple[np.ndarray, np.ndarray, np.ndarray]):
        m = la.mat_compose(*value)
        np.sign(value[2], out=self._scaling_signs)
        self.matrix = m

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

    @right.setter
    def right(self, value):
        value = la.vec_normalize(value)

        if np.allclose(value, (0, 0, 0)):
            raise ValueError("A coordinate axis can't point at its origin.")

        if np.linalg.norm(np.cross(value, self.reference_up)) == 0:
            # target and reference_up are parallel
            rotation = la.quat_from_vecs(self.forward, value)
        else:
            matrix = la.mat_look_at((0, 0, 0), value, self.reference_up)
            rotation = la.quat_from_mat(matrix)

        if self.is_camera_space:
            part2 = la.quat_from_axis_angle((0, 1, 0), -np.pi / 2)
        else:
            part2 = la.quat_from_axis_angle((0, 1, 0), np.pi / 2)

        self.rotation = la.quat_mul(rotation, part2)

    @up.setter
    def up(self, value):
        value = la.vec_normalize(value)

        if np.allclose(value, (0, 0, 0)):
            raise ValueError("A coordinate axis can't point at its origin.")

        if np.linalg.norm(np.cross(value, self.reference_up)) == 0:
            # target and reference_up are parallel
            rotation = la.quat_from_vecs(self.forward, value)
        else:
            matrix = la.mat_look_at((0, 0, 0), value, self.reference_up)
            rotation = la.quat_from_mat(matrix)

        part2 = la.quat_from_axis_angle((1, 0, 0), np.pi / 2)
        self.rotation = la.quat_mul(rotation, part2)

    @forward.setter
    def forward(self, value):
        value = la.vec_normalize(value)

        if np.allclose(value, (0, 0, 0)):
            raise ValueError("A coordinate axis can't point at its origin.")

        if np.linalg.norm(np.cross(value, self.reference_up)) < PRECISION_EPSILON:
            # target and reference_up are parallel
            if np.linalg.norm(np.cross(self.forward, value)) < PRECISION_EPSILON:
                # target and forward are parallel, no rotation needed
                return
            rotation = la.quat_from_vecs(self.forward, value)
        elif self.is_camera_space:
            matrix = la.mat_look_at((0, 0, 0), -value, self.reference_up)
            rotation = la.quat_from_mat(matrix)
        else:
            matrix = la.mat_look_at((0, 0, 0), value, self.reference_up)
            rotation = la.quat_from_mat(matrix)

        self.rotation = rotation

    def __array__(self, dtype=None) -> np.ndarray:
        return self.matrix.astype(dtype, copy=False)


class AffineTransform(AffineBase):
    """A single affine transform.

    Parameters
    ----------
    position : ndarray, [3]
        The position of this transform expressed in the target frame.
        This argument is only consumed if ``state_basis`` is set to "components".
    rotation : ndarray, [4]
        The rotation quaternion of this transform expressed in the target frame.
        This argument is only consumed if ``state_basis`` is set to "components".
    scale : ndarray, [3]
        The per-axis scale of this transform expressed in the target frame.
        This argument is only consumed if ``state_basis`` is set to "components".
    reference_up : ndarray, [3]
        The direction of the reference_up vector expressed in the target frame.
        It indicates neutral tilt and is used by the axis properties (right, up,
        forward) to maintain a common level of rotation around an axis when it
        is updated by it's setter. By default, it points along the Y-axis.
    is_camera_space : bool
        If True, the transform represents a camera space which means that it's
        ``forward`` and ``right`` directions are inverted.
    state_basis : str
        One of "components" or "matrix". The default is "components".
        If "components", the transform will store its state in the form of
        position, rotation, and scale, while the matrix will be computed.
        If "matrix", the transform will store its state in the form of a matrix,
        while the position, rotation and scale will be computed.
        This option is provided to allow for performance optimizations when
        the transform is used in a context where the state is updated in one
        form but queried in another. Depending on your usage patterns, you
        may avoid significant overhead by choosing the appropriate basis.
    matrix : ndarray, [4, 4]
        The affine matrix describing this transform.
        This argument is only consumed if ``state_basis`` is set to "matrix".

    Notes
    -----
    When updating the underlying numpy arrays in-place these updates will not
    propagate via the transform's callback system nor will they invalidate
    existing caches. To inform the transform of these updates call
    ``transform.flag_update()``, which will trigger both callbacks and cache
    invalidation.

    See Also
    --------
    AffineBase
        Base class defining various useful properties for this transform.

    """

    __slots__ = (
        "_matrix",
        "_matrix_view",
        "_position",
        "_position_view",
        "_rotation",
        "_rotation_view",
        "_scale",
        "_scale_view",
        "_state_basis",
        "_wrapper",
        "cache__composed_matrix_cache",
        "cache__computed_scaling_signs_cache",
        "last_modified",
    )

    def __init__(
        self,
        position=(0, 0, 0),
        rotation=(0, 0, 0, 1),
        scale=(1, 1, 1),
        *,
        reference_up=(0, 1, 0),
        is_camera_space=False,
        state_basis="components",
        matrix: np.ndarray = None,
    ) -> None:
        super().__init__(reference_up=reference_up, is_camera_space=is_camera_space)
        self._state_basis = state_basis
        self.last_modified = perf_counter_ns()

        self._position = np.asarray(position, dtype=float)
        self._rotation = np.asarray(rotation, dtype=float)
        self._scale = np.asarray(scale, dtype=float)

        self._position_view = self._position.view()
        self._position_view.flags.writeable = False
        self._rotation_view = self._rotation.view()
        self._rotation_view.flags.writeable = False
        self._scale_view = self._scale.view()
        self._scale_view.flags.writeable = False

        # The ._matrix is only used when state_basis is "matrix"
        if state_basis == "matrix" and matrix is not None:
            self._matrix = np.asarray(matrix, dtype=float)
        else:
            self._matrix = np.identity(4, dtype=float)

        self._matrix_view = self._matrix.view()
        self._matrix_view.flags.writeable = False

        self._wrapper: RecursiveTransform = None

    def flag_update(self):
        """Signal that this transform has updated."""
        # note: this function has been heavily micro-optimized
        # please don't modify it carelessly
        last_modified = perf_counter_ns()
        self.last_modified = last_modified
        if wrapper := self._wrapper:
            wrapper.flag_update(last_modified)

    def _set_wrapper(self, wrapper: RecursiveTransform):
        self._wrapper = wrapper

    @property
    def reference_up(self) -> np.ndarray:
        """The zero-tilt reference vector used for the direction setters."""
        if self._wrapper:
            return self._wrapper._parent_reference_up
        return super().reference_up

    @reference_up.setter
    def reference_up(self, value):
        if self._wrapper:
            self._wrapper._parent_reference_up = value
            return
        AffineBase.reference_up.fset(self, value)

    @property
    def state_basis(self) -> str:
        """The basis of the transform, either "components" (default) or "matrix"."""
        return self._state_basis

    @state_basis.setter
    def state_basis(self, value: str):
        if value not in ("components", "matrix"):
            raise ValueError("state_basis must be either 'components' or 'matrix'")
        if value == "matrix":
            self._matrix[:] = self.matrix
        elif value == "components":
            self._position[:] = self.position
            self._rotation[:] = self.rotation
            self._scale[:] = self.scale
        self._state_basis = value
        self.flag_update()

    @property
    def components(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.state_basis == "components":
            return (self._position_view, self._rotation_view, self._scale_view)
        return super().components

    @components.setter
    def components(self, value: Tuple[np.ndarray, np.ndarray, np.ndarray]):
        if self.state_basis == "components":
            if (pos := value[0]) is not None:
                self._position[:] = pos
            if (rot := value[1]) is not None:
                self._rotation[:] = rot
            if (sca := value[2]) is not None:
                self._scale[:] = sca
        else:
            la.mat_compose(*value, out=self._matrix)
            np.sign(value[2], out=self._scaling_signs)
        self.flag_update()

    @property
    def position(self) -> np.ndarray:
        """The origin of source."""
        if self.state_basis == "components":
            return self._position_view
        return super().position

    @position.setter
    def position(self, value):
        if self.state_basis == "components":
            self._position[:] = value
        else:
            la.mat_compose(value, self.rotation, self.scale, out=self._matrix)
        self.flag_update()

    @property
    def rotation(self) -> np.ndarray:
        """The orientation of source as a quaternion."""
        if self.state_basis == "components":
            return self._rotation_view
        return super().rotation

    @rotation.setter
    def rotation(self, value):
        if self.state_basis == "components":
            self._rotation[:] = value
        else:
            la.mat_compose(self.position, value, self.scale, out=self._matrix)
        self.flag_update()

    @property
    def scale(self) -> np.ndarray:
        """The scale of source."""
        if self.state_basis == "components":
            return self._scale_view
        return super().scale

    @scale.setter
    def scale(self, value):
        if self.state_basis == "components":
            self._scale[:] = value
        else:
            la.mat_compose(self.position, self.rotation, value, out=self._matrix)
            np.sign(value, out=self._scaling_signs)
        self.flag_update()

    @cached
    def _computed_scaling_signs(self) -> np.ndarray:
        signs = np.sign(self._scale)
        signs.flags.writeable = False
        return signs

    @property
    def scaling_signs(self) -> np.ndarray:
        """Property used to track and preserve the scale factor signs
        over matrix decomposition operations."""
        if self.state_basis == "components":
            return self._computed_scaling_signs
        return super().scaling_signs

    @cached
    def _rotation_matrix(self) -> np.ndarray:
        rotation = la.mat_from_quat(self._rotation)
        rotation.flags.writeable = False
        return rotation

    @property
    def rotation_matrix(self) -> np.ndarray:
        """The orientation of source as a rotation matrix."""
        if self.state_basis == "components":
            return self._rotation_matrix
        return super().rotation_matrix

    rotation_matrix = rotation_matrix.setter(AffineBase.rotation_matrix.fset)

    @cached
    def _euler(self) -> np.ndarray:
        euler = la.quat_to_euler(self._rotation)
        euler.flags.writeable = False
        return euler

    @property
    def euler(self) -> np.ndarray:
        """The orientation of source as XYZ euler angles."""
        if self.state_basis == "components":
            return self._euler
        return super().euler

    euler = euler.setter(AffineBase.euler.fset)

    @cached
    def _composed_matrix(self) -> np.ndarray:
        mat = la.mat_compose(self._position, self._rotation, self._scale)
        mat.flags.writeable = False
        return mat

    @property
    def matrix(self) -> np.ndarray:
        """Affine matrix describing this transform.

        ``vec_target = matrix @ vec_source``.

        """
        if self.state_basis == "components":
            return self._composed_matrix
        return self._matrix_view

    @matrix.setter
    def matrix(self, value: np.ndarray):
        if self.state_basis == "components":
            try:
                # when the matrix is set manually
                # try to maintain the most recent configured scaling signs
                position, rotation, scale = la.mat_decompose(
                    value, scaling_signs=self.scaling_signs
                )
            except ValueError:
                position, rotation, scale = la.mat_decompose(value)
            self._position[:] = position
            self._rotation[:] = rotation
            self._scale[:] = scale
        elif self.state_basis == "matrix":
            self._matrix[:] = value
        self.flag_update()

    def __matmul__(self, other) -> Union[AffineTransform, np.ndarray]:
        if isinstance(other, AffineTransform):
            matrix = self.matrix @ other.matrix

            state_basis = self.state_basis
            if la.mat_has_shear(matrix):
                # if the resulting transform has shearing
                # force matrix state_basis - we don't
                # support shearing in components, see #920
                state_basis = "matrix"

            kwargs = dict(
                is_camera_space=self.is_camera_space,
                state_basis=state_basis,
                matrix=matrix,
            )
            if state_basis == "components":
                try:
                    decomposed = la.mat_decompose(
                        matrix,
                        scaling_signs=self.scaling_signs * other.scaling_signs,
                    )
                except ValueError:
                    decomposed = la.mat_decompose(matrix)
                kwargs["position"] = decomposed[0]
                kwargs["rotation"] = decomposed[1]
                kwargs["scale"] = decomposed[2]

            return AffineTransform(**kwargs)

        return np.asarray(self) @ other


class RecursiveTransform(AffineBase):
    """A transform that may be preceded by another transform.

    This transform behaves semantically identical to an ordinary
    ``AffineTransform`` (same properties), except that users may define a
    ``parent`` transform which precedes the ``matrix`` used by the ordinary
    ``AffineTransform``. The resulting ``RecursiveTransform`` then controls the
    total transform that results from combining the two transforms via::

        recursive_transform = parent @ matrix

    In other words, the source frame of ``RecursiveTransform`` is the source
    frame of ``matrix`` and the target frame of ``RecursiveTransform`` is the
    target frame of ``parent``. Implying that ``RecursiveTransform``'s
    properties are given in the target frame.

    The use case for this class is to allow getting and setting of properties of
    a WorldObjects world transform, i.e., it implements ``WorldObject.world``.

    Under the hood, this transform wraps another transform (passed in as
    ``base``), similar to how ``AffineTransform`` wraps a numpy array.
    Setting properties of ``RecursiveTransform`` will internally
    transform the new values into ``matrix`` target frame and then set the
    obtained value on the wrapped transform. This means that the ``parent``
    transform is not affected by changes made to this transform.

    Parameters
    ----------
    base : AffineTransform
        The base transform that will be wrapped by this transform.
    parent : RecursiveTransform, optional
        The parent transform that precedes the base transform.
    reference_up : ndarray, [3]
        The direction of the reference_up vector expressed in the target
        frame. It is used by the axis properties (right, up, forward)
        to maintain a common level of rotation around an axis when it
        is updated by it's setter. By default, it points along the
        positive Y-axis.
    is_camera_space : bool
        If True, the transform represents a camera space which means that it's
        ``forward`` and ``right`` directions are inverted.

    See Also
    --------
    AffineBase
        Base class defining various useful properties for this transform.

    """

    __slots__ = (
        "_parent",
        "cache__computed_parent_reference_up_cache",
        "cache__computed_scaling_signs_cache",
        "cache__matrix_cache",
        "children",
        "last_modified",
        "own",
    )

    def __init__(
        self,
        base: AffineTransform,
        /,
        *,
        parent: RecursiveTransform = None,
        reference_up=(0, 1, 0),
        is_camera_space=False,
    ) -> None:
        super().__init__(reference_up=reference_up, is_camera_space=is_camera_space)
        self.last_modified = perf_counter_ns()
        self._parent = parent
        self.children = []
        self.own = base
        self.own._set_wrapper(self)

    def flag_update(self, last_modified=None):
        """Signal that this transform has updated."""
        # note: this function has been heavily micro-optimized
        # please don't modify it carelessly
        if last_modified is None:
            last_modified = perf_counter_ns()
        self.last_modified = last_modified
        if children := self.children:
            for child in children:
                if child.children:
                    child.flag_update(last_modified)
                else:
                    child.last_modified = last_modified

    @cached
    def _computed_parent_reference_up(self) -> np.ndarray:
        """The direction of the reference_up vector expressed in the parent frame."""
        new_ref = la.vec_transform(self._reference_up, self._parent.inverse_matrix)
        origin = la.vec_transform((0, 0, 0), self._parent.inverse_matrix)
        vec = la.vec_normalize(new_ref - origin)
        vec.flags.writeable = False
        return vec

    @property
    def _parent_reference_up(self) -> np.ndarray:
        if self._parent:
            return self._computed_parent_reference_up
        return self._reference_up_view

    @_parent_reference_up.setter
    def _parent_reference_up(self, value):
        if self._parent:
            new_ref = la.vec_transform(value, self._parent.matrix)
            origin = self._parent.position
            la.vec_normalize(new_ref - origin, out=self._reference_up)
        else:
            la.vec_normalize(value, out=self._reference_up)
        self.flag_update()
        # Note: since _parent_reference_up is only used in a setter in AffineBase
        # we do not need to call flag_update() on self.own; all its state and cache
        # remains intact

    @property
    def parent(self) -> RecursiveTransform:
        """The transform that preceeds the own/local transform."""
        return self._parent

    @parent.setter
    def parent(self, value: RecursiveTransform):
        self._parent = value
        self.flag_update()

    @cached
    def _matrix(self) -> np.ndarray:
        mat = self._parent.matrix @ self.own.matrix
        mat.flags.writeable = False
        return mat

    @property
    def matrix(self) -> np.ndarray:
        """Affine matrix describing this transform.

        ``vec_target = matrix @ vec_source``.

        """
        if self._parent:
            return self._matrix
        return self.own.matrix

    @matrix.setter
    def matrix(self, value: np.ndarray):
        if self._parent:
            if self.own.state_basis == "matrix":
                np.matmul(self._parent.inverse_matrix, value, out=self.own._matrix)
                self.own.flag_update()
            else:
                self.own.matrix = self._parent.inverse_matrix @ value
        else:
            self.own.matrix = value

    def __matmul__(self, other) -> Union[RecursiveTransform, np.ndarray]:
        if isinstance(other, AffineBase):
            return RecursiveTransform(other, parent=self)
        else:
            return np.asarray(self) @ other

    @cached
    def _computed_scaling_signs(self) -> np.ndarray:
        signs = self._parent.scaling_signs * self.own.scaling_signs
        signs.flags.writeable = False
        return signs

    @property
    def scaling_signs(self) -> np.ndarray:
        """Property used to track and preserve the scale factor signs
        over matrix decomposition operations."""
        if self._parent:
            return self._computed_scaling_signs
        return self.own.scaling_signs
