from math import tan, pi

import numpy as np
import pylinalg as la

from ._base import Camera
from ..objects._base import WorldObject
from ..linalg import Matrix4


class GenericCamera(Camera):
    """
    A generic controllable camera that can provide a perspective or orthographic view.

    Parameters
    ----------
    fov: float
        The field of view as an angle in degrees. Higher values give a
        wide-angle lens effect. This value is limited between 0 and
        179. If zero, it behaves like an orthographic camera.
    aspect : float
        The desired aspect ratio, which is used to determine the vision pyramid's
        boundaries depending on the viewport size. Common values are 16/9 or 4/3. Default 1.
    extent : float
        A measure for the size of the scene that the camera is
        observing. This is also set by `show_object()`. When the fov
        is zero, it defines the view frustrum. It is also used to set
        the near and far clipping planes, and controllers use it to
        determine what is being looked at.
    """

    _maintain_aspect = True
    _fov_range = 0, 179

    def __init__(self, fov, aspect=1, extent=1):
        super().__init__()

        self._width = 1
        self._height = 1
        self.fov = fov
        self.aspect = aspect
        self.extent = extent
        self.zoom = 1

        self.set_view_size(1, 1)
        self.update_projection_matrix()

    def __repr__(self) -> str:
        return f"GenericCamera({self.fov}, {self.aspect}, {self.extent})"

    @property
    def fov(self):
        """The field of view (in degrees), between 0-179."""
        return self._fov

    @fov.setter
    def fov(self, value):
        fov = float(value)
        fov = min(max(fov, self._fov_range[0]), self._fov_range[1])
        # Don't allow values between 0 and 1, as it becomes numerically unstable
        # For the record, zero is allowed, meaning orthographic projection
        if 0 < fov < 1:
            fov = 1
        self._fov = fov

    @property
    def width(self):
        """The (minimum) width of the view-cube.
        Together with the `height`, this also defines the aspect and extent.
        """
        return self._width

    @width.setter
    def width(self, value):
        self._width = float(value)

    @property
    def height(self):
        """The (minimum) height of the view-cube.
        Together with the `width`, this also defines the aspect and extent.
        """
        return self._height

    @height.setter
    def height(self, value):
        self._height = float(value)

    @property
    def aspect(self):
        """The aspect ratio. (The ratio between width and height.)"""
        return self._width / self._height

    @aspect.setter
    def aspect(self, value):
        aspect = float(value)
        if aspect <= 0:
            raise ValueError("aspect must be > 0")
        extent = self.extent
        self._height = 2 * extent / (1 + aspect)
        self._width = self._height * aspect

    @property
    def extent(self):
        """A measure of the size of the scene that the camera observing.
        This is also set by `show_object()`. (The mean of width and height.)
        """
        return 0.5 * (self._width + self._height)

    @extent.setter
    def extent(self, value):
        extent = float(value)
        if extent <= 0:
            raise ValueError("extend must be > 0")
        aspect = self.aspect
        self._height = 2 * extent / (1 + aspect)
        self._width = self._height * aspect

    @property
    def zoom(self):
        """The camera zoom level."""
        return self._zoom

    @zoom.setter
    def zoom(self, value):
        self._zoom = float(value)

    @property
    def maintain_aspect(self):
        """Whether the aspect ration is maintained as the window size
        changes. Default True. Note that it only make sense to set this
        to False in combination with a panzoom controller.
        """
        return self._maintain_aspect

    @maintain_aspect.setter
    def maintain_aspect(self, value):
        self._maintain_aspect = bool(value)

    def get_state(self):
        return {
            "position": tuple(self.position.to_array()),
            "rotation": tuple(self.rotation.to_array()),
            "scale": tuple(self.scale.to_array()),
            "up": tuple(self.up.to_array()),
            "fov": self.fov,
            "width": self.width,
            "height": self.height,
            "zoom": self.zoom,
            "maintain_aspect": self.maintain_aspect,
        }

    def set_state(self, state):
        # Set the more complex props
        if "position" in state:
            self.position.set(*state["position"])
        if "rotation" in state:
            self.rotation.set(*state["rotation"])
        if "scale" in state:
            self.scale.set(*state["scale"])
        if "up" in state:
            self.up.set(*state["up"])
        # Set simple props
        for key in ("fov", "width", "height", "zoom", "maintain_aspect"):
            if key in state:
                setattr(self, key, state[key])

    def set_view_size(self, width, height):
        self._view_aspect = width / height

    def _get_near_and_far_plane(self):
        if self.fov > 0:
            d = self.extent * 45 / self.fov
            return d / 1000, 1000 * d
        else:
            d = self.extent
            return -500 * d, 500 * d

    def update_projection_matrix(self):
        if self.fov > 0:
            # Get the reference width / height
            near, far = self._get_near_and_far_plane()
            size = 2 * near * tan(pi / 180 * 0.5 * self.fov) / self.zoom
            # Pre-apply the reference aspect ratio
            width = size * self.aspect**0.5
            height = size / self.aspect**0.5
            # Increase eihter the width or height, depending on the view size
            if not self._maintain_aspect:
                pass
            elif self.aspect < self._view_aspect:
                width *= self._view_aspect / self.aspect
            else:
                height *= self.aspect / self._view_aspect
            # Calculate bounds
            top = +0.5 * height
            bottom = -0.5 * height
            left = -0.5 * width
            right = +0.5 * width
            # Set matrices
            # The linalg perspective projection puts xyz in the range -1..1,
            # but in the coordinate system of wgpu (and this lib) the depth
            # is expressed in 0..1, so we also correct for that.
            self.projection_matrix.make_perspective(left, right, top, bottom, near, far)
            self.projection_matrix.premultiply(
                Matrix4(1, 0, 0.0, 0, 0, 1, 0.0, 0, 0.0, 0.0, 0.5, 0.0, 0, 0, 0.5, 1)
            )
            self.projection_matrix_inverse.get_inverse(self.projection_matrix)

        else:
            # The reference view plane is scaled with the zoom factor
            width = self._width / self.zoom
            height = self._height / self.zoom
            # Increase either the width or height, depending on the viewport shape
            aspect = width / height
            if not self._maintain_aspect:
                pass
            elif aspect < self._view_aspect:
                width *= self._view_aspect / aspect
            else:
                height *= aspect / self._view_aspect

            bottom = -0.5 * height
            top = +0.5 * height
            left = -0.5 * width
            right = +0.5 * width
            near, far = self._get_near_and_far_plane()
            # Set matrices
            # The linalg ortho projection puts xyz in the range -1..1, but
            # in the coordinate system of wgpu (and this lib) the depth is
            # expressed in 0..1, so we also correct for that.
            self.projection_matrix.make_orthographic(
                left, right, top, bottom, near, far
            )
            self.projection_matrix.premultiply(
                Matrix4(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0.5, 0, 0, 0, 0.5, 1)
            )
            self.projection_matrix_inverse.get_inverse(self.projection_matrix)

    def show_object(
        self, target: WorldObject, view_dir=(-1, -1, -1), *, up=None, size_weight=2
    ):
        """Position the camera such that the given world object in is in view.

        Parameters
        ----------
        target: WorldObject
            The object to look at.
        view_dir: 3-tuple of float or Vector3
            Look at the object from this direction.
        up: 3-tuple of float or Vector3
            Also set the up vector.
        size_weight: float
            How much extra space the camera must show.
            The target's bounding sphere radius is multiplied by this weight.
            Default 2. If you know your data is square and you look at it frontally,
            you can set it to 1.5.

        Returns:
            pos: Vector3
                The world coordinate the camera is looking at.
        """

        view_dir = tuple(view_dir)
        if len(view_dir) == 1:
            raise TypeError(f"Expected view_dir to be tuple, not {view_dir[0]}")
        elif len(view_dir) != 3:
            raise ValueError("Expected view_dir to be a tuple of 3 floats.")

        if isinstance(target, WorldObject):
            bsphere = target.get_world_bounding_sphere()
            if bsphere is None:
                pos = target.get_world_position().to_array()
                bsphere = tuple(pos) + (1,)
        elif isinstance(target, (tuple, list, np.ndarray)) and len(target) == 4:
            bsphere = tuple(target)
        else:
            raise TypeError(
                "show_object target must be a world object, or a (x, y, z, radius) tuple."
            )

        view_pos = bsphere[:3]
        radius = bsphere[3]
        extent = radius * size_weight

        fov = getattr(self, "fov", None)

        if fov:
            fov_rad = fov * pi / 180
            distance = 0.5 * extent / tan(0.5 * fov_rad)
        else:
            distance = extent * 1.0

        camera_pos = view_pos - la.vector_normalize(view_dir) * distance

        if up is not None:
            self.up = up

        self.position.set(*camera_pos)
        self.look_at(view_pos)
        self.extent = extent

        return view_pos
