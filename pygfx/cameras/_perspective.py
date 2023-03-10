from math import tan, pi

import numpy as np
import pylinalg as la

from ._base import Camera
from ..objects._base import WorldObject


class PerspectiveCamera(Camera):
    """A generic 3D camera with a configurable field of view (fov).

    Parameters
    ----------
    fov: float
        The field of view as an angle in degrees. Higher values give a
        wide-angle lens effect. This value is limited between 0 and
        179. If zero, it operates in orthographic mode.
    aspect: float
        The desired aspect ratio, which is used to determine the vision pyramid's
        boundaries depending on the viewport size. Common values are 16/9 or 4/3. Default 1.
    extent: float
        A measure for the size of the scene that the camera is
        observing. This is also set by `show_object()`. When the fov
        is zero, it defines the view frustrum. It is also used to set
        the near and far clipping planes, and controllers use it to
        determine what is being looked at.
    zoom: float
        An additional zoom factor, equivalent to attaching a zoom lens.
    maintain_aspect: bool
        Whether the aspect ration is maintained as the window size changes.
        Default True. If false, the dimensions are stretched to fit the window.
    depth_range: 2-tuple
        The values for the near and far clipping planes. If not given
        or None, the clip planes will be calculated automatically based
        on the fov and extent.
    """

    _fov_range = 0, 179

    def __init__(
        self, fov, aspect=1, extent=1, *, zoom=1, maintain_aspect=True, depth_range=None
    ):
        super().__init__()

        self.fov = fov
        self._width = 1
        self._height = 1
        self.aspect = aspect
        self.extent = extent
        self.zoom = zoom
        self.maintain_aspect = maintain_aspect
        self.depth_range = depth_range

        self.set_view_size(1, 1)
        self.update_projection_matrix()

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

    @property
    def depth_range(self):
        """The values for the near and far clip planes. If None, these values
        are calculated from fov and extent.
        """
        return self._depth_range

    @depth_range.setter
    def depth_range(self, value):
        if value is None:
            self._depth_range = None
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            self._depth_range = float(value[0]), float(value[1])
        else:
            raise TypeError("depth_range must be None or a 2-tuple.")

    def _get_near_and_far_plane(self):
        if self._depth_range:
            return self._depth_range
        elif self.fov > 0:
            # Take the distance that the camera is likely from the objects being viewed
            # Scale with a factor 1000
            extent = self.extent
            d = distance_from_fov_and_extent(self.fov, extent)
            return d / 1000, d + 1000 * extent
        else:
            d = self.extent
            return -449 * d, 501 * d

    @property
    def near(self):
        """The location of the near clip plane.
        Use `depth_range` so overload the computed value, if necessary.
        """
        near, far = self._get_near_and_far_plane()
        return near

    @property
    def far(self):
        """The location of the far clip plane.
        Use `depth_range` so overload the computed value, if necessary.
        """
        near, far = self._get_near_and_far_plane()
        return far

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
            "depth_range": self.depth_range,
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
        for key in ("fov", "width", "height", "zoom", "maintain_aspect", "depth_range"):
            if key in state:
                setattr(self, key, state[key])

    def set_view_size(self, width, height):
        self._view_aspect = width / height

    def update_projection_matrix(self):
        if self.fov > 0:
            # Get the reference width / height
            near, far = self._get_near_and_far_plane()
            size = 2 * near * tan(pi / 180 * 0.5 * self.fov) / self.zoom
            # Pre-apply the reference aspect ratio
            height = 2 * size / (1 + self.aspect)
            width = height * self.aspect
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
            proj = la.matrix_make_perspective(
                left, right, top, bottom, near, far, depth_range=(0, 1)
            )
            proj_i = np.linalg.inv(proj)
            self.projection_matrix.set(*proj.flat)
            self.projection_matrix_inverse.set(*proj_i.flat)

        else:
            # The reference view plane is scaled with the zoom factor
            width = self.width / self.zoom
            height = self.height / self.zoom
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
            proj = la.matrix_make_orthographic(
                left, right, top, bottom, near, far, depth_range=(0, 1)
            )
            proj_i = np.linalg.inv(proj)
            self.projection_matrix.set(*proj.flat)
            self.projection_matrix_inverse.set(*proj_i.flat)

    def look_at(self, target, *, up=None):
        """Look at the given position or object, without changing the camera's position.

        Parameters
        ----------
        target: WorldObject or a position (x, y, z)
            The target to point the camera towards.
        up: 3-tuple
            If given, also sets the up vector of the camera.
        """

        # Get pos from target
        if isinstance(target, WorldObject):
            pos = target.position.to_array()
        elif isinstance(target, (tuple, list, np.ndarray)) and len(target) in (3, 4):
            pos = tuple(target)[:3]
        elif hasattr(target, "to_array"):
            pos = target.to_array()
        else:
            raise TypeError(
                "look_at target must be a WorldObject, or a (x, y, z) tuple."
            )

        # Look at the provided position, taking up into account
        if up is not None:
            self.up = up
        super().look_at(pos)

        # Also set the extent. This way, a user can position the camera,
        # use look_at to point it at the center of the scene, attach a controller,
        # and everything works! Oh, and the near and far plane are also set this way.
        distance = la.vector_distance_between(pos, self.position.to_array())
        self.extent = extent_from_fov_and_distance(self.fov, distance)

    def show_object(
        self, target: WorldObject, view_dir=None, *, up=None, size_weight=2
    ):
        """Position and orient the camera such that the given WorldObject in is in view.

        This method is mainly intended for viewing 3D data. For 2D data
        it is not uncommon that the margins may feel somewhat large.

        Parameters
        ----------
        target: WorldObject or a sphere (x, y, z, r)
            The object to look at.
        view_dir: 3-tuple of float
            Look at the object from this direction. If not given or None,
            uses the current view direction.
        up: 3-tuple
            Sets the up vector of the camera. If not given or None, the
            up property is not changed.
        size_weight: float
            How much extra space the camera must show.
            The target's bounding sphere radius is multiplied by this weight.
            Default 2. If you know your data is square and you look at it frontally,
            you can set it to 1.5.

        """

        # Get bounding sphere from target
        if isinstance(target, WorldObject):
            bsphere = target.get_world_bounding_sphere()
            if bsphere is None:
                raise ValueError(
                    "Given target does not have a bounding sphere, you should probably just provide a sphere (x, y, z, r) yourself."
                )
        elif isinstance(target, (tuple, list, np.ndarray)) and len(target) == 4:
            bsphere = tuple(target)
        else:
            raise TypeError(
                "show_object target must be a WorldObject, or a (x, y, z, radius) tuple."
            )

        # Obtain view direction
        if view_dir is None:
            rotation = self.rotation.to_array()
            view_dir = la.quaternion_rotate((0, 0, -1), rotation)
        elif isinstance(view_dir, (tuple, list, np.ndarray)) and len(view_dir) == 3:
            view_dir = tuple(view_dir)
        else:
            raise TypeError(f"Expected view_dir to be sequence, not {view_dir}")
        view_dir = la.vector_normalize(view_dir)

        # Do the math ...

        view_pos = bsphere[:3]
        radius = bsphere[3]
        extent = radius * size_weight
        distance = distance_from_fov_and_extent(self.fov, extent)

        camera_pos = view_pos - view_dir * distance

        self.position.set(*camera_pos)
        self.look_at(view_pos, up=up)
        self.extent = extent

    def show_rect(self, left, right, top, bottom, *, view_dir=None, up=None):
        """Position and orient the camera such that the given rectangle in is in view.

        The rectangle represents a plane in world coordinates, centered
        at the origin of the world, and rotated to be orthogonal to the
        view_dir.

        This method is mainly intended for viewing 2D data, especially
        when `maintain_aspect` is set to False, and is convenient
        for setting the initial view before attaching a PanZoomController.

        Parameters
        ----------
        left: float
            The left boundary of the plane to show.
        right: float
            The right boundary of the plane to show.
        top: float
             The top boundary of the plane to show.
        bottom: float
             The bottom boundary of the plane to show.
        view_dir: 3-tuple of float
            Look at the rectang;e from this direction. If not given or None,
            uses the current view direction.
        up: 3-tuple
            Sets the up vector of the camera. If not given or None, the
            up property is not changed.

        """

        # Obtain view direction
        if view_dir is None:
            rotation = self.rotation.to_array()
            view_dir = la.quaternion_rotate((0, 0, -1), rotation)
        elif isinstance(view_dir, (tuple, list, np.ndarray)) and len(view_dir) == 3:
            view_dir = tuple(view_dir)
        else:
            raise TypeError(f"Expected view_dir to be sequence, not {view_dir}")
        view_dir = la.vector_normalize(view_dir)

        # Set bounds (note that this implicitly sets extent and aspect)
        self.width = right - left
        self.height = bottom - top
        # extent = (self.width**2 + self.height ** 2)**0.5
        # First move so we view towards the origin with the correct vector
        distance = distance_from_fov_and_extent(self.fov, self.extent)
        camera_pos = (0, 0, 0) - view_dir * distance
        self.position.set(*camera_pos)
        self.look_at((0, 0, 0), up=up)

        # Now we have a rotation that we can use to orient our rect
        position = self.position.to_array()
        rotation = self.rotation.to_array()

        offset = 0.5 * (left + right), 0.5 * (top + bottom), 0
        new_position = position + la.quaternion_rotate(offset, rotation)
        self.position.set(*new_position)


def fov_distance_factor(fov):
    if fov > 0:
        fov_rad = fov * pi / 180
        return 0.5 / tan(0.5 * fov_rad)
    else:
        return 1.0


def distance_from_fov_and_extent(fov, extent):
    # It's important that controller and camera use the same distance calculations,
    return extent * fov_distance_factor(fov)


def extent_from_fov_and_distance(fov, distance):
    return distance / fov_distance_factor(fov)
