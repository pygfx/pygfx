from math import pi, tan

import numpy as np
import pylinalg as la

from ..objects._base import WorldObject
from ..utils.transform import cached
from ._base import Camera


class PerspectiveCamera(Camera):
    """A generic 3D camera with a configurable field of view (fov).

    Parameters
    ----------
    fov: float
        The field of view as an angle in degrees. Higher values give a
        wide-angle lens effect. This value is limited between 0 and
        179. If zero, it operates in orthographic mode. Default 50.
    aspect: float
        The desired aspect ratio (width divided by height), which determines the
        vision pyramid's boundaries. Default 1.
    width: float
        The width of the scene to view. If omitted or None, the width
        is derived from aspect and height.
    height: float
        The height of the scene to view. If omitted or None, the height
        is derived from aspect and width.
    zoom: float
        An additional zoom factor, equivalent to attaching a zoom lens.
    maintain_aspect: bool
        Whether the aspect ration is maintained as the window size changes.
        Default True. If false, the dimensions are stretched to fit the window.
    depth_range: 2-tuple
        The values for the near and far clipping planes. If not given
        or None, the clip planes will be calculated automatically based
        on the fov, width, and height.

    Note
    ----
    The width and/or height should be set when using a fov of zero,
    when you want to manipulate the camera with a controller, or when
    you want to make use of the automatic depth_range. However, if you
    also call ``show_pos``, ``show_object``, or ``show_rect`` you can omit
    width and height, because these methods set them for you.

    """

    _fov_range = 0, 179

    def __init__(
        self,
        fov=50,
        aspect=1,
        *,
        width=None,
        height=None,
        zoom=1,
        maintain_aspect=True,
        depth_range=None,
    ):
        super().__init__()

        self.fov = fov

        # Set width and height. Note that if both width and height are given, it overrides aspect
        aspect = aspect or 1
        if width is None and height is None:
            height = 100  # 100 produces safer near plane for most use-cases than 1
            width = height * aspect
        elif width is None:
            width = height * aspect
        elif height is None:
            height = width / aspect
        self.width = width
        self.height = height

        self.zoom = zoom
        self.maintain_aspect = maintain_aspect
        self.depth_range = depth_range

        self.set_view_size(1, 1)

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
        self.flag_update()

    @property
    def width(self):
        """The (minimum) width of the view-cube.
        Changing the width changes the aspect, but not the height.
        """
        return self._width

    @width.setter
    def width(self, value):
        self._width = float(value)
        self.flag_update()

    @property
    def height(self):
        """The (minimum) height of the view-cube.
        Changing the height changes the aspect, but not the width.
        """
        return self._height

    @height.setter
    def height(self, value):
        self._height = float(value)
        self.flag_update()

    @property
    def aspect(self):
        """The aspect ratio (width divided by height).

        The aspect determines the vision pyramid's boundaries. It can help fit
        the scene better to the window, and keep it fitting as the window is
        resized.

        Setting the aspect updates width and height such that their mean is
        unchanged.

        Note that ``show_object(match_aspect=True)`` also sets the aspect.
        """
        return self._width / self._height

    @aspect.setter
    def aspect(self, value):
        aspect = float(value)
        if aspect <= 0:
            raise ValueError("aspect must be > 0")
        extent = 0.5 * (self._width + self._height)
        self._height = 2 * extent / (1 + aspect)
        self._width = self._height * aspect
        self.flag_update()

    def _set_extent(self, extent):
        """Set the mean of width and height while maintaining aspect."""
        extent = float(extent)
        if extent <= 0:
            raise ValueError("extent must be > 0")
        aspect = self.aspect
        self._height = 2 * extent / (1 + aspect)
        self._width = self._height * aspect
        self.flag_update()

    @property
    def zoom(self):
        """The camera zoom level."""
        return self._zoom

    @zoom.setter
    def zoom(self, value):
        self._zoom = float(value)
        self.flag_update()

    @property
    def maintain_aspect(self):
        """Whether the aspect ration is maintained as the window size changes.

        Default True. Note that it only make sense to set this to False in
        combination with a panzoom controller.
        """
        return self._maintain_aspect

    @maintain_aspect.setter
    def maintain_aspect(self, value):
        self._maintain_aspect = bool(value)
        self.flag_update()

    @property
    def depth_range(self):
        """The values for the near and far clip planes. If None, these values
        are calculated from fov, width, amd heiht.
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
        self.flag_update()

    def _get_near_and_far_plane(self):
        # Dept range explicitly given?
        if self._depth_range:
            return self._depth_range

        # Put 1000 units between the near and far plane, scaled by extent
        extent = 0.5 * (self._width + self._height)
        if self.fov > 0:
            # Scale near plane with the fov to compensate for the fact
            # that with very small fov you're probably looking at something
            # in the far distance.
            f = fov_distance_factor(self.fov)
            return (extent * f) / 1000, 1000 * extent
        else:
            return -500 * extent, 500 * extent

    @property
    def near(self) -> float:
        """The location of the near clip plane.
        Use `depth_range` so overload the computed value, if necessary.
        """
        near, _ = self._get_near_and_far_plane()
        return near

    @property
    def far(self) -> float:
        """The location of the far clip plane.
        Use `depth_range` so overload the computed value, if necessary.
        """
        _, far = self._get_near_and_far_plane()
        return far

    def get_state(self):
        """Get the state of the camera as a dict.

        The fields contain "position", "rotation", "scale", and
        "reference_up", representing the camera's transform. The scale
        is typically not used, but included for completeness. Further,
        the following properties are included: "fov", "width", "height",
        "zoom", "maintain_aspect", and "depth_range".

        """
        return {
            "position": self.local.position,
            "rotation": self.local.rotation,
            "scale": self.local.scale,
            "reference_up": self.world.reference_up,
            "fov": self.fov,
            "width": self.width,
            "height": self.height,
            "zoom": self.zoom,
            "maintain_aspect": self.maintain_aspect,
            "depth_range": self.depth_range,
        }

    def set_state(self, state):
        """Set the state of the camera from a dict.

        Accepted fields are the same as in ``get_state()``. In addition,
        the fields ``x``, ``y``, and ``z`` are also accepted to set the
        position along a singular dimension.

        """
        # Set the more complex props
        for key, value in state.items():
            if key == "position":
                self.local.position = value
            if key in ("x", "y", "z"):
                setattr(self.local, key, value)
            elif key == "scale":
                self.local.scale = value
            elif key == "rotation":
                self.local.rotation = value
            elif key == "reference_up":
                self.world.reference_up = value
            elif key in (
                "fov",
                "width",
                "height",
                "zoom",
                "maintain_aspect",
                "depth_range",
            ):
                # Simple props
                setattr(self, key, value)
        self.flag_update()

    def _update_projection_matrix(self):
        zoom_factor = self._zoom
        near, far = self._get_near_and_far_plane()

        view_aspect = self._view_size[0] / self._view_size[1]

        if self._view_offset is not None:
            # The view_offset should override the aspect, via its full (virtual) size
            view_aspect = (
                self._view_offset["full_width"] / self._view_offset["full_height"]
            )

        if self.fov > 0:
            # Get the reference width / height
            size = 2 * near * tan(pi / 180 * 0.5 * self.fov) / zoom_factor
            # Pre-apply the reference aspect ratio
            height = 2 * size / (1 + self.aspect)
            width = height * self.aspect
            # Increase either the width or height, depending on the view size
            if not self._maintain_aspect:
                pass
            elif self.aspect < view_aspect:
                width *= view_aspect / self.aspect
            else:
                height *= self.aspect / view_aspect
            # Calculate bounds
            top = +0.5 * height
            bottom = -0.5 * height
            left = -0.5 * width
            right = +0.5 * width
            # Set matrices
            projection_matrix = la.mat_perspective(
                left, right, top, bottom, near, far, depth_range=(0, 1)
            )

        else:
            # The reference view plane is scaled with the zoom factor
            width = self.width / zoom_factor
            height = self.height / zoom_factor
            # Increase either the width or height, depending on the viewport shape
            aspect = width / height
            if not self._maintain_aspect:
                pass
            elif aspect < view_aspect:
                width *= view_aspect / aspect
            else:
                height *= aspect / view_aspect
            # Calculate bounds
            bottom = -0.5 * height
            top = +0.5 * height
            left = -0.5 * width
            right = +0.5 * width
            # Set matrices
            projection_matrix = la.mat_orthographic(
                left, right, top, bottom, near, far, depth_range=(0, 1)
            )

        projection_matrix.flags.writeable = False
        return projection_matrix

    def show_pos(self, target, *, up=None):
        """Look at the given position or object.

        This is similar to `look_at()` but it also sets the width and height
        (while honoring aspect). The camera's position is not changed, but it's
        'reference point' does. E.g. with an orbit controller, it orbits around
        the given target.

        Parameters
        ----------
        target: WorldObject or (x, y, z)
            The target to point the camera towards.
        up: 3-tuple
            If given, set ``camera.world.reference_up`` to the given value.

        """

        # Get pos from target
        if isinstance(target, WorldObject):
            pos = target.local.position
        else:
            pos = np.asarray(target)

        if pos.shape != (3,):
            raise ValueError("Expected position to have 3 values.")

        # Look at the provided position, taking up into account
        if up is not None:
            self.world.reference_up = up
        self.look_at(pos)

        # Update extent
        distance = la.vec_dist(pos, self.local.position)
        self._set_extent(distance / fov_distance_factor(self.fov))

    def show_object(
        self,
        target: WorldObject,
        view_dir=None,
        *,
        up=None,
        scale=1,
        match_aspect=False,
    ):
        """Position and orientate the camera such that the given target in is in view.

        Sets the position and rotation of the camera, and adjusts
        its width and height to the target's size (while honoring aspect).

        The fit is such that as the object is rotated (or as the camera rotates around the object)
        the object still fits inside the viewport, which is usually good for 3D scenes.

        When setting ``match_aspect``, the bounding box of the object is matched
        to the screen, and the camera's ``aspect`` adjusted accordingly. This
        gives a tight fit that is usually preferred for 2D scenes (with the
        panzoom-controller).

        Parameters
        ----------
        target: WorldObject or sphere (x, y, z, r)
            The object to look at.
        view_dir: 3-tuple of float
            Look at the object from this direction. If not given or None,
            uses the current view direction.
        up: 3-tuple
            If given, set ``camera.world.reference_up`` to the given value.
        scale: float
            Scale the size of what's shown. Default 1.
        match_aspect: bool
            Whether to match the camera's ``width`` and ``height`` to the
            target's bounding box so it tighly fits the viewport. Useful with a
            PanZoomController, less so with a 3D scene.
        """

        if up is None:
            up = self.world.reference_up
        else:
            up = np.asarray(up)
            self.world.reference_up = up

        # Get bounding sphere from target
        bbox = None
        if isinstance(target, WorldObject):
            bsphere = target.get_world_bounding_sphere()
            bbox = target.get_world_bounding_box()
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
            rotation = self.local.rotation
            view_dir = la.vec_transform_quat((0, 0, -1), rotation)
        elif isinstance(view_dir, (tuple, list, np.ndarray)) and len(view_dir) == 3:
            view_dir = tuple(view_dir)
        else:
            raise TypeError(f"Expected view_dir to be sequence, not {view_dir}")
        view_dir = la.vec_normalize(view_dir)

        # Do the math ...
        view_pos = bsphere[:3]
        radius = max(0.0, bsphere[3]) or 1.0
        extent = radius * 2 * scale

        # Apply
        distance = fov_distance_factor(self.fov) * extent
        self.local.position = view_pos - view_dir * distance
        self.look_at(view_pos)
        self._set_extent(extent)

        if match_aspect and bbox is not None:
            # Re-calculate width and height using the aligned bbox, so that the
            # contents keep fitting snugly as the viewport is resized.
            bbox = la.aabb_transform(bbox, self.world.inverse_matrix)
            extent_xy = (bbox[1, :2] - bbox[0, :2]) * scale
            self.width = float(extent_xy[0])
            self.height = float(extent_xy[1])
            # Adust distance to match the new extent (the direction is unchanged)
            extent = 0.5 * (self._width + self._height)
            distance = fov_distance_factor(self.fov) * extent
            self.local.position = view_pos - view_dir * distance

    def show_rect(self, left, right, top, bottom, *, view_dir=None, up=None):
        """Position the camera such that the given rectangle is in view.

        The rectangle represents a plane in world coordinates, centered
        at the origin of the world, and rotated to be orthogonal to the
        view_dir.

        Sets the position and rotation of the camera, and adjusts
        width and height to the rectangle (thus also changing the aspect).

        This method is mainly intended for viewing 2D data, especially
        when `maintain_aspect` is set to False, and is convenient
        for setting the initial view before attaching a PanZoomController.

        See ``show_object(..., match_aspect=True)`` for a more automatic approach.

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
            Look at the rectangle from this direction. If not given or None,
            uses the current view direction.
        up: 3-tuple
           If given, set ``camera.world.reference_up`` to the given value.

        """

        if up is not None:
            self.world.reference_up = up

        # Obtain view direction
        if view_dir is None:
            view_dir = la.vec_transform_quat((0, 0, -1), self.world.rotation)
        else:
            view_dir = la.vec_normalize(view_dir)

        # Set bounds, note that this implicitly sets width, height (and aspect)
        self.width = right - left
        self.height = bottom - top
        extent = 0.5 * (self.width + self.height)
        # First move so we view towards the origin with the correct vector
        distance = fov_distance_factor(self.fov) * extent
        self.world.position = (0, 0, 0) - view_dir * distance
        self.look_at((0, 0, 0))

        # Now we have a rotation that we can use to orient our rect
        position = self.world.position
        rotation = self.world.rotation

        offset = 0.5 * (left + right), 0.5 * (top + bottom), 0
        new_position = position + la.vec_transform_quat(offset, rotation)
        self.world.position = new_position

    @cached
    def frustum(self):
        """Corner positions of the viewing frustum in world space.

        Returns
        -------
        frustum : ndarray, [2, 4, 3]
            The coordinates of the frustum. The first axis corresponds to the
            frustum's plane (near, far), the second to the corner within the
            plane ((left, bottom), (right, bottom), (right, top), (left, top)),
            and the third to the world position of that corner.

        """
        ndc_corners = np.array([(-1, -1), (1, -1), (1, 1), (-1, 1)])
        depths = np.array((0, 1))[:, None]
        local_corners = la.vec_unproject(
            ndc_corners, self.projection_matrix, depth=depths
        )
        world_corners = la.vec_transform(local_corners, self.world.matrix)
        world_corners.flags.writeable = False
        return world_corners


def fov_distance_factor(fov):
    # It's important that controller and camera use the same distance calculations
    if fov > 0:
        fov_rad = fov * pi / 180
        factor = 0.5 / tan(0.5 * fov_rad)
    else:
        factor = 1.0
    return factor
