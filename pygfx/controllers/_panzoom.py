import numpy as np

from ._base import Controller


class PanZoomController(Controller):
    """A controller to pan and zoom a camera in a 2D plane  parallel to the screen.

    Controls:

    * Left mouse button: pan.
    * Right mouse button: zoom (if `camera.maintain_aspect==False`, zooms in both dimensions).
    * Middle mouse button: quick zoom.
    * Scroll: zoom.

    """

    _default_controls = {
        "mouse1": ("pan", "drag", (1, 1)),
        "mouse2": ("zoom", "drag", (0.005, -0.005)),
        "arrowLeft": ("pan", "repeat", (-50, 0)),
        "arrowRight": ("pan", "repeat", (+50, 0)),
        "arrowUp": ("pan", "repeat", (0, -50)),
        "arrowDown": ("pan", "repeat", (0, +50)),
        "z": ("quickzoom", "peak", 2),
        "wheel": ("zoom_to_point", "push", -0.001),
    }

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.mouse_zoom_factor = 0.005
        self.scroll_zoom_factor = 0.0015
        self.quick_zoom_factor = 4

    @property
    def mouse_zoom_factor(self):
        """The factor to turn mouse motion (in logical pixels) to a zoom factor)."""
        return self._mouse_zoom_factor

    @mouse_zoom_factor.setter
    def mouse_zoom_factor(self, value):
        self._mouse_zoom_factor = float(value)

    @property
    def scroll_zoom_factor(self):
        """The factor to turn mouse scrolling to a zoom factor)."""
        return self._scroll_zoom_factor

    @scroll_zoom_factor.setter
    def scroll_zoom_factor(self, value):
        self._scroll_zoom_factor = float(value)

    @property
    def quick_zoom_factor(self):
        """The multiplier to use for quickzoom."""
        return self._quick_zoom_factor

    @quick_zoom_factor.setter
    def quick_zoom_factor(self, value):
        self._quick_zoom_factor = float(value)

    def pan(self, delta, rect):
        """Pan the camera (move relative to its local coordinate frame)."""

        if self._cameras:
            vecx, vecy = self.get_camera_vecs(rect)
            self._update_pan(delta, vecx=vecx, vecy=vecy)
            self.update_cameras()

    def _update_pan(self, delta, *, vecx, vecy):
        # These update methods all accept one positional arg: the delta.
        # it can additionally require keyword args, from a set of names
        # that new actions cache. These include:
        # rect, screen_pos, vecx, vecy

        assert isinstance(delta, tuple) and len(delta) == 2

        cam_state = self.get_camera_state()
        position = cam_state["position"]

        # Update position, panning left means dragging the scene to the
        # left, i.e. move the camera to the right, thus the minus. But
        # since screen pixels go from top to bottom, while the camera's
        # up vector points ... up, the y component is negated twice.
        new_position = position - vecx * delta[0] + vecy * delta[1]

        self.set_camera_state({"position": new_position})

    def zoom(self, delta, rect):
        """Zoom the view with the given amount.

        The delta can be either a scalar or 2-element tuple. The zoom
        multiplier is calculated using ``2**delta``. If the camera has
        maintain_aspect set to True, only the second value is used.

        Note that the camera's distance, width, and height are adjusted,
        not its zoom property.
        """

        if self._cameras:
            self._update_zoom(delta)
            self.update_cameras_cameras()

    def _update_zoom(self, delta):
        if isinstance(delta, (int, float)):
            delta = (delta, delta)
        assert isinstance(delta, tuple) and len(delta) == 2

        fx = 2 ** delta[0]
        fy = 2 ** delta[1]
        new_cam_state = self._zoom(fx, fy, self.get_camera_state())
        self.set_camera_state(new_cam_state)

    def zoom_to_point(self, zoom_value, pos, rect):
        """Zoom the view while panning to keep the position under the cursor fixed."""

        if self._cameras:
            self._update_zoom_to_point(zoom_value, screen_pos=pos, rect=rect)
            self.update_cameras()

    def _update_zoom_to_point(self, delta, *, screen_pos, rect):
        if isinstance(delta, tuple) and len(delta) == 2:
            delta = delta[1]
        assert isinstance(delta, (int, float))

        # Actuall only zoom in one direction
        fy = 2**delta

        new_cam_state = self._zoom(fy, fy, self.get_camera_state())
        self.set_camera_state(new_cam_state)

        pan_delta = self._get_panning_to_compensate_zoom(fy, screen_pos, rect)
        vecx, vecy = self.get_camera_vecs(rect)
        self._update_pan(pan_delta, vecx=vecx, vecy=vecy)

    def _zoom(self, fx, fy, cam_state):
        position = cam_state["position"]
        maintain_aspect = cam_state["maintain_aspect"]
        width = cam_state["width"]
        height = cam_state["height"]
        extent = 0.5 * (width + height)

        # Scale width and height equally, or use width and height.
        if maintain_aspect:
            new_width = width / fy
            new_height = height / fy
        else:
            new_width = width / fx
            new_height = height / fy

        # Get new position
        new_extent = 0.5 * (new_width + new_height)
        pos2target1 = self._get_target_vec(cam_state, extent=extent)
        pos2target2 = self._get_target_vec(cam_state, extent=new_extent)
        new_position = position + pos2target1 - pos2target2

        return {
            "width": new_width,
            "height": new_height,
            "position": new_position,
            "fov": cam_state["fov"],
        }

    def _get_panning_to_compensate_zoom(self, multiplier, screen_pos, rect):
        # Get viewport info
        x, y, w, h = rect

        # Distance from the center of the rect
        delta_screen_x = screen_pos[0] - x - w / 2
        delta_screen_y = screen_pos[1] - y - h / 2
        delta_screen1 = np.array([delta_screen_x, delta_screen_y])

        # New position after zooming
        delta_screen2 = delta_screen1 * multiplier

        # The amount to pan is the difference, but also scaled with the multiplier
        # because pixels take more/less space now.
        return tuple((delta_screen1 - delta_screen2) / multiplier)

    def quickzoom(self, zoom_value):
        """Zoom the view using the camera's zoom property. This is intended
        for temporary zoom operations.
        """
        if self._cameras:
            self._update_quickzoom(zoom_value)
            self.update_cameras()

    def _update_quickzoom(self, delta):
        assert isinstance(delta, (int, float))
        zoom = self.get_camera_state()["zoom"]
        new_cam_state = {"zoom": zoom * 2**delta}
        self.set_camera_state(new_cam_state)

    def adjust_fov(self, delta: float):
        """Adjust the field of view with the given delta value (Limited to [1, 179])."""

        if self._action:
            return

        if self._cameras:
            camera = self._cameras[0]

            # Get current state
            camera_state = camera.get_state()
            position = camera_state["position"]
            fov = camera_state["fov"]

            # Update fov and position
            new_fov = min(max(fov + delta, camera._fov_range[0]), camera._fov_range[1])
            pos2target1 = self._get_target_vec(camera_state, fov=fov)
            pos2target2 = self._get_target_vec(camera_state, fov=new_fov)
            new_position = position + pos2target1 - pos2target2

            # Apply to cameras
            new_camera_state = {
                **camera_state,
                "fov": new_fov,
                "position": new_position,
            }
            for camera in self._cameras:
                camera.set_state(new_camera_state)

        return self
