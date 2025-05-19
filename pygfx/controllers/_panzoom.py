from typing import Tuple

import numpy as np

from ._base import Controller


class PanZoomController(Controller):
    """A controller to pan and zoom a camera in a 2D plane  parallel to the screen.

    Default controls:

    * Left mouse button: pan.
    * Right mouse button: zoom (if `camera.maintain_aspect==False`, zooms in both dimensions).
    * Fourth mouse button: quickzoom
    * wheel: zoom to point.
    * alt+wheel: adjust fov.

    """

    _default_controls = {
        "mouse1": ("pan", "drag", (1, 1)),
        "mouse2": ("zoom", "drag", (0.005, -0.005)),
        "mouse4": ("quickzoom", "peek", 2),
        "wheel": ("zoom_to_point", "push", -0.001),
        "alt+wheel": ("fov", "push", -0.01),
    }

    def pan(self, delta: Tuple, rect: Tuple, *, animate=False):
        """Pan the camera (move relative to its local coordinate frame).

        If animate is True, the motion is damped. This requires the
        controller to receive events from the renderer/viewport.
        """

        # Note that we need that rect to determine mouse positions relative
        # to the viewport. If all events were adjusted to the viewport
        # than code like this would not have to care ...

        # Note how we return the result of _update_cameras when not animating.
        # If auto_update is False, _update_cameras will do nothing. The return value
        # is the camera state, which the calling code can then process further.

        if animate:
            action_tuple = ("pan", "push", (1.0, 1.0))
            action = self._create_action(None, action_tuple, (0.0, 0.0), None, rect)
            action.set_target(delta)
            action.done = True
        elif self._cameras:
            vecx, vecy = self._get_camera_vecs(rect)
            self._update_pan(delta, vecx=vecx, vecy=vecy)
            return self._update_cameras()

    def _update_pan(self, delta, *, vecx, vecy):
        # These update methods all accept one positional arg: the delta.
        # it can additionally require keyword args, from a set of names
        # that new actions store. These include:
        # rect, screen_pos, vecx, vecy

        assert isinstance(delta, tuple) and len(delta) == 2

        cam_state = self._get_camera_state()
        position = cam_state["position"]

        # Update position, panning left means dragging the scene to the
        # left, i.e. move the camera to the right, thus the minus. But
        # since screen pixels go from top to bottom, while the camera's
        # up vector points ... up, the y component is negated twice.
        new_position = position - vecx * delta[0] + vecy * delta[1]
        self._set_camera_state({"position": new_position})

    def zoom(self, delta: Tuple, rect: Tuple, *, animate=False):
        """Zoom the view with the given amount.

        The delta can be either a scalar or 2-element tuple. The zoom
        multiplier is calculated using ``2**delta``. If the camera has
        maintain_aspect set to True, only the second value is used.

        Note that the camera's distance, width, and height are adjusted,
        not its zoom property.

        If animate is True, the motion is damped. This requires the
        controller to receive events from the renderer/viewport.
        """

        if animate:
            action_tuple = ("zoom", "push", (1.0, 1.0))
            action = self._create_action(None, action_tuple, (0.0, 0.0), None, rect)
            action.set_target(delta)
            action.done = True
        elif self._cameras:
            self._update_zoom(delta)
            return self._update_cameras()

    def _update_zoom(self, delta):
        if isinstance(delta, (int, float)):
            delta = (delta, delta)
        assert isinstance(delta, tuple) and len(delta) == 2

        fx = 2 ** delta[0]
        fy = 2 ** delta[1]
        new_cam_state = self._zoom(fx, fy, self._get_camera_state())
        self._set_camera_state(new_cam_state)

    def zoom_to_point(self, delta: float, pos: Tuple, rect: Tuple, *, animate=False):
        """Zoom the view while panning to keep the position under the cursor fixed.

        If animate is True, the motion is damped. This requires the
        controller to receive events from the renderer/viewport.
        """

        if animate:
            action_tuple = ("zoom_to_point", "push", 1.0)
            action = self._create_action(None, action_tuple, (0.0, 0.0), None, rect)
            action.set_target(delta)
            action.done = True
        elif self._cameras:
            self._update_zoom_to_point(delta, screen_pos=pos, rect=rect)
            return self._update_cameras()

    def _update_zoom_to_point(self, delta, *, screen_pos, rect):
        if isinstance(delta, tuple) and len(delta) == 2:
            delta = delta[1]
        assert isinstance(delta, (int, float))

        # Actually only zoom in one direction
        fy = 2**delta

        new_cam_state = self._zoom(fy, fy, self._get_camera_state())
        self._set_camera_state(new_cam_state)

        pan_delta = self._get_panning_to_compensate_zoom(fy, screen_pos, rect)
        vecx, vecy = self._get_camera_vecs(rect)
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
