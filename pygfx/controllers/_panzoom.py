from typing import Tuple

import numpy as np

from ..utils.viewport import Viewport
from ._base import Controller, get_screen_vectors_in_world_cords


class PanZoomController(Controller):
    """A controller to pan and zoom a camera in a 2D plane  parallel to the screen.

    Controls:

    * Left mouse button: pan.
    * Right mouse button: zoom (if `camera.maintain_aspect==False`, zooms in both dimensions).
    * Middle mouse button: quick zoom.
    * Scroll: zoom.

    """

    _default_controls = {
        "drag1": "pan(1)",
        "drag2": "zoom(0.005, -0.005)",
        "arrowleft*": "pan(-4, 0)",
        "arrowright*": "pan(+4, 0)",
        "z": "quickzoom(0.1)",
        "x*": "quickzoom(0.1)",
        "c!": "quickzoom(2)",
        "wheel": "zoom_to_point(0.0015, 0.0015)",
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

    def pan(self, dxdy, rect):
        # action = self.begin_pan((0, 0), rect)
        # self.update_pan(delta, action)
        # self._update_all_cameras()

        action = self.begin_pan((0, 0), rect)
        action.set_target(dxdy)
        action.tick(1)
        self.apply_action(action)

    def begin_pan(self, screen_pos, rect):
        assert len(screen_pos) == 2

        # # Get offset in correct shape
        # if offset is None:
        #     offset = (0, 0)
        # else:
        #     offset = tuple(offset)
        #     assert len(offset) == 2

        action = self._create_new_action("pan", (0, 0), screen_pos, rect)
        return action

    def _update_pan(self, action, delta=None):
        cam_state = action.last_cam_state
        vecx = action.vecx
        vecy = action.vecy

        position = cam_state["position"]
        if delta is None:
            delta = action.delta
        new_position = position + vecy * delta[1] - vecx * delta[0]

        self._apply_new_camera_state({"position": new_position})

    def zoom(self, zoom_value, rect):
        """Zoom the view with the given multipliers.

        If the camera has maintain_aspect set to True, only fy is used.
        The camera's distance, width, and height are adjusted, not its
        zoom property.
        """

        if isinstance(zoom_value, (int, float)):
            zoom_value = (zoom_value, zoom_value)

        action = self.begin_zoom((0, 0), rect)
        action.set_target(zoom_value)
        action.tick(1)
        self.apply_action(action)

    def begin_zoom(self, screen_pos, rect):
        action = self._create_new_action("zoom", (0, 0), screen_pos, rect)
        return action

    def _update_zoom(self, action):
        delta = action.delta
        fx = 2 ** delta[0]
        fy = 2 ** delta[1]
        new_cam_state = self._zoom(fx, fy, action.last_cam_state)
        self._apply_new_camera_state(new_cam_state)

    def zoom_to_point(self, zoom_value, pos, rect):
        if isinstance(zoom_value, (int, float)):
            zoom_value = (zoom_value, zoom_value)

    def begin_zoom_to_point(self, screen_pos, rect):
        action = self._create_new_action("zoom_to_point", (0, 0), screen_pos, rect)
        return action

    def _update_zoom_to_point(self, action):
        fy = 2 ** action.delta[1]
        new_cam_state = self._zoom(fy, fy, action.last_cam_state)

        # new_cam_state["position"] -= pan_delta
        self._apply_new_camera_state(new_cam_state)

        pan_delta = self._get_panning_to_compensate_zoom(fy, action)
        self.pan(pan_delta, action.rect)

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

    def _get_panning_to_compensate_zoom(self, multiplier, action):
        # Get viewport info
        x, y, w, h = action.rect

        # The position that we want to keep in the same place
        screen_pos = action.screen_pos

        # Distance from the center of the rect
        delta_screen_x = screen_pos[0] - x - w / 2
        delta_screen_y = screen_pos[1] - y - h / 2
        delta_screen1 = np.array([delta_screen_x, delta_screen_y])

        # New position after zooming
        delta_screen2 = delta_screen1 * multiplier

        # The amount to pan is the difference, but also scaled with the multiplier
        # because pixels take more/less space now.
        return (delta_screen1 - delta_screen2) / multiplier

    def quickzoom(self, zoom_value, rect):
        assert isinstance(zoom_value, (int, float))

        action = self.begin_quickzoom(0, rect)
        action.set_target(zoom_value)
        action.tick(1)
        self.apply_action(action)

    def begin_quickzoom(self, screen_pos, rect):
        action = self._create_new_action("quickzoom", 0, screen_pos, rect)
        return action

    def _update_quickzoom(self, action):
        zoom = action.last_cam_state["zoom"]
        print(zoom)
        new_cam_state = {"zoom": zoom + action.delta}
        self._apply_new_camera_state(new_cam_state)

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
