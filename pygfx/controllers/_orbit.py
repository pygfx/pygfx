from typing import Tuple
import numpy as np
import pylinalg as la

import numpy as np
import pylinalg as la

from ..utils.viewport import Viewport
from ._base import Controller
from ._panzoom import PanZoomController


def _get_axis_aligned_up_vector(up):
    # Not actually used, but I have a feeling we might need it at some point :)
    ref_up, largest_dot = None, 0
    for up_vec in [(1, 0, 0), (0, 1, 0), (0, 0, 1), (-1, 0, 0), (0, -1, 0), (0, 0, -1)]:
        up_vec = np.array(up_vec)
        v = np.dot(up_vec, up)
        if v > largest_dot:
            ref_up, largest_dot = up_vec, v
    return ref_up


class OrbitController(PanZoomController):
    """A controller to move a camera in an orbit around a center position.

    Supports panning parallel to the screen, zooming, orbiting.

    The direction of rotation is defined such that it feels like you're
    grabbing onto something in the foreground; if you move the mouse
    to the right, the objects in the foreground move to the right, and
    those in the background (on the opposite side of the center of rotation)
    move to the left.

    Controls:

    * Left mouse button: orbit / rotate.
    * Right mouse button: pan.
    * Middle mouse button: quick zoom.
    * Scroll: zoom.
    * Alt + Scroll: change FOV.

    """

    def rotate(self, delta_azimuth: float, delta_elevation: float) -> Controller:
        """Rotate using angles (in radians)."""
        if self._action:
            return

        if self._cameras:
            camera = self._cameras[0]

            self._rotate(delta_azimuth, delta_elevation, camera.get_state())

        return self

    def _rotate(self, delta_azimuth, delta_elevation, camera_state):
        # Note: this code does not use la.vector_euclidean_to_spherical and
        # la.vector_spherical_to_euclidean, because those functions currently
        # have a way to specify a different up vector.

        position = camera_state["position"]
        rotation = camera_state["rotation"]
        up = camera_state["up"]

        # Where is the camera looking at right now
        forward = la.vector_apply_quaternion((0, 0, -1), rotation)

        # # Get a reference vector, that is orthogonal to up, in a deterministic way.
        # # Might need this if we ever want the azimuth
        # aligned_up = _get_axis_aligned_up_vector(up)
        # orthogonal_vec = np.cross(up, np.roll(aligned_up, 1))

        # Get current elevation, so we can clip it.
        # We don't need the azimuth. When we do, it'd need more care to get a proper 0..2pi range
        elevation = la.vector_angle_between(forward, up) - 0.5 * np.pi

        # Apply boundaries to the elevation
        new_elevation = elevation + delta_elevation
        bounds = -89 * np.pi / 180, 89 * np.pi / 180
        if new_elevation < bounds[0]:
            delta_elevation = bounds[0] - elevation
        elif new_elevation > bounds[1]:
            delta_elevation = bounds[1] - elevation

        r_azimuth = la.quaternion_make_from_axis_angle(up, -delta_azimuth)
        r_elevation = la.quaternion_make_from_axis_angle((1, 0, 0), -delta_elevation)

        # Get rotations
        rot1 = rotation
        rot2 = la.quaternion_multiply(
            r_azimuth, la.quaternion_multiply(rot1, r_elevation)
        )

        # Calculate new position
        pos1 = position
        pos2target1 = self._get_target_vec(camera_state, rotation=rot1)
        pos2target2 = self._get_target_vec(camera_state, rotation=rot2)
        pos2 = pos1 + pos2target1 - pos2target2

        # Apply new state to all cameras
        new_camera_state = {"position": pos2, "rotation": rot2}
        for camera in self._cameras:
            camera.set_state(new_camera_state)

        # Note that for ortho cameras, we also orbit around the scene,
        # even though it could be positioned at the center (i.e.
        # target). Doing it this way makes the code in the controllers
        # easier. The only downside I can think of is that the far plane
        # is now less far away but this effect is only 0.2%, since the
        # far plane is 500 * dist.

    def rotate_start(self, pos: Tuple[float, float], viewport: Viewport) -> Controller:
        """Start a rotation operation based (2D) screen coordinates."""
        if self._action:
            return

        if self._cameras:
            camera = self._cameras[0]

            self._action = {"name": "rotate"}
            self._action["camera_state"] = camera.get_state()
            self._action["mouse_pos"] = pos

        return self

    def rotate_stop(self) -> Controller:
        self._action = None
        return self

    def rotate_move(
        self, pos: Tuple[float, float], speed: float = 0.0175
    ) -> Controller:
        """Rotate, based on a (2D) screen location. Call rotate_start first.
        The speed is 1 degree per pixel by default.
        """
        if self._action and self._action["name"] == "rotate":
            delta_azimuth = (pos[0] - self._action["mouse_pos"][0]) * speed
            delta_elevation = (pos[1] - self._action["mouse_pos"][1]) * speed
            self._rotate(delta_azimuth, delta_elevation, self._action["camera_state"])
        return self

    def handle_event(self, event, viewport):
        """Implements a default interaction mode that consumes wgpu autogui events
        (compatible with the jupyter_rfb event specification).
        """
        if not self.enabled:
            return
        need_update = False

        type = event.type
        if type == "pointer_down" and viewport.is_inside(event.x, event.y):
            xy = event.x, event.y
            if event.button == 1:
                self.rotate_start(xy, viewport)
            elif event.button == 2:
                self.pan_start(xy, viewport)
            elif event.button == 3:
                self.quickzoom_start(xy, viewport)
                need_update = True
        elif type == "pointer_up":
            xy = event.x, event.y
            if event.button == 1:
                self.rotate_stop()
            elif event.button == 2:
                self.pan_stop()
            elif event.button == 3:
                self.quickzoom_stop()
                need_update = True
        elif type == "pointer_move":
            xy = event.x, event.y
            if 1 in event.buttons:
                self.rotate_move(xy),
                need_update = True
            if 2 in event.buttons:
                self.pan_move(xy),
                need_update = True
            if 3 in event.buttons:
                self.quickzoom_move(xy)
                need_update = True
        elif type == "wheel" and viewport.is_inside(event.x, event.y):
            if not event.modifiers:
                xy = event.x, event.y
                d = event.dy or event.dx
                f = 2 ** (-d * self.scroll_zoom_factor)
                self.zoom(f)
                need_update = True
            elif event.modifiers == ["Alt"]:
                d = event.dy or event.dx
                self.adjust_fov(-d / 10)
                need_update = True

        if need_update and self.auto_update:
            viewport.renderer.request_draw()
