from typing import Tuple

import numpy as np
import pylinalg as la

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

    Default controls:

    * Left mouse button: orbit / rotate.
    * Right mouse button: pan.
    * Fourth mouse button: quickzoom
    * wheel: zoom to point.
    * alt+wheel: adjust fov.

    """

    _default_controls = {
        "mouse1": ("rotate", "drag", (0.005, 0.005)),
        "mouse2": ("pan", "drag", (1, 1)),
        "mouse4": ("quickzoom", "peek", 2),
        "wheel": ("zoom", "push", -0.001),
        "alt+wheel": ("fov", "push", -0.01),
    }

    def rotate(self, delta: Tuple, rect: Tuple, *, animate=False):
        """Rotate in an orbit around the target, using two angles (azimuth and elevation, in radians).

        If animate is True, the motion is damped. This requires the
        controller to receive events from the renderer/viewport.
        """

        if animate:
            action_tuple = ("rotate", "push", (1.0, 1.0))
            action = self._create_action(None, action_tuple, (0.0, 0.0), None, rect)
            action.set_target(delta)
            action.snap_distance = 0.01
            action.done = True
        elif self._cameras:
            self._update_rotate(delta)
            return self._update_cameras()

    def _update_rotate(self, delta):
        assert isinstance(delta, tuple) and len(delta) == 2

        delta_azimuth, delta_elevation = delta
        camera_state = self._get_camera_state()

        # Note: this code does not use la.vec_euclidean_to_spherical and
        # la.vec_spherical_to_euclidean, because those functions currently
        # have no way to specify a different up vector.

        position = camera_state["position"]
        rotation = camera_state["rotation"]
        up = camera_state["reference_up"]

        # Where is the camera looking at right now
        forward = la.vec_transform_quat((0, 0, -1), rotation)

        # # Get a reference vector, that is orthogonal to up, in a deterministic way.
        # # Might need this if we ever want the azimuth
        # aligned_up = _get_axis_aligned_up_vector(up)
        # orthogonal_vec = np.cross(up, np.roll(aligned_up, 1))

        # Get current elevation, so we can clip it.
        # We don't need the azimuth. When we do, it'd need more care to get a proper 0..2pi range
        elevation = la.vec_angle(forward, up) - 0.5 * np.pi

        # Apply boundaries to the elevation
        new_elevation = elevation + delta_elevation
        bounds = -89 * np.pi / 180, 89 * np.pi / 180
        if new_elevation < bounds[0]:
            delta_elevation = bounds[0] - elevation
        elif new_elevation > bounds[1]:
            delta_elevation = bounds[1] - elevation

        r_azimuth = la.quat_from_axis_angle(up, -delta_azimuth)
        r_elevation = la.quat_from_axis_angle((1, 0, 0), -delta_elevation)

        # Get rotations
        rot1 = rotation
        rot2 = la.quat_mul(r_azimuth, la.quat_mul(rot1, r_elevation))

        # Calculate new position
        pos1 = position
        pos2target1 = self._get_target_vec(camera_state, rotation=rot1)
        pos2target2 = self._get_target_vec(camera_state, rotation=rot2)
        pos2 = pos1 + pos2target1 - pos2target2

        # Apply new state
        new_camera_state = {"position": pos2, "rotation": rot2}
        self._set_camera_state(new_camera_state)

        # Note that for ortho cameras, we also orbit around the scene,
        # even though it could be positioned at the center (i.e.
        # target). Doing it this way makes the code in the controllers
        # easier. The only downside I can think of is that the far plane
        # is now less far away but this effect is only 0.2%, since the
        # far plane is 500 * dist.
