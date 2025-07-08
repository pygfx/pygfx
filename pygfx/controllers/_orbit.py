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

    Parameters
    ----------
    target : tuple of float, optional
        The custom target position (x, y, z) that the camera orbits around.
        Default is None, which means the target is determined from the camera state.

    Notes
    -----
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

    def __init__(self, *args, target=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.target = np.array(target) if target is not None else None

    @property
    def target(self):
        """The target position (x, y, z) that the camera orbits around.

        Set to None to use an implicit target based on the camera state. This only works
        well in combination with ``camera.show_object()`` and ``camera.show_pos()``.
        """
        if self._custom_target is not None:
            return self._custom_target.copy()
        else:
            camera_state = self._get_camera_state()
            return self._get_target_vec(camera_state)

    @target.setter
    def target(self, value):
        if value is None:
            self._custom_target = None
        else:
            self._custom_target = np.array(value, dtype=np.float32)
            for camera, _, _ in self._cameras:
                camera.look_at(self._custom_target)

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
        if self._custom_target is not None:
            # If we have a custom target, we need to calculate the target vector

            target_pos = self._custom_target
            pos1_to_target = target_pos - pos1

            pos1_to_target_rotated = la.vec_transform_quat(pos1_to_target, r_azimuth)

            right = la.vec_transform_quat((1, 0, 0), rot1)
            r_elevation_world = la.quat_from_axis_angle(right, -delta_elevation)
            pos1_to_target_final = la.vec_transform_quat(
                pos1_to_target_rotated, r_elevation_world
            )

            pos2 = target_pos - pos1_to_target_final
        else:
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

    def _update_pan(self, delta, *, vecx, vecy):
        if self._custom_target is not None:
            # If we have a custom target, we need to pan relative to that
            target_pos = self._custom_target

            distance_to_target = la.vec_dist(
                target_pos, self._get_camera_state()["position"]
            )

            scaled_delta = (
                delta[0] * distance_to_target * 0.01,
                delta[1] * distance_to_target * 0.01,
            )
            offest = -vecx * scaled_delta[0] + vecy * scaled_delta[1]
            self._custom_target = target_pos + offest
        else:
            scaled_delta = delta

        return super()._update_pan(scaled_delta, vecx=vecx, vecy=vecy)

    def _update_zoom(self, delta):
        if isinstance(delta, (int, float)):
            delta = (delta, delta)
        assert isinstance(delta, tuple) and len(delta) == 2

        if self._custom_target is not None:
            target_pos = self._custom_target
            radius_vec = self._get_camera_state()["position"] - target_pos

            scale = 1 - delta[0]
            if scale < 0.1:
                scale = 0.1

            radius_vec *= scale
            new_position = target_pos + radius_vec
            self._set_camera_state(
                {
                    "position": new_position,
                }
            )

        else:
            fx = 2 ** delta[0]
            fy = 2 ** delta[1]
            new_cam_state = self._zoom(fx, fy, self._get_camera_state())
            self._set_camera_state(new_cam_state)
