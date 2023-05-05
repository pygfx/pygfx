from typing import Tuple

import pylinalg as la

from ._orbit import OrbitController


class TrackballController(OrbitController):
    """A controller to freely rotate a camera around a center position.

    This controller is similar to the OrbitController, but it does not
    maintain a constant camera up vector.

    Default controls:

    * Left mouse button: orbit / rotate.
    * Right mouse button: pan.
    * Fourth mouse button: quickzoom
    * wheel: zoom to point.
    * alt+wheel: adjust fov.
    """

    _default_controls = OrbitController._default_controls.copy()

    def rotate(self, delta: Tuple, rect: Tuple, *, animate=False):
        """Rotate around the target, using two angles (in radians).

        If animate is True, the motion is damped. This requires the
        controller to receive events from the renderer/viewport.
        """
        return super().rotate(delta, rect, animate=animate)

    def _update_rotate(self, delta):
        assert isinstance(delta, tuple) and len(delta) == 2

        dx, dy = delta
        camera_state = self._get_camera_state()

        position = camera_state["position"]
        rotation = camera_state["rotation"]

        qx = la.quat_from_axis_angle((0, 1, 0), -dx)
        qy = la.quat_from_axis_angle((1, 0, 0), -dy)

        delta_rot = la.quat_mul((0, 0, 0, 1), la.quat_mul(qy, qx))

        # Get rotations
        rot1 = rotation
        rot2 = la.quat_mul(rot1, delta_rot)

        # Calculate new position
        pos1 = position
        pos2target1 = self._get_target_vec(camera_state, rotation=rot1)
        pos2target2 = self._get_target_vec(camera_state, rotation=rot2)
        pos2 = pos1 + pos2target1 - pos2target2

        # Apply new state
        new_camera_state = {"position": pos2, "rotation": rot2}
        self._set_camera_state(new_camera_state)
