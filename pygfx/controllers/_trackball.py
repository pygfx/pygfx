from typing import Tuple

import numpy as np
import pylinalg as la

from ..utils.viewport import Viewport
from ._base import Controller
from ._orbit import OrbitController


def _get_axis_aligned_up_vector(up):
    # Not actually used, but I have a feeling we might need it at some point :)
    ref_up, largest_dot = None, 0
    for up_vec in [(1, 0, 0), (0, 1, 0), (0, 0, 1), (-1, 0, 0), (0, -1, 0), (0, 0, -1)]:
        up_vec = np.array(up_vec)
        v = np.dot(up_vec, up)
        if v > largest_dot:
            ref_up, largest_dot = up_vec, v
    return ref_up


class TrackballController(OrbitController):
    """A controller to freely rotate a camera around a center position.

    This controller is similar to the OrbitController, but it does not
    maintain a constant camera up vector.

    Controls:

    * Left mouse button: rotate.
    * Right mouse button: pan.
    * Middle mouse button: quick zoom.
    * Scroll: zoom.
    * Alt + Scroll: change FOV.

    """

    def _rotate(self, dx, dy, camera_state):
        # Note: this code does not use la.vector_euclidean_to_spherical and
        # la.vector_spherical_to_euclidean, because those functions currently
        # have no way to specify a different up vector.

        position = camera_state["position"]
        rotation = camera_state["rotation"]

        qx = la.quaternion_make_from_axis_angle((0, 1, 0), -dx)
        qy = la.quaternion_make_from_axis_angle((1, 0, 0), -dy)

        base_rot = (0, 0, 0, 1)
        if self._action and "base_rot" in self._action:
            base_rot = self._action["base_rot"]

        base_rot = la.quaternion_multiply(base_rot, la.quaternion_multiply(qy, qx))
        if self._action:
            self._action["base_rot"] = base_rot

        # Get rotations
        rot1 = rotation
        rot2 = la.quaternion_multiply(rot1, base_rot)
        #

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

    def rotate_move(
        self, pos: Tuple[float, float], speed: float = 0.0175
    ) -> Controller:
        if self._action and self._action["name"] == "rotate":
            dx = (pos[0] - self._action["last_pos"][0]) * speed
            dy = (pos[1] - self._action["last_pos"][1]) * speed

            self._rotate(dx, dy, self._action["camera_state"])

        return self
