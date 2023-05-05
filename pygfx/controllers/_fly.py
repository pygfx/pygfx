from typing import Tuple

import pylinalg as la

from ._base import Controller


class FlyController(Controller):
    """A controller to fly around a scene as if it's a flight simulator.

    Default controls:

    * Left mouse button: rotate.
    * "wasd": move forward, backwards, and strafe to the sides.
    * space/shift: move up/down.
    * "qe": roll the camera/aircraft around it's axis.
    * wheel: increase/decrease maximum speed.
    * Fourth mouse button: quickzoom
    * alt+wheel: adjust fov.
    """

    _default_controls = {
        "mouse1": ("rotate", "drag", (0.005, 0.005)),
        "q": ("roll", "repeat", -2),
        "e": ("roll", "repeat", +2),
        "w": ("move", "repeat", (0, 0, -1)),
        "s": ("move", "repeat", (0, 0, +1)),
        "a": ("move", "repeat", (-1, 0, 0)),
        "d": ("move", "repeat", (+1, 0, 0)),
        " ": ("move", "repeat", (0, +1, 0)),
        "shift": ("move", "repeat", (0, -1, 0)),
        "mouse4": ("quickzoom", "peek", 2),
        "wheel": ("speed", "push", -0.001),
        "alt+wheel": ("fov", "push", -0.01),
    }

    def __init__(self, camera, *, speed=None, **kwargs):
        super().__init__(camera, **kwargs)

        if speed is None:
            cam_state = camera.get_state()
            approx_scene_size = 0.5 * (cam_state["width"] + cam_state["height"])
            scene_fly_thru_time = 5  # seconds
            speed = approx_scene_size / scene_fly_thru_time
        self.speed = speed

    @property
    def speed(self):
        """The (maximum) speed that the camera will move, in units per second.
        By default it's based off the width and height of the camera.
        """
        return self._speed

    @speed.setter
    def speed(self, value):
        self._speed = float(value)

    def rotate(self, delta: Tuple, rect: Tuple, *, animate=False):
        """Rotate the camera over two angles (pitch and yaw in radians).

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

        dx, dy = delta
        camera_state = self._get_camera_state()
        rotation = camera_state["rotation"]

        qx = la.quat_from_axis_angle((0, 1, 0), -dx)
        qy = la.quat_from_axis_angle((1, 0, 0), -dy)

        delta_rot = la.quat_mul((0, 0, 0, 1), la.quat_mul(qy, qx))

        new_rotation = la.quat_mul(rotation, delta_rot)

        # Apply new state
        new_camera_state = {"rotation": new_rotation}
        self._set_camera_state(new_camera_state)

    def roll(self, delta: float, rect: Tuple, *, animate=False):
        """Rotate the camera over the z-axis (roll, in radians).

        If animate is True, the motion is damped. This requires the
        controller to receive events from the renderer/viewport.
        """
        if animate:
            action_tuple = ("roll", "push", 1.0)
            action = self._create_action(None, action_tuple, 0.0, None, rect)
            action.set_target(delta)
            action.snap_distance = 0.01
            action.done = True
        elif self._cameras:
            self._update_rotate(delta)
            return self._update_cameras()

    def _update_roll(self, delta):
        assert isinstance(delta, float)

        camera_state = self._get_camera_state()
        rotation = camera_state["rotation"]

        qz = la.quat_from_axis_angle((0, 0, 1), -delta)
        new_rotation = la.quat_mul(rotation, qz)

        new_camera_state = {"rotation": new_rotation}
        self._set_camera_state(new_camera_state)

    def move(self, delta: Tuple, rect: Tuple, *, animate=False):
        """Move the camera in the given (x, y, z) direction.

        The delta is expressed in the camera's local coordinate frame.
        Forward is in -z direction, because as (per the gltf spec) a
        camera looks down it's negative Z-axis.

        If animate is True, the motion is damped. This requires the
        controller to receive events from the renderer/viewport.
        """

        if animate:
            action_tuple = ("move", "push", (1.0, 1.0, 1.0))
            action = self._create_action(
                None, action_tuple, (0.0, 0.0, 0.0), None, rect
            )
            action.set_target(delta)
            action.done = True
        elif self._cameras:
            self._update_move(delta)
            return self._update_cameras()

    def _update_move(self, delta):
        assert isinstance(delta, tuple) and len(delta) == 3

        cam_state = self._get_camera_state()
        position = cam_state["position"]
        rotation = cam_state["rotation"]
        delta_world = la.vec_transform_quat(delta, rotation)

        new_position = position + delta_world * self.speed
        self._set_camera_state({"position": new_position})

    def _update_speed(self, delta):
        assert isinstance(delta, float)
        speed = self.speed * 2**delta
        self.speed = max(0.001, speed)
