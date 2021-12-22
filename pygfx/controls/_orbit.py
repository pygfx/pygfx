from typing import Tuple

from ..linalg import Vector3, Matrix4, Quaternion, Spherical


# todo: maybe make an OrbitOrthoControls for ortho cameras, instead of this zoom param?


def get_screen_vectors_in_world_cords(
    center_world: Vector3, canvas_size: Tuple[float, float], camera: "Camera"
) -> Tuple[Vector3, Vector3]:
    """Given a reference center location (in 3D world coordinates)
    Get the vectors corresponding to the x and y direction in screen coordinates.
    These vectors are scaled so that they can simply be multiplied with the
    delta x and delta y.
    """
    center = center_world.clone().project(camera)
    pos1 = Vector3(100, 0, center.z).unproject(camera)
    pos2 = Vector3(0, 100, center.z).unproject(camera)
    pos1.multiply_scalar(0.02 / canvas_size[0])
    pos2.multiply_scalar(0.02 / canvas_size[1])
    return pos1, pos2  # now they're vecs, really


class OrbitControls:
    """A class implementing orbit controls, where the camera is
    rotated around a center position (orbiting around it).
    """

    def __init__(
        self,
        eye: Vector3 = None,
        target: Vector3 = None,
        up: Vector3 = None,
        zoom_changes_distance=True,
        min_zoom: float = 0.0001,
    ) -> None:
        self.rotation = Quaternion()
        if eye is None:
            eye = Vector3(50.0, 50.0, 50.0)
        if target is None:
            target = Vector3()
        if up is None:
            up = Vector3(0.0, 1.0, 0.0)
        self.zoom_changes_distance = bool(zoom_changes_distance)
        self.zoom_value = 1
        self.min_zoom = min_zoom

        # State info used during a pan or rotate operation
        self._pan_info = None
        self._rotate_info = None

        # Temp objects (to avoid garbage collection)
        self._m = Matrix4()
        self._v = Vector3()
        self._origin = Vector3()
        self._orbit_up = Vector3(0, 1, 0)
        self._s = Spherical()

        # Initialize orientation
        self.look_at(eye, target, up)
        self._initial_distance = self.distance

    def look_at(self, eye: Vector3, target: Vector3, up: Vector3) -> "OrbitControls":
        self.distance = eye.distance_to(target)
        self.target = target
        self.up = up
        self.rotation.set_from_rotation_matrix(self._m.look_at(eye, target, up))
        self._up_quat = Quaternion().set_from_unit_vectors(self.up, self._orbit_up)
        self._up_quat_inv = self._up_quat.clone().inverse()
        return self

    def pan(self, vec3: Vector3) -> "OrbitControls":
        """Pan in 3D world coordinates."""
        self.target.add(vec3)
        return self

    def pan_start(
        self,
        pos: Tuple[float, float],
        canvas_size: Tuple[float, float],
        camera: "Camera",
    ) -> "OrbitControls":
        """Start a panning operation based (2D) screen coordinates."""
        vecx, vecy = get_screen_vectors_in_world_cords(self.target, canvas_size, camera)
        self._pan_info = {"last": pos, "vecx": vecx, "vecy": vecy}
        return self

    def pan_stop(self) -> "OrbitControls":
        self._pan_info = None
        return self

    def pan_move(self, pos: Tuple[float, float]) -> "OrbitControls":
        """Pan the center of rotation, based on a (2D) screen location. Call pan_start first."""
        if self._pan_info is None:
            return
        delta = tuple((pos[i] - self._pan_info["last"][i]) for i in range(2))
        self.pan(
            self._pan_info["vecx"]
            .clone()
            .multiply_scalar(-delta[0])
            .add_scaled_vector(self._pan_info["vecy"], +delta[1])
        )
        self._pan_info["last"] = pos
        return self

    def rotate(self, theta: float, phi: float) -> "OrbitControls":
        """Rotate using angles (in radians). theta and phi are also known
        as azimuth and elevation.
        """
        # offset
        self._v.set(0, 0, self.distance).apply_quaternion(self.rotation)
        # to neutral up
        self._v.apply_quaternion(self._up_quat)
        # to spherical
        self._s.set_from_vector3(self._v)
        # apply delta
        self._s.theta -= theta
        self._s.phi -= phi
        # clip
        self._s.make_safe()
        # back to cartesian
        self._v.set_from_spherical(self._s)
        # back to camera up
        self._v.apply_quaternion(self._up_quat_inv)
        # compute new rotation
        self.rotation.set_from_rotation_matrix(
            self._m.look_at(self._v, self._origin, self.up)
        )
        return self

    def rotate_start(
        self,
        pos: Tuple[float, float],
        canvas_size: Tuple[float, float],
        camera: "Camera",
    ) -> "OrbitControls":
        """Start a rotation operation based (2D) screen coordinates."""
        self._rotate_info = {"last": pos}
        return self

    def rotate_stop(self) -> "OrbitControls":
        self._rotate_info = None
        return self

    def rotate_move(
        self, pos: Tuple[float, float], speed: float = 0.0175
    ) -> "OrbitControls":
        """Rotate, based on a (2D) screen location. Call rotate_start first.
        The speed is 1 degree per pixel by default.
        """
        if self._rotate_info is None:
            return
        delta = tuple((pos[i] - self._rotate_info["last"][i]) * speed for i in range(2))
        self.rotate(*delta)
        self._rotate_info["last"] = pos
        return self

    def zoom(self, multiplier: float) -> "OrbitControls":
        self.zoom_value = max(self.min_zoom, float(multiplier) * self.zoom_value)
        if self.zoom_changes_distance:
            self.distance = self._initial_distance / self.zoom_value
        return self

    def get_view(self) -> Tuple[Vector3, Vector3, float]:
        self._v.set(0, 0, self.distance).apply_quaternion(self.rotation).add(
            self.target
        )
        zoom = 1 if self.zoom_changes_distance else self.zoom_value
        return self.rotation, self._v, zoom

    def update_camera(self, camera: "Camera") -> "OrbitControls":
        rot, pos, zoom = self.get_view()
        camera.rotation.copy(rot)
        camera.position.copy(pos)
        camera.zoom = zoom
        return self

    def add_default_event_handlers(self, canvas, camera):
        """Apply the default interaction mechanism to a wgpu autogui canvas."""
        canvas.add_event_handler(
            lambda event: self.handle_event(event, canvas, camera),
            "pointer_down",
            "pointer_move",
            "pointer_up",
            "wheel",
        )

    def handle_event(self, event, canvas, camera):
        """Implements a default interaction mode that consumes wgpu autogui events
        (compatible with the jupyter_rfb event specification).
        """
        type = event["event_type"]
        if type == "pointer_down":
            xy = event["x"], event["y"]
            if event["button"] == 1:
                self.rotate_start(xy, canvas.get_logical_size(), camera)
            elif event["button"] == 2:
                self.pan_start(xy, canvas.get_logical_size(), camera)
        elif type == "pointer_up":
            if event["button"] == 1:
                self.rotate_stop()
            elif event["button"] == 2:
                self.pan_stop()
            canvas.request_draw()
        elif type == "pointer_move":
            xy = event["x"], event["y"]
            if 1 in event["buttons"]:
                self.rotate_move(xy),
            if 2 in event["buttons"]:
                self.pan_move(xy),
            canvas.request_draw()
        elif type == "wheel":
            xy = event["x"], event["y"]
            f = 2 ** (-event["dy"] * 0.0015)
            self.zoom(f)
            canvas.request_draw()
