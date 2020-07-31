from ..linalg import Vector3, Matrix4, Quaternion, Spherical


# todo: maybe make an OrbitOrthoControls for ortho cameras, instead of this zoom param?


class OrbitControls:
    _m = Matrix4()
    _v = Vector3()
    _origin = Vector3()
    _orbit_up = Vector3(0, 1, 0)
    _s = Spherical()

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
        self.look_at(eye, target, up)
        self.zoom_changes_distance = bool(zoom_changes_distance)
        self.zoom_value = 1
        self.min_zoom = min_zoom
        self._initial_distance = self.distance

    def look_at(self, eye: Vector3, target: Vector3, up: Vector3) -> "OrbitControls":
        self.distance = eye.distance_to(target)
        self.target = target
        self.up = up
        self.rotation.set_from_rotation_matrix(self._m.look_at(eye, target, up))
        self._up_quat = Quaternion().set_from_unit_vectors(self.up, self._orbit_up)
        self._up_quat_inv = self._up_quat.clone().inverse()
        return self

    def pan(self, x: float, y: float) -> "OrbitControls":
        self._v.set(x, -y, 0).apply_quaternion(self.rotation)
        self.target.sub(self._v)
        return self

    def rotate(self, theta: float, phi: float) -> "OrbitControls":
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

    def zoom(self, multiplier: float) -> "OrbitControls":
        self.zoom_value = max(self.min_zoom, float(multiplier) * self.zoom_value)
        if self.zoom_changes_distance:
            self.distance = self._initial_distance * self.zoom_value
        return self

    def get_view(self) -> (Vector3, Vector3):
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
