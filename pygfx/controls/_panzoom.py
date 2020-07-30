from typing import Tuple

from ..linalg import Vector3, Matrix4, Quaternion, Spherical


class PanZoomControls:
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
        zoom: float = 1.0,
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
        self.zoom_ = zoom
        self.min_zoom = min_zoom

    def look_at(self, eye: Vector3, target: Vector3, up: Vector3) -> "PanZoomControls":
        self.distance = eye.distance_to(target)
        self.target = target
        self.up = up
        self.rotation.set_from_rotation_matrix(self._m.look_at(eye, target, up))
        self._up_quat = Quaternion().set_from_unit_vectors(self.up, self._orbit_up)
        self._up_quat_inv = self._up_quat.clone().inverse()
        return self

    def pan(self, x: float, y: float) -> "PanZoomControls":
        self._v.set(x, -y, 0).apply_quaternion(self.rotation)
        self.target.sub(self._v)
        return self

    def zoom(self, delta: float) -> "PanZoomControls":
        if self.zoom_ < 1.0:
            delta *= self.zoom_
        self.zoom_ += delta
        if self.zoom_ < self.min_zoom:
            self.zoom_ = self.min_zoom
        return self

    def zoom_to_point(
        self,
        delta: float,
        mouse: Tuple[float, float],
        canvas: Tuple[float, float],
        view: Tuple[float, float],
    ) -> "PanZoomControls":
        # convert current mouse position to fractions relative to widget center
        # (fracpos x and y range becomes [-50%, 50%])
        fracpos = tuple((mouse[i] - canvas[i] * 0.5) / canvas[i] for i in range(2))
        # this gives us the relative position of the mouse in viewport space
        relpos_old = tuple(fracpos[i] * view[i] for i in range(2))
        # now apply the zoom delta
        zoom_old = self.zoom_
        self.zoom(delta)
        # compute the new viewport dimensions
        zoom_ratio = zoom_old / self.zoom_
        view_new = tuple(view[i] * zoom_ratio for i in range(2))
        # and the new relative position of the mouse in viewport space
        relpos = tuple(fracpos[i] * view_new[i] for i in range(2))
        # finally compute the delta and pan accordingly to compensate
        # such that the point under the mouse stays under the mouse
        delta = tuple(relpos[i] - relpos_old[i] for i in range(2))
        self.pan(*delta)
        return self

    def get_view(self) -> (Vector3, Vector3, float):
        self._v.set(0, 0, self.distance).apply_quaternion(self.rotation).add(
            self.target
        )
        return self.rotation, self._v, self.zoom_

    def update_camera(self, camera: "Camera") -> "PanZoomControls":
        rot, pos, zoom = self.get_view()
        camera.rotation.copy(rot)
        camera.position.copy(pos)
        camera.zoom = zoom
        return self
