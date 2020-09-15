from typing import Tuple

from ..linalg import Vector3, Matrix4, Quaternion
from ._orbit import get_screen_vectors_in_world_cords


class PanZoomControls:
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
        self.zoom_value = zoom
        self.min_zoom = min_zoom

        # State info used during a pan or rotate operation
        self._pan_info = None

        # Temp objects (to avoid garbage collection)
        self._m = Matrix4()
        self._v = Vector3()

        # Initialize orientation
        self.look_at(eye, target, up)

    def look_at(self, eye: Vector3, target: Vector3, up: Vector3) -> "PanZoomControls":
        self.distance = eye.distance_to(target)
        self.target = target
        self.up = up
        self.rotation.set_from_rotation_matrix(self._m.look_at(eye, target, up))
        return self

    def pan(self, vec3: Vector3) -> "PanZoomControls":
        """Pan in 3D world coordinates."""
        self.target.add(vec3)
        return self

    def pan_start(
        self,
        pos: Tuple[float, float],
        canvas_size: Tuple[float, float],
        camera: "Camera",
    ) -> "PanZoomControls":
        # Using this function may be a bit overkill. We can also simply
        # get the ortho cameras world_size (camera.visible_world_size).
        # However, now the panzoom controls work with a perspecive camera ...
        vecx, vecy = get_screen_vectors_in_world_cords(self.target, canvas_size, camera)
        self._pan_info = {"last": pos, "vecx": vecx, "vecy": vecy}
        return self

    def pan_stop(self) -> "PanZoomControls":
        self._pan_info = None
        return self

    def pan_move(self, pos: Tuple[float, float]) -> "PanZoomControls":
        """Pan the camera, based on a (2D) screen location. Call pan_start first."""
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

    def zoom(self, multiplier: float) -> "PanZoomControls":
        self.zoom_value = max(self.min_zoom, float(multiplier) * self.zoom_value)
        return self

    def zoom_to_point(
        self,
        multiplier: float,
        pos: Tuple[float, float],
        canvas_size: Tuple[float, float],
        camera: "Camera",
    ) -> "PanZoomControls":

        # Apply zoom
        zoom_old = self.zoom_value
        self.zoom(multiplier)
        zoom_ratio = zoom_old / self.zoom_value  # usually == multiplier

        # Now pan such that what was previously under the mouse is again under the mouse.
        vecx, vecy = get_screen_vectors_in_world_cords(self.target, canvas_size, camera)
        delta = tuple(pos[i] - canvas_size[i] / 2 for i in (0, 1))
        delta1 = vecx.multiply_scalar(delta[0]).add(vecy.multiply_scalar(-delta[1]))
        delta2 = delta1.clone().multiply_scalar(zoom_ratio)
        self.pan(delta1.sub(delta2))
        return self

    def get_view(self) -> Tuple[Vector3, Vector3, float]:
        self._v.set(0, 0, self.distance).apply_quaternion(self.rotation).add(
            self.target
        )
        return self.rotation, self._v, self.zoom_value

    def update_camera(self, camera: "Camera") -> "PanZoomControls":
        rot, pos, zoom = self.get_view()
        camera.rotation.copy(rot)
        camera.position.copy(pos)
        camera.zoom = zoom
        return self
