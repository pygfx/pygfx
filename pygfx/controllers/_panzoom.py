from typing import Tuple

from ..cameras import Camera, OrthographicCamera
from ..utils.viewport import Viewport
from ..linalg import Vector3, Matrix4, Quaternion
from ._base import Controller, get_screen_vectors_in_world_cords


class PanZoomController(Controller):
    """A class implementing two-dimensional pan-zoom camera controller."""

    def __init__(
        self,
        eye: Vector3 = None,
        target: Vector3 = None,
        up: Vector3 = None,
        zoom: float = 1.0,
        min_zoom: float = 0.0001,
        auto_update: bool = True,
    ) -> None:
        super().__init__()
        self.rotation = Quaternion()
        self.target = Vector3()
        self.up = Vector3()
        if eye is None:
            eye = Vector3(0, 0, 0)
        if target is None:
            target = Vector3(eye.x, eye.y, eye.z - 100)
        if up is None:
            up = Vector3(0.0, 1.0, 0.0)
        self.zoom_value = zoom
        self.min_zoom = min_zoom
        self.auto_update = True

        # State info used during a pan or rotate operation
        self._pan_info = None

        # Temp objects (to avoid garbage collection)
        self._m = Matrix4()
        self._v = Vector3()

        # Initialize orientation
        self.look_at(eye, target, up)

        # Save initial state
        self.save_state()

    def save_state(self):
        self._saved_state = {
            "rotation": self.rotation.clone(),
            "distance": self.distance,
            "target": self.target.clone(),
            "up": self.up.clone(),
            "zoom_value": self.zoom_value,
            "min_zoom": self.min_zoom,
        }
        return self._saved_state

    def load_state(self, state=None):
        state = state or self._saved_state
        self.rotation = state["rotation"].clone()
        self.distance = state["distance"]
        self.target = state["target"].clone()
        self.up = state["up"].clone()
        self.zoom_value = state["zoom_value"]
        self.min_zoom = state["min_zoom"]

    def look_at(self, eye: Vector3, target: Vector3, up: Vector3) -> Controller:
        self.distance = eye.distance_to(target)
        self.target.copy(target)
        self.up.copy(up)
        self.rotation.set_from_rotation_matrix(self._m.look_at(eye, target, up))
        return self

    def pan(self, vec3: Vector3) -> Controller:
        """Pan in 3D world coordinates."""
        self.target.add(vec3)
        return self

    def pan_start(
        self,
        pos: Tuple[float, float],
        viewport: Viewport,
        camera: Camera,
    ) -> Controller:
        # Using this function may be a bit overkill. We can also simply
        # get the ortho cameras world_size (camera.visible_world_size).
        # However, now the panzoom controller work with a perspecive camera ...
        scene_size = viewport.logical_size
        vecx, vecy = get_screen_vectors_in_world_cords(self.target, scene_size, camera)
        self._pan_info = {"last": pos, "vecx": vecx, "vecy": vecy}
        return self

    def pan_stop(self) -> Controller:
        self._pan_info = None
        return self

    def pan_move(self, pos: Tuple[float, float]) -> Controller:
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

    def zoom(self, multiplier: float) -> Controller:
        self.zoom_value = max(self.min_zoom, float(multiplier) * self.zoom_value)
        return self

    def zoom_to_point(
        self,
        multiplier: float,
        pos: Tuple[float, float],
        viewport: Viewport,
        camera: Camera,
    ) -> Controller:

        x, y, w, h = viewport.rect
        offset = x, y
        size = w, h

        # Apply zoom
        zoom_old = self.zoom_value
        self.zoom(multiplier)
        zoom_ratio = zoom_old / self.zoom_value  # usually == multiplier

        # Now pan such that what was previously under the mouse is again under the mouse.
        vecx, vecy = get_screen_vectors_in_world_cords(self.target, size, camera)
        delta = tuple(pos[i] - offset[i] - size[i] / 2 for i in (0, 1))
        delta1 = vecx.multiply_scalar(delta[0]).add(vecy.multiply_scalar(-delta[1]))
        delta2 = delta1.clone().multiply_scalar(zoom_ratio)
        self.pan(delta1.sub(delta2))
        return self

    def get_view(self) -> Tuple[Vector3, Vector3, float]:
        self._v.set(0, 0, self.distance).apply_quaternion(self.rotation).add(
            self.target
        )
        return self.rotation, self._v, self.zoom_value

    def handle_event(self, event, viewport, camera):
        """Implements a default interaction mode that consumes wgpu autogui events
        (compatible with the jupyter_rfb event specification).
        """
        type = event.type
        if type == "pointer_down" and viewport.is_inside(event.x, event.y):
            if event.button == 1:
                xy = event.x, event.y
                self.pan_start(xy, viewport, camera)
        elif type == "pointer_up":
            if event.button == 1:
                self.pan_stop()
        elif type == "pointer_move":
            if 1 in event.buttons:
                xy = event.x, event.y
                self.pan_move(xy)
                if self.auto_update:
                    viewport.renderer.request_draw()
        elif type == "wheel" and viewport.is_inside(event.x, event.y):
            xy = event.x, event.y
            f = 2 ** (-event.dy * 0.0015)
            self.zoom_to_point(f, xy, viewport, camera)
            if self.auto_update:
                viewport.renderer.request_draw()

    def show_object(self, camera, target):
        # TODO: implement for perspective camera
        if not isinstance(camera, OrthographicCamera):
            raise NotImplementedError

        target_pos = camera.show_object(target, self.target.clone().sub(self._v), 1.2)
        self.look_at(camera.position, target_pos, camera.up)
        bsphere = target.get_world_bounding_sphere()
        if bsphere is not None:
            radius = bsphere[3]
            center_world_coord = Vector3(0, 0, 0).unproject(camera)
            right_world_coord = Vector3(1, 0, 0).unproject(camera)
            top_world_coord = Vector3(0, 1, 0).unproject(camera)

            min_distance = min(
                right_world_coord.distance_to(center_world_coord),
                top_world_coord.distance_to(center_world_coord),
            )
            self.zoom_value = min_distance / radius * self.zoom_value
