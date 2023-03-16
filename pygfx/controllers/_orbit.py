from typing import Tuple
import numpy as np
import pylinalg as la

from ..cameras import Camera
from ..utils.viewport import Viewport
from ._base import Controller, get_screen_vectors_in_world_cords


class OrbitController(Controller):
    """A class implementing an orbit camera controller, where the camera is
    rotated around a center position (orbiting around it).
    """

    def __init__(
        self,
        eye: np.ndarray = None,
        target: np.ndarray = None,
        up: np.ndarray = None,
        *,
        zoom_changes_distance=True,
        min_zoom: float = 0.0001,
        auto_update: bool = True,
    ) -> None:
        super().__init__()
        self.rotation = np.array((0, 0, 0, 0), dtype=float)
        self.target = np.array((0, 0, 0), dtype=float)
        self.up = np.array((0, 0, 0), dtype=float)
        if eye is None:
            eye = np.array((50.0, 50.0, 50.0), dtype=float)
        if target is None:
            target = np.array((0, 0, 0), dtype=float)
        if up is None:
            up = np.array((0.0, 1.0, 0.0), dtype=float)
        self.zoom_changes_distance = bool(zoom_changes_distance)
        self.zoom_value = 1
        self.min_zoom = min_zoom
        self.auto_update = auto_update

        # State info used during a pan or rotate operation
        self._pan_info = None
        self._rotate_info = None

        # Temp objects (to avoid garbage collection)
        self._m = np.eye(4, dtype=float)
        self._v = np.array((0, 0, 0), dtype=float)
        self._origin = np.array((0, 0, 0), dtype=float)
        self._orbit_up = np.array((0, 1, 0), dtype=float)

        # Initialize orientation
        self.look_at(eye, target, up)
        self._initial_distance = self.distance

        # Save initial state
        self.save_state()

    def save_state(self):
        self._saved_state = {
            "rotation": self.rotation.clone(),
            "distance": self.distance,
            "target": self.target.clone(),
            "up": self.up.clone(),
            "zoom_changes_distance": self.zoom_changes_distance,
            "zoom_value": self.zoom_value,
            "min_zoom": self.min_zoom,
            "initial_distance": self._initial_distance,
        }
        return self._saved_state

    def load_state(self, state=None):
        state = state or self._saved_state
        self.rotation = state["rotation"].clone()
        self.distance = state["distance"]
        self.target = state["target"].clone()
        self.up = state["up"].clone()
        self.zoom_changes_distance = state["zoom_changes_distance"]
        self.zoom_value = state["zoom_value"]
        self.min_zoom = state["min_zoom"]
        self.initial_distance = state["initial_distance"]
        self._update_up_quats()

    def _update_up_quats(self):
        self._up_quat = la.quaternion_make_from_unit_vectors(self.up, self._orbit_up)
        self._up_quat_inv = self._up_quat.clone().inverse()

    def look_at(
        self, eye: np.ndarray, target: np.ndarray, up: np.ndarray
    ) -> Controller:
        self.distance = la.vector_distance_between(eye, target)
        self.target = target
        self.up = up
        self.rotation = la.matrix_make_look_at(eye, target, up)
        self._update_up_quats()

        return self

    def pan(self, vec3) -> Controller:
        """Pan in 3D world coordinates."""
        self.target.add(vec3)
        return self

    def pan_start(
        self,
        pos: Tuple[float, float],
        viewport: Viewport,
        camera: Camera,
    ) -> Controller:
        """Start a panning operation based (2D) screen coordinates."""
        scene_size = viewport.logical_size
        vecx, vecy = get_screen_vectors_in_world_cords(self.target, scene_size, camera)
        self._pan_info = {"last": pos, "vecx": vecx, "vecy": vecy}
        return self

    def pan_stop(self) -> Controller:
        self._pan_info = None
        return self

    def pan_move(self, pos: Tuple[float, float]) -> Controller:
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

    def rotate(self, theta: float, phi: float) -> Controller:
        """Rotate using angles (in radians). theta and phi are also known
        as azimuth and elevation.
        """

        offset = np.array((0, 0, self.distance))
        offset = la.vector_apply_quaternion(offset, self.rotation)
        offset = la.vector_apply_quaternion(offset, self._up_quat)
        offset = la.vector_euclidean_to_spherical(offset)
        offset -= (0, theta, phi)
        offset = la.vector_make_spherical_safe(offset)
        offset = la.vector_spherical_to_euclidean(offset)
        offset = la.vector_apply_quaternion(offset, self._up_quat_inv)

        self.rotation = la.matrix_make_look_at(offset, self._origin, self.up)

        return self

    def rotate_start(
        self,
        pos: Tuple[float, float],
        viewport: Viewport,
        camera: Camera,
    ) -> Controller:
        """Start a rotation operation based (2D) screen coordinates."""
        self._rotate_info = {"last": pos}
        return self

    def rotate_stop(self) -> Controller:
        self._rotate_info = None
        return self

    def rotate_move(
        self, pos: Tuple[float, float], speed: float = 0.0175
    ) -> Controller:
        """Rotate, based on a (2D) screen location. Call rotate_start first.
        The speed is 1 degree per pixel by default.
        """
        if self._rotate_info is None:
            return
        delta = tuple((pos[i] - self._rotate_info["last"][i]) * speed for i in range(2))
        self.rotate(*delta)
        self._rotate_info["last"] = pos
        return self

    def zoom(self, multiplier: float) -> Controller:
        self.zoom_value = max(self.min_zoom, float(multiplier) * self.zoom_value)
        if self.zoom_changes_distance:
            self.distance = self._initial_distance / self.zoom_value
        return self

    def get_view(self):
        """
        Returns view parameters with which a camera can be updated.

        Returns:
            rotation: ndarray, [4]
                Rotation of camera
            position: ndarray, [3]
                Position of camera
            zoom: float
                Zoom value for camera
        """
        self._v.set(0, 0, self.distance).apply_quaternion(self.rotation).add(
            self.target
        )
        zoom = 1 if self.zoom_changes_distance else self.zoom_value
        return self.rotation, self._v, zoom

    def handle_event(self, event, viewport, camera):
        """Implements a default interaction mode that consumes wgpu autogui events
        (compatible with the jupyter_rfb event specification).
        """
        type = event.type
        if type == "pointer_down" and viewport.is_inside(event.x, event.y):
            xy = event.x, event.y
            if event.button == 1:
                self.rotate_start(xy, viewport, camera)
            elif event.button == 2:
                self.pan_start(xy, viewport, camera)
        elif type == "pointer_up":
            if event.button == 1:
                self.rotate_stop()
            elif event.button == 2:
                self.pan_stop()
        elif type == "pointer_move":
            xy = event.x, event.y
            if 1 in event.buttons:
                self.rotate_move(xy),
                if self.auto_update:
                    viewport.renderer.request_draw()
            if 2 in event.buttons:
                self.pan_move(xy),
                if self.auto_update:
                    viewport.renderer.request_draw()
        elif type == "wheel" and viewport.is_inside(event.x, event.y):
            xy = event.x, event.y
            d = event.dy or event.dx
            f = 2 ** (-d * 0.0015)
            self.zoom(f)
            if self.auto_update:
                viewport.renderer.request_draw()

    def show_object(self, camera, target):
        target_pos = camera.show_object(target, self.target.clone().sub(self._v), 1.2)
        self.look_at(camera.position, target_pos, camera.up)
        if self.zoom_changes_distance:
            self.zoom_value = self._initial_distance / self.distance
        else:
            # TODO: implement for orthographic camera
            raise NotImplementedError


class OrbitOrthoController(OrbitController):
    """An orbit controller for orthographic camera's (zooming is done
    by projection, instead of changing the distance.
    """

    def __init__(
        self,
        eye=None,
        target=None,
        up=None,
        *,
        min_zoom: float = 0.0001,
        auto_update: bool = True,
    ):
        super().__init__(
            eye,
            target,
            up,
            zoom_changes_distance=False,
            min_zoom=min_zoom,
            auto_update=auto_update,
        )
