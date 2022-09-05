from typing import Tuple, Union

from ..linalg import Vector3
from ..utils.viewport import Viewport
from ..renderers import Renderer
from ..cameras import Camera


class Controller:
    """Base camera controller."""

    def get_view(self) -> Tuple[Vector3, Vector3, float]:
        """
        Returns view parameters with which a camera can be updated.

        Returns:
            rotation: Vector3
                Rotation of camera
            position: Vector3
                Position of camera
            zoom: float
                Zoom value for camera
        """
        raise NotImplementedError()

    def handle_event(self, event, viewport, camera):
        raise NotImplementedError()

    def save_state(self):
        raise NotImplementedError()

    def load_state(self, state=None):
        raise NotImplementedError()

    def show_object(self, camera, target):
        raise NotImplementedError()

    def update_camera(self, camera: Camera) -> "Controller":
        """Update the transform of the camera with the internal transform."""
        rot, pos, zoom = self.get_view()
        camera.rotation.copy(rot)
        camera.position.copy(pos)
        camera.zoom = zoom
        return self

    def add_default_event_handlers(
        self, viewport: Union[Viewport, Renderer], camera: Camera
    ):
        """Apply the default interaction mechanism to a wgpu autogui canvas.
        Needs either a viewport or renderer, pus the camera.
        """
        viewport = Viewport.from_viewport_or_renderer(viewport)
        viewport.renderer.add_event_handler(
            lambda event: self.handle_event(event, viewport, camera),
            "pointer_down",
            "pointer_move",
            "pointer_up",
            "wheel",
        )


def get_screen_vectors_in_world_cords(
    center_world: Vector3, scene_size: Tuple[float, float], camera: Camera
) -> Tuple[Vector3, Vector3]:
    """Given a reference center location (in 3D world coordinates)
    Get the vectors corresponding to the x and y direction in screen coordinates.
    These vectors are scaled so that they can simply be multiplied with the
    delta x and delta y.
    """
    center = center_world.clone().project(camera)
    pos1 = Vector3(100, 0, center.z).unproject(camera)
    pos2 = Vector3(0, 100, center.z).unproject(camera)
    pos1.multiply_scalar(0.02 / scene_size[0])
    pos2.multiply_scalar(0.02 / scene_size[1])
    return pos1, pos2  # now they're vecs, really
