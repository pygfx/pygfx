from typing import Tuple, Union

from ..linalg import Vector3
from ..utils.viewport import Viewport
from ..renderers import Renderer
from ..cameras import Camera


class Controller:
    """The base camera controller.

    The purpose of a controller is to provide an API to control a camera,
    and to convert user (mouse) events into camera adjustments.
    """

    def __init__(self, camera=None):
        self._cameras = []
        if camera is not None:
            self.add_camera(camera)

    def cameras(self):
        """A tuple with the cameras under control, in the order that they were added."""
        return tuple(self._cameras)

    def add_camera(self, camera):
        """Add a camera to control."""
        if not isinstance(camera, Camera):
            raise TypeError("Controller.add_camera expects a Camera object.")
        self.remove_camera(camera)
        self._cameras.append(camera)

    def remove_camera(self, camera):
        """Remove a camera from the list of cameras to control."""
        if not isinstance(camera, Camera):
            raise TypeError("Controller.remove_camera expects a Camera object.")
        while camera in self._cameras:
            self._cameras.remove(camera)

    def handle_event(self, event, viewport, camera):
        raise NotImplementedError()

    # todo: these must all be on the camera object
    # def get_view(self)
    #      ...
    # def save_state(self):
    #     raise NotImplementedError()
    #
    # def load_state(self, state=None):
    #     raise NotImplementedError()
    #
    # def show_object(self, camera, target):
    #     raise NotImplementedError()

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
