from typing import Tuple, Union

import pylinalg as la
import numpy as np

from ..utils.viewport import Viewport
from ..renderers import Renderer
from ..cameras import Camera, OrthographicCamera, PerspectiveCamera


class Controller:
    """The base camera controller.

    The purpose of a controller is to provide an API to control a camera,
    and to convert user (mouse) events into camera adjustments.
    """

    def __init__(self, camera=None):
        self._cameras = []
        if camera is not None:
            self.add_camera(camera)

    @property
    def cameras(self):
        """A tuple with the cameras under control, in the order that they were added."""
        return tuple(self._cameras)

    def add_camera(self, camera):
        """Add a camera to control."""
        if not isinstance(camera, Camera):
            raise TypeError("Controller.add_camera expects a Camera object.")
        if not isinstance(camera, (OrthographicCamera, PerspectiveCamera)):
            raise TypeError(
                "Controller.add_camera expects an orthographic or perspective camera."
            )
        self.remove_camera(camera)
        self._cameras.append(camera)

    def remove_camera(self, camera):
        """Remove a camera from the list of cameras to control."""
        if not isinstance(camera, Camera):
            raise TypeError("Controller.remove_camera expects a Camera object.")
        while camera in self._cameras:
            self._cameras.remove(camera)

    def _get_target_vec(self, camera_state, **kwargs):
        rotation = kwargs.get("rotation", camera_state["rotation"])
        extent = kwargs.get("dist", camera_state["dist"])
        fov = kwargs.get("fov", camera_state.get("fov", None))

        if fov:
            fov_rad = fov * np.pi / 180
            distance = 0.5 * extent / np.tan(0.5 * fov_rad)
        else:
            distance = extent * 1.0

        return la.quaternion_rotate((0, 0, -distance), rotation)

    def handle_event(self, event, viewport, camera):
        raise NotImplementedError()

    # def show_object(self, camera, target):
    #     raise NotImplementedError()

    def add_default_event_handlers(self, viewport: Union[Viewport, Renderer]):
        """Apply the default interaction mechanism to a wgpu autogui canvas.
        Needs either a viewport or renderer.
        """
        viewport = Viewport.from_viewport_or_renderer(viewport)
        viewport.renderer.add_event_handler(
            lambda event: self.handle_event(event, viewport),
            "pointer_down",
            "pointer_move",
            "pointer_up",
            "key_down",
            "key_up",
            "wheel",
        )


def get_screen_vectors_in_world_cords(
    center_world: Tuple[float, float, float],
    scene_size: Tuple[float, float],
    camera: Camera,
):
    """Given a reference center location (in 3D world coordinates)
    Get the vectors corresponding to the x and y direction in screen coordinates.
    These vectors are scaled so that they can simply be multiplied with the
    delta x and delta y.
    """

    # Linalg conv
    camera_world = camera.matrix_world.to_ndarray()
    camera_world_inverse = camera.matrix_world_inverse.to_ndarray()
    camera_projection = camera.projection_matrix.to_ndarray()
    camera_projection_inverse = camera.projection_matrix_inverse.to_ndarray()

    # Get center location on screen
    center = la.vector_apply_matrix(
        la.vector_apply_matrix(center_world, camera_world_inverse), camera_projection
    )

    # Get vectors
    screen_dist = 100
    pos1 = la.vector_apply_matrix(
        la.vector_apply_matrix((screen_dist, 0, center[2]), camera_projection_inverse),
        camera_world,
    )
    pos2 = la.vector_apply_matrix(
        la.vector_apply_matrix((0, screen_dist, center[2]), camera_projection_inverse),
        camera_world,
    )

    # Return scaled
    return pos1 * 0.02 / scene_size[0], pos2 * 0.02 / scene_size[1]
