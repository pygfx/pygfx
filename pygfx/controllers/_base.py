from typing import Tuple, Union
import numpy as np
import pylinalg as la

from ..utils.viewport import Viewport
from ..renderers import Renderer
from ..cameras import Camera, PerspectiveCamera
from ..cameras._perspective import fov_distance_factor


class Controller:
    """The base camera controller.

    The purpose of a controller is to provide an API to control a camera,
    and to convert user (mouse) events into camera adjustments

    Parameters
    ----------
    camera: Camera
        The camera to control (optional). Must be a perspective or orthographic camera.
    enabled: bool
        Whether the controller is enabled (i.e. responds to events).
    auto_update : bool
        Whether the controller requests a new draw when the camera has changed.
    register_events: Renderer or Viewport
        If given and not None, will call ``.register_events()``..
    """

    def __init__(
        self, camera=None, *, enabled=True, auto_update=True, register_events=None
    ):
        self._cameras = []
        if camera is not None:
            self.add_camera(camera)
        self.enabled = enabled
        self.auto_update = auto_update
        if register_events is not None:
            self.register_events(register_events)

    @property
    def cameras(self):
        """A tuple with the cameras under control, in the order that they were added."""
        return tuple(self._cameras)

    def add_camera(self, camera):
        """Add a camera to control."""
        if not isinstance(camera, Camera):
            raise TypeError("Controller.add_camera expects a Camera object.")
        if not isinstance(camera, PerspectiveCamera):
            raise TypeError(
                "Controller.add_camera expects a perspective or orthographic camera."
            )
        self.remove_camera(camera)
        self._cameras.append(camera)

    def remove_camera(self, camera):
        """Remove a camera from the list of cameras to control."""
        if not isinstance(camera, Camera):
            raise TypeError("Controller.remove_camera expects a Camera object.")
        while camera in self._cameras:
            self._cameras.remove(camera)

    @property
    def enabled(self):
        """Whether the controller responds to events."""
        return self._enabled

    @enabled.setter
    def enabled(self, value):
        self._enabled = bool(value)

    @property
    def auto_update(self):
        """Whether the controller automatically requests a new draw at the canvas."""
        return self._auto_update

    @auto_update.setter
    def auto_update(self, value):
        self._auto_update = bool(value)

    def _get_target_vec(self, camera_state, **kwargs):
        """Method used by the controler implementations to determine the "target"."""
        rotation = kwargs.get("rotation", camera_state["rotation"])
        extent = 0.5 * (camera_state["width"] + camera_state["height"])
        extent = kwargs.get("extent", extent)
        fov = kwargs.get("fov", camera_state.get("fov"))

        distance = fov_distance_factor(fov) * extent
        return la.quaternion_rotate((0, 0, -distance), rotation)

    def register_events(self, viewport_or_renderer: Union[Viewport, Renderer]):
        """Apply the default interaction mechanism to a wgpu autogui canvas.
        Needs either a viewport or renderer.
        """
        viewport = Viewport.from_viewport_or_renderer(viewport_or_renderer)
        viewport.renderer.add_event_handler(
            lambda event: self.handle_event(event, viewport),
            "pointer_down",
            "pointer_move",
            "pointer_up",
            "key_down",
            "key_up",
            "wheel",
        )

    def handle_event(self, event, viewport, camera):
        raise NotImplementedError()

    def add_default_event_handlers(self, *args):
        raise DeprecationWarning(
            "controller.add_default_event_handlers(viewport, camera) -> controller.register_events(viewport)"
        )

    def update_camera(self, *args):
        raise DeprecationWarning("controller.update_camera() is no longer necessary")


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
