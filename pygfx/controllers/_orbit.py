from typing import Tuple

import numpy as np
import pylinalg.func as la

from ..cameras import Camera
from ..utils.viewport import Viewport
from ._base import Controller, get_screen_vectors_in_world_cords


def get_axis_aligned_up_vector(up):
    ref_up, largest_dot = None, 0
    for up_vec in [(1, 0, 0), (0, 1, 0), (0, 0, 1)]:
        up_vec = np.array(up_vec)
        v = np.dot(up_vec, up)
        if v > largest_dot:
            ref_up, largest_dot = up_vec, v
    return ref_up


class OrbitController(Controller):
    """A controller to move a camera in an orbit around a center position.

    The direction of rotation is defined such that it feels like you're
    grabbing onto something in the foreground; if you move the mouse
    to the right, the objects in the foreground move to the right, and
    those in the background (on the opposite side of the center of
    rotation) move to the left.
    """

    def __init__(
        self,
        camera=None,
        *,
        auto_update: bool = True,
    ) -> None:
        super().__init__(camera)

        # The zoom value when doing quickzoom
        self._zoom_value = 4

        self.auto_update = auto_update

        # State info used during a pan or rotate operation
        self._pan_info = None
        self._rotate_info = None

    def pan(self, vec3) -> Controller:
        """Pan in 3D world coordinates."""
        if self._cameras:
            camera = self._cameras[0]
            self._pan(vec3, camera.get_state())
        return self

    def _pan(self, vec3, camera_state):
        # Get new position
        position = camera_state["position"]
        new_position = position + vec3
        # Apply new state to all cameras
        new_camera_state = {**camera_state, "position": new_position}
        for camera in self._cameras:
            camera.set_state(new_camera_state)

    def pan_start(
        self,
        pos: Tuple[float, float],
        viewport: Viewport,
        camera: Camera,
    ) -> Controller:
        """Start a panning operation based (2D) screen coordinates."""

        # Get camera state
        self._pan_info = camera_state = camera.get_state()
        position = camera_state["position"]
        rotation = camera_state["rotation"]
        dist = camera_state["dist"]

        # Get target, the reference location where translations should map to screen distances
        target = position + la.quaternion_rotate((0, 0, -dist), rotation)

        # Get the vectors that point in the axis direction
        scene_size = viewport.logical_size
        vecx, vecy = get_screen_vectors_in_world_cords(target, scene_size, camera)

        # Store pan info
        self._pan_info.update({"mouse_pos": pos, "vecx": vecx, "vecy": vecy})

        return self

    def pan_stop(self) -> Controller:
        self._pan_info = None
        return self

    def pan_move(self, pos: Tuple[float, float]) -> Controller:
        """Pan the center of rotation, based on a (2D) screen location. Call pan_start first."""
        if self._pan_info is None:
            return
        original_pos = self._pan_info["mouse_pos"]
        delta = pos[0] - original_pos[0], pos[1] - original_pos[1]
        vec3 = -self._pan_info["vecx"] * delta[0] + self._pan_info["vecy"] * delta[1]
        self._pan(vec3, self._pan_info)
        return self

    def rotate(self, delta_azimuth: float, delta_elevation: float) -> Controller:
        """Rotate using angles (in radians)."""
        if self._cameras:
            camera = self._cameras[0]
            self._rotate(delta_azimuth, delta_elevation, camera.get_state())
        return self

    def _rotate(self, delta_azimuth, delta_elevation, camera_state):
        position = camera_state["position"]
        rotation = camera_state["rotation"]
        up = camera_state["up"]
        dist = camera_state["dist"]

        # # Get a reference vector, that is orthogonal to up,
        # # and use that to calculate the azimuth.
        # aligned_up = get_axis_aligned_up_vector(up)
        # orthogonal_vec = np.cross(up, np.roll(aligned_up, 1))
        # azimuth = la.vector_angle_between(forward, orthogonal_vec)
        # -> we currently don't use the azimuth

        # Obtain the current elevation by taking the angle between forward and up
        forward = la.quaternion_rotate((0, 0, -1), rotation)
        elevation = la.vector_angle_between(forward, up) - 0.5 * np.pi

        # Apply boundaries to the elevation
        new_elevation = elevation + delta_elevation
        bounds = -89 * np.pi / 180, 89 * np.pi / 180
        if new_elevation < bounds[0]:
            delta_elevation = bounds[0] - elevation
        elif new_elevation > bounds[1]:
            delta_elevation = bounds[1] - elevation

        # todo: are these axii local or must they change as up changes?
        r_azimuth = la.quaternion_make_from_axis_angle((0, -1, 0), delta_azimuth)
        r_elevation = la.quaternion_make_from_axis_angle((-1, 0, 0), delta_elevation)

        # Get rotations
        rot1 = rotation
        rot2 = la.quaternion_multiply(
            r_azimuth, la.quaternion_multiply(rot1, r_elevation)
        )

        # Calculate new position
        pos1 = position
        pos2target1 = la.quaternion_rotate((0, 0, -dist), rot1)
        pos2target2 = la.quaternion_rotate((0, 0, -dist), rot2)
        pos2 = pos1 + pos2target1 - pos2target2

        # Apply new state to all cameras
        new_camera_state = {**camera_state, "position": pos2, "rotation": rot2}
        for camera in self._cameras:
            camera.set_state(new_camera_state)

        # Note that for ortho cameras, we also orbit around the scene,
        # even though it could be positioned at the center (i.e.
        # target). Doing it this way makes the code in the controllers
        # easier. The only downside I can think of is that the far plane
        # is now less far away but this effect is only 0.2%, since the
        # far plane is 500 * dist.

    def rotate_start(
        self,
        pos: Tuple[float, float],
        viewport: Viewport,
        camera: Camera,
    ) -> Controller:
        """Start a rotation operation based (2D) screen coordinates."""
        # Store the start-state, and the mouse pos
        self._rotate_info = camera.get_state()
        self._rotate_info["mouse_pos"] = pos
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
        delta_azimuth = (pos[0] - self._rotate_info["mouse_pos"][0]) * speed
        delta_elevation = (pos[1] - self._rotate_info["mouse_pos"][1]) * speed
        self._rotate(delta_azimuth, delta_elevation, self._rotate_info)
        return self

    def zoom(self, multiplier: float) -> Controller:
        # todo: maybe this can have a name similar to dist?
        # todo: though I'm not 100% sure about the dist prop either
        if self._cameras:
            # Get current state
            camera_state = self._cameras[0].get_state()
            position = camera_state["position"]
            rotation = camera_state["rotation"]
            dist = camera_state["dist"]
            # Get new dist and new position
            new_dist = dist * (1 / multiplier)
            target = position + la.quaternion_rotate((0, 0, -dist), rotation)
            new_position = target - la.quaternion_rotate((0, 0, -new_dist), rotation)
            # Apply new state to all cameras
            new_camera_state = {
                **camera_state,
                "position": new_position,
                "dist": new_dist,
            }
            for camera in self._cameras:
                camera.set_state(new_camera_state)

        return self

    def quick_zoom(self, zoom):
        """Use the camera's zoom prop to zoom in, as if using binoculars."""
        if self._cameras:
            # Get current state
            camera_state = self._cameras[0].get_state()
            # Apply new state to all cameras
            new_camera_state = {**camera_state, "zoom": zoom}
            for camera in self._cameras:
                camera.set_state(new_camera_state)

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
            elif event.button == 3:
                self.quick_zoom(self._zoom_value)
        elif type == "pointer_up":
            if event.button == 1:
                self.rotate_stop()
            elif event.button == 2:
                self.pan_stop()
            elif event.button == 3:
                self.quick_zoom(1)
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
        elif type == "key_down":
            if event.key == "Escape":
                pass  # todo: cancel camera action

    def show_object(self, camera, target):
        target_pos = camera.show_object(target, self.target.clone().sub(self._v), 1.2)
        self.look_at(camera.position, target_pos, camera.up)
        if self.zoom_changes_distance:
            self.zoom_value = self._initial_distance / self.distance
        else:
            # TODO: implement for orthographic camera
            raise NotImplementedError
