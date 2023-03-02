from typing import Tuple

import pylinalg as la

from ..cameras import Camera
from ..utils.viewport import Viewport
from ._base import Controller, get_screen_vectors_in_world_cords


class PanZoomController(Controller):
    """A controller to move a camera in a 2D plane."""

    def __init__(
        self, camera, min_zoom: float = 0.0001, auto_update: bool = True
    ) -> None:
        super().__init__(camera)

        self._min_zoom = min_zoom
        self.auto_update = auto_update

        # State info used during a pan operation
        self._pan_info = None

    # todo: the pan logic for the orbit and panzoom is exactly the same -> move to base class
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
        """Pan the camera, based on a (2D) screen location. Call pan_start first."""
        if self._pan_info is None:
            return
        original_pos = self._pan_info["mouse_pos"]
        delta = pos[0] - original_pos[0], pos[1] - original_pos[1]
        vec3 = -self._pan_info["vecx"] * delta[0] + self._pan_info["vecy"] * delta[1]
        self._pan(vec3, self._pan_info)
        return self

    # def zoom(self, multiplier: float) -> Controller:
    #     return self.zoom_to_point(multiplier, None, None)

    def zoom(self, multiplier: float) -> Controller:
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

    # todo: can we also have zoom-to-point in the orbit controller?
    def zoom_to_point(
        self,
        multiplier: float,
        pos,
        viewport,
    ) -> Controller:
        # Get Target
        camera_state = self._cameras[0].get_state()
        position = camera_state["position"]
        rotation = camera_state["rotation"]
        dist = camera_state["dist"]
        target = position + la.quaternion_rotate((0, 0, -dist), rotation)

        # Get viewport info
        x, y, w, h = viewport.rect
        offset = x, y
        size = w, h

        # Calculate pan such that what was previously under the mouse is again under the mouse.
        vecx, vecy = get_screen_vectors_in_world_cords(target, size, self._cameras[0])
        delta = tuple(pos[i] - offset[i] - size[i] / 2 for i in (0, 1))
        delta1 = vecx * delta[0] - vecy * delta[1]
        delta2 = delta1 / multiplier

        # Apply
        self.zoom(multiplier)
        self.pan(delta1 - delta2)

        return self

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
            self.zoom_to_point(f, xy, viewport)
            if self.auto_update:
                viewport.renderer.request_draw()

    # def show_object(self, camera, target):
    #     # TODO: implement for perspective camera
    #     if not isinstance(camera, OrthographicCamera):
    #         raise NotImplementedError
    #
    #     target_pos = camera.show_object(target, self.target.clone().sub(self._v), 1.2)
    #     self.look_at(camera.position, target_pos, camera.up)
    #     bsphere = target.get_world_bounding_sphere()
    #     if bsphere is not None:
    #         radius = bsphere[3]
    #         center_world_coord = Vector3(0, 0, 0).unproject(camera)
    #         right_world_coord = Vector3(1, 0, 0).unproject(camera)
    #         top_world_coord = Vector3(0, 1, 0).unproject(camera)
    #
    #         min_distance = min(
    #             right_world_coord.distance_to(center_world_coord),
    #             top_world_coord.distance_to(center_world_coord),
    #         )
    #         self.zoom_value = min_distance / radius * self.zoom_value
