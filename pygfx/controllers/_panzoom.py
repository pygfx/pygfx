from typing import Tuple

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

    def pan(self, vec3) -> Controller:
        """Pan in 3D world coordinates."""
        # Cannot pan in "controller space" (i.e. using 2D coords) because we
        # need the screen size for get_screen_vectors_in_world_cords, and
        # we only have that via an event.
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

        # Get target, the reference location where translations should map to screen distances
        target = position + self._get_target_vec(camera_state)

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

    def zoom(self, multiplier: float) -> Controller:
        if self._cameras:
            # Get current state
            camera_state = self._cameras[0].get_state()
            position = camera_state["position"]
            dist = camera_state["dist"]
            # Get new dist and new position
            new_dist = dist * (1 / multiplier)
            pos2target1 = self._get_target_vec(camera_state, dist=dist)
            pos2target2 = self._get_target_vec(camera_state, dist=new_dist)
            new_position = position + pos2target1 - pos2target2
            # Apply new state to all cameras
            new_camera_state = {
                **camera_state,
                "position": new_position,
                "dist": new_dist,
            }
            for camera in self._cameras:
                camera.set_state(new_camera_state)
        return self

    def _get_panning_to_compensate_zoom(self, multiplier, pos, viewport):
        # Get Target
        camera_state = self._cameras[0].get_state()
        position = camera_state["position"]
        target = position + self._get_target_vec(camera_state)

        # Get viewport info
        x, y, w, h = viewport.rect
        offset = x, y
        size = w, h

        # Calculate pan such that what was previously under the mouse is again under the mouse.
        vecx, vecy = get_screen_vectors_in_world_cords(target, size, self._cameras[0])
        delta = tuple(pos[i] - offset[i] - size[i] / 2 for i in (0, 1))
        delta1 = vecx * delta[0] - vecy * delta[1]
        delta2 = delta1 / multiplier

        return delta1 - delta2

    def zoom_to_point(
        self,
        multiplier: float,
        pos,
        viewport,
    ) -> Controller:
        self.zoom(multiplier)
        self.pan(self._get_panning_to_compensate_zoom(multiplier, pos, viewport))
        return self

    def zoom_start(
        self,
        pos: Tuple[float, float],
    ) -> Controller:
        # Get camera state
        self._zoom_info = self._cameras[0].get_state()
        # Store pan info
        self._zoom_info.update({"mouse_pos": pos})
        return self

    def zoom_stop(self) -> Controller:
        self._zoom_info = None
        return self

    def zoom_move(self, pos: Tuple[float, float]) -> Controller:
        if self._zoom_info is None:
            return

        # Get state
        camera_state = self._zoom_info
        original_pos = camera_state["mouse_pos"]
        maintain_aspect = camera_state.get("maintain_aspect", True)

        # Calculate zoom factors
        delta = pos[0] - original_pos[0], pos[1] - original_pos[1]
        fx = 2 ** (-delta[0] * 0.01)
        fy = 2 ** (delta[1] * 0.01)

        # Apply
        if maintain_aspect:
            # Use dist
            dist = camera_state["dist"] * fy
            new_camera_state = {
                **camera_state,
                "dist": dist,
            }
        else:
            # Use width and height. Include dist, in case we control
            # a mix of orthographic and perspective cameras.
            width = camera_state["width"] * fx
            height = camera_state["height"] * fy
            dist = 0.5 * (width + height)
            new_camera_state = {
                **camera_state,
                "width": width,
                "height": height,
                "dist": dist,
            }

        # Apply
        for camera in self._cameras:
            camera.set_state(new_camera_state)

    def quickzoom_start(self, pos, camera, viewport) -> Controller:
        multiplier = 4

        # Get original state, we go back to this when quickzoom stops
        self._quickzoom_info1 = self._cameras[0].get_state()

        pan_vec = self._get_panning_to_compensate_zoom(multiplier, pos, viewport)

        # Zoom in
        new_camera_state = {**self._quickzoom_info1, "zoom": multiplier}
        for camera in self._cameras:
            camera.set_state(new_camera_state)

        # Pan to focus on cursor pos
        self.pan(pan_vec)

        # Get state using the pan_start logic
        # Becaue the projection matrix has not been set yet, the panning will
        # be as it would normally be, so - being zoomed in - it'd be quite fast.
        # This is deliberate as it is a sign that we're indeed zoomed in.
        self.pan_start(pos, viewport, camera)
        self._quickzoom_info2 = self._pan_info
        self._pan_info = None

        return self

    def quickzoom_stop(self):
        if self._quickzoom_info1 is not None:
            for camera in self._cameras:
                camera.set_state(self._quickzoom_info1)
        self._quickzoom_info1 = None
        self._quickzoom_info2 = None
        return self

    def quickzoom_move(self, pos):
        if self._quickzoom_info2 is None:
            return self
        original_pos = self._quickzoom_info2["mouse_pos"]
        delta = pos[0] - original_pos[0], pos[1] - original_pos[1]
        vec3 = (
            -self._quickzoom_info2["vecx"] * delta[0]
            + self._quickzoom_info2["vecy"] * delta[1]
        )
        self._pan(vec3, self._quickzoom_info2)
        return self

    def handle_event(self, event, viewport, camera):
        """Implements a default interaction mode that consumes wgpu autogui events
        (compatible with the jupyter_rfb event specification).
        """
        type = event.type
        if type == "pointer_down" and viewport.is_inside(event.x, event.y):
            xy = event.x, event.y
            if event.button == 1:
                self.pan_start(xy, viewport, camera)
            elif event.button == 2:
                xy = event.x, event.y
                self.zoom_start(xy)
            elif event.button == 3:
                self.quickzoom_start(xy, camera, viewport)
                if self.auto_update:
                    viewport.renderer.request_draw()
        elif type == "pointer_up":
            xy = event.x, event.y
            if event.button == 1:
                self.pan_stop()
            elif event.button == 2:
                self.zoom_stop()
            elif event.button == 3:
                self.quickzoom_stop()
                if self.auto_update:
                    viewport.renderer.request_draw()
        elif type == "pointer_move":
            xy = event.x, event.y
            if 1 in event.buttons:
                self.pan_move(xy)
                if self.auto_update:
                    viewport.renderer.request_draw()
            elif 2 in event.buttons:
                self.zoom_move(xy)
                if self.auto_update:
                    viewport.renderer.request_draw()
            elif 3 in event.buttons:
                self.quickzoom_move(xy)
                if self.auto_update:
                    viewport.renderer.request_draw()
        elif type == "wheel" and viewport.is_inside(event.x, event.y):
            if not event.modifiers:
                xy = event.x, event.y
                d = event.dy or event.dx
                f = 2 ** (-d * 0.0015)
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
