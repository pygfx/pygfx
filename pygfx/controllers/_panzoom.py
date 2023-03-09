from typing import Tuple

from ..utils.viewport import Viewport
from ._base import Controller, get_screen_vectors_in_world_cords


class PanZoomController(Controller):
    """A controller to move a camera in a 2D plane."""

    def __init__(self, camera, *, enabled=True, auto_update=True) -> None:
        super().__init__(camera, enabled=enabled, auto_update=auto_update)

        # State info used during pan/zoom operations
        self._pan_info = None
        self._quickzoom_info1 = None
        self._quickzoom_info2 = None

        self.mouse_zoom_factor = 0.005
        self.scroll_zoom_factor = 0.0015
        self.quick_zoom_factor = 4

    @property
    def mouse_zoom_factor(self):
        """The factor to turn mouse motion (in logical pixels) to a zoom factor)."""
        return self._mouse_zoom_factor

    @mouse_zoom_factor.setter
    def mouse_zoom_factor(self, value):
        self._mouse_zoom_factor = float(value)

    @property
    def scroll_zoom_factor(self):
        """The factor to turn mouse scrolling to a zoom factor)."""
        return self._scroll_zoom_factor

    @scroll_zoom_factor.setter
    def scroll_zoom_factor(self, value):
        self._scroll_zoom_factor = float(value)

    @property
    def quick_zoom_factor(self):
        """The multiplier to use for quickzoom."""
        return self._quick_zoom_factor

    @quick_zoom_factor.setter
    def quick_zoom_factor(self, value):
        self._quick_zoom_factor = float(value)

    def pan(self, vec3: Tuple[float, float, float]) -> Controller:
        """Pan in 3D world coordinates."""
        # Note: cannot pan in "controller space" (i.e. using 2D coords)
        # because we need the screen size for get_screen_vectors_in_world_cords,
        # and we only have that via an event.

        if self._cameras:
            camera = self._cameras[0]

            self._pan(vec3, camera.get_state())

        return self

    def _pan(self, vec3, camera_state):
        # Get new position
        position = camera_state["position"]
        new_position = position + vec3

        # Apply new state to all cameras
        new_camera_state = {"position": new_position}
        for camera in self._cameras:
            camera.set_state(new_camera_state)

    def pan_start(self, pos: Tuple[float, float], viewport: Viewport) -> Controller:
        """Start a panning operation based on (2D) screen coordinates."""

        if self._cameras:
            camera = self._cameras[0]

            # Get camera state
            self._pan_info = camera_state = camera.get_state()
            position = camera_state["position"]

            # Get the vectors that point in the axis direction
            target = position + self._get_target_vec(camera_state)
            scene_size = viewport.logical_size
            vecx, vecy = get_screen_vectors_in_world_cords(target, scene_size, camera)

            # Store pan info
            self._pan_info.update({"mouse_pos": pos, "vecx": vecx, "vecy": vecy})

        return self

    def pan_stop(self) -> Controller:
        """Stop the current panning operation."""
        self._pan_info = None
        return self

    def pan_move(self, pos: Tuple[float, float]) -> Controller:
        """Handle mouse move during a panning operation. Call `pan_start()` first."""

        if self._pan_info:
            # Get state
            original_pos = self._pan_info["mouse_pos"]
            vecx = self._pan_info["vecx"]
            vecy = self._pan_info["vecy"]

            # Update
            delta = pos[0] - original_pos[0], pos[1] - original_pos[1]
            vec3 = vecy * delta[1] - vecx * delta[0]
            self._pan(vec3, self._pan_info)

        return self

    def zoom(self, multiplier: float) -> Controller:
        """Zoom the view with the given multiplier.

        Note that this sets the camera's width/height for an orthographic camera,
        and distance from target for the perpective camera.
        """
        if self._cameras:
            camera = self._cameras[0]

            # Get current state
            camera_state = camera.get_state()
            position = camera_state["position"]
            width = camera_state["width"]
            height = camera_state["height"]
            extent = 0.5 * (width + height)

            # Get new width, height, and extent
            new_width = width / multiplier
            new_height = height / multiplier
            new_extent = 0.5 * (new_width + new_height)

            # Get new position
            pos2target1 = self._get_target_vec(camera_state, extent=extent)
            pos2target2 = self._get_target_vec(camera_state, extent=new_extent)
            new_position = position + pos2target1 - pos2target2

            # Apply new state to all cameras
            new_camera_state = {
                "position": new_position,
                "width": new_width,
                "height": new_height,
                "fov": camera_state["fov"],
            }
            for camera in self._cameras:
                camera.set_state(new_camera_state)

        return self

    def _get_panning_to_compensate_zoom(self, multiplier, pos, viewport):
        camera = self._cameras[0]

        # Get Target
        camera_state = camera.get_state()
        position = camera_state["position"]
        target = position + self._get_target_vec(camera_state)

        # Get viewport info
        x, y, w, h = viewport.rect
        offset = x, y
        size = w, h

        # Calculate pan such that what was previously under the mouse is again under the mouse.
        vecx, vecy = get_screen_vectors_in_world_cords(target, size, camera)
        delta = tuple(pos[i] - offset[i] - size[i] / 2 for i in (0, 1))
        delta1 = vecx * delta[0] - vecy * delta[1]
        delta2 = delta1 / multiplier

        return delta1 - delta2

    def zoom_to_point(
        self, multiplier: float, pos: Tuple[float, float], viewport: Viewport
    ) -> Controller:
        """Zoom, but keep the spot that's at the cursor centered at the cursor."""
        if self._cameras:
            self.zoom(multiplier)
            self.pan(self._get_panning_to_compensate_zoom(multiplier, pos, viewport))
        return self

    def zoom_start(self, pos: Tuple[float, float], viewport=Viewport) -> Controller:
        """Start a zoom operation.

        Note that this sets the camera's width/height for an orthographic camera,
        and distance from target for the perpective camera.
        """
        if self._cameras:
            camera = self._cameras[0]

            # Get camera state
            self._zoom_info = camera.get_state()
            # Store pan info
            self._zoom_info.update({"mouse_pos": pos})

        return self

    def zoom_stop(self) -> Controller:
        """Stop the current zoom operation."""
        self._zoom_info = None
        return self

    def zoom_move(self, pos: Tuple[float, float]) -> Controller:
        """Handle mouse move during a zoom operation. Call `zoom_start()` first."""
        if self._zoom_info:
            # Get state
            camera_state = self._zoom_info
            position = camera_state["position"]
            original_pos = camera_state["mouse_pos"]
            maintain_aspect = camera_state.get("maintain_aspect", True)
            width = camera_state["width"]
            height = camera_state["height"]
            extent = 0.5 * (width + height)

            # Calculate zoom factors
            delta = pos[0] - original_pos[0], pos[1] - original_pos[1]
            fx = 2 ** (-delta[0] * self.mouse_zoom_factor)
            fy = 2 ** (delta[1] * self.mouse_zoom_factor)

            # Apply
            if maintain_aspect:
                # Scale width and height equally
                new_width = width * fy
                new_height = height * fy
            else:
                # Use width and height.
                new_width = width * fx
                new_height = height * fy

            # Get  new position
            new_extent = 0.5 * (new_width + new_height)
            pos2target1 = self._get_target_vec(camera_state, extent=extent)
            pos2target2 = self._get_target_vec(camera_state, extent=new_extent)
            new_position = position + pos2target1 - pos2target2

            # Apply
            new_camera_state = {
                "width": new_width,
                "height": new_height,
                "position": new_position,
                "fov": camera_state["fov"],
            }
            for camera in self._cameras:
                camera.set_state(new_camera_state)

        return self

    def quickzoom_start(
        self, pos: Tuple[float, float], viewport: Viewport
    ) -> Controller:
        """Start a quickzoom operation.

        In contrast to the other zoom methods, this actually uses the
        camera zoom property.
        """
        multiplier = self.quick_zoom_factor

        if self._cameras:
            camera = self._cameras[0]

            # Get original state, we go back to this when quickzoom stops
            self._quickzoom_info1 = camera.get_state()

            # Zoom in
            new_camera_state = {"zoom": multiplier}
            for camera in self._cameras:
                camera.set_state(new_camera_state)

            # Pan to focus on cursor pos, commented, because it feels unnatural
            # pan_vec = self._get_panning_to_compensate_zoom(multiplier, pos, viewport)
            # self.pan(pan_vec)

            # Get state using the pan_start logic
            # Becaue the projection matrix has not been set yet, the panning will
            # be as it would normally be, so - being zoomed in - it'd be quite fast.
            # This is deliberate as it is a sign that we're indeed zoomed in.
            self.pan_start(pos, viewport)
            self._quickzoom_info2 = self._pan_info
            self._pan_info = None

        return self

    def quickzoom_stop(self) -> Controller:
        """Stop the current quickzoom operation."""
        if self._quickzoom_info1 is not None:
            # Restore the zoom level (maintain the changed panning)
            for camera in self._cameras:
                camera.set_state({"zoom": self._quickzoom_info1["zoom"]})
        self._quickzoom_info1 = None
        self._quickzoom_info2 = None
        return self

    def quickzoom_move(self, pos: Tuple[float, float]) -> Controller:
        """Handle mouse move during a quickzoom operation. Call `quickzoom_start()` first."""
        if self._quickzoom_info2:
            original_pos = self._quickzoom_info2["mouse_pos"]
            delta = pos[0] - original_pos[0], pos[1] - original_pos[1]
            vec3 = (
                -self._quickzoom_info2["vecx"] * delta[0]
                + self._quickzoom_info2["vecy"] * delta[1]
            )
            self._pan(vec3, self._quickzoom_info2)

        return self

    def handle_event(self, event, viewport):
        """Implements a default interaction mode that consumes wgpu autogui events
        (compatible with the jupyter_rfb event specification).
        """
        if not self.enabled:
            return
        need_update = False

        type = event.type
        if type == "pointer_down" and viewport.is_inside(event.x, event.y):
            xy = event.x, event.y
            if event.button == 1:
                self.pan_start(xy, viewport)
            elif event.button == 2:
                xy = event.x, event.y
                self.zoom_start(xy, viewport)
            elif event.button == 3:
                self.quickzoom_start(xy, viewport)
                need_update = True
        elif type == "pointer_up":
            xy = event.x, event.y
            if event.button == 1:
                self.pan_stop()
            elif event.button == 2:
                self.zoom_stop()
            elif event.button == 3:
                self.quickzoom_stop()
                need_update = True
        elif type == "pointer_move":
            xy = event.x, event.y
            if 1 in event.buttons:
                self.pan_move(xy)
                need_update = True
            elif 2 in event.buttons:
                self.zoom_move(xy)
                need_update = True
            elif 3 in event.buttons:
                self.quickzoom_move(xy)
                need_update = True
        elif type == "wheel" and viewport.is_inside(event.x, event.y):
            if not event.modifiers:
                xy = event.x, event.y
                d = event.dy or event.dx
                f = 2 ** (-d * self.scroll_zoom_factor)
                self.zoom_to_point(f, xy, viewport)
                need_update = True

        if need_update and self.auto_update:
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
