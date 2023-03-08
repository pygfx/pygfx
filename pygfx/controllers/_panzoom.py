from typing import Tuple

from ..utils.viewport import Viewport
from ._base import Controller, get_screen_vectors_in_world_cords


class PanZoomController(Controller):
    """A controller to move a camera in a 2D plane."""

    def __init__(self, camera, *, auto_update: bool = True) -> None:
        super().__init__(camera)

        self.auto_update = auto_update

        # State info used during pan/zoom operations
        self._pan_info = None
        self._quickzoom_info1 = None
        self._quickzoom_info2 = None

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
        new_camera_state = {**camera_state, "position": new_position}
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
            extent = camera_state["extent"]

            # Get new extent and new position
            new_extent = extent * (1 / multiplier)
            pos2target1 = self._get_target_vec(camera_state, extent=extent)
            pos2target2 = self._get_target_vec(camera_state, extent=new_extent)
            new_position = position + pos2target1 - pos2target2

            # Apply new state to all cameras
            new_camera_state = {
                **camera_state,
                "position": new_position,
                "extent": new_extent,
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
            extent = camera_state["extent"]
            aspect = camera_state["aspect"]

            # Calculate zoom factors
            delta = pos[0] - original_pos[0], pos[1] - original_pos[1]
            fx = 2 ** (-delta[0] * 0.01)
            fy = 2 ** (delta[1] * 0.01)

            # Apply
            if maintain_aspect:
                # Use extent
                new_extent = extent * fy
                new_camera_state = {**camera_state}
            else:
                # Use width and height. Include extent, in case we control
                # a mix of orthographic and perspective cameras.
                sqrt_aspect = aspect**0.5
                width = extent * sqrt_aspect
                height = extent / sqrt_aspect
                new_width = width * fx
                new_height = height * fy
                new_extent = 0.5 * (new_width + new_height)
                new_aspect = new_width / new_height
                new_camera_state = {
                    **camera_state,
                    "aspect": new_aspect,
                }

            # Get  new position
            pos2target1 = self._get_target_vec(camera_state, extent=extent)
            pos2target2 = self._get_target_vec(camera_state, extent=new_extent)
            new_position = position + pos2target1 - pos2target2

            # Apply
            new_camera_state = {
                **new_camera_state,
                "extent": new_extent,
                "position": new_position,
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
        multiplier = 4

        if self._cameras:
            camera = self._cameras[0]

            # Get original state, we go back to this when quickzoom stops
            self._quickzoom_info1 = camera.get_state()

            # Zoom in
            new_camera_state = {**self._quickzoom_info1, "zoom": multiplier}
            for camera in self._cameras:
                camera.set_state(new_camera_state)

            # Pan to focus on cursor pos
            pan_vec = self._get_panning_to_compensate_zoom(multiplier, pos, viewport)
            self.pan(pan_vec)

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
            for camera in self._cameras:
                camera.set_state(self._quickzoom_info1)
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
