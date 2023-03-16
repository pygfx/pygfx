from typing import Tuple

from ..utils.viewport import Viewport
from ._base import Controller, get_screen_vectors_in_world_cords


class PanZoomController(Controller):
    """A controller to pan and zoom a camera in a 2D plane  parallel to the screen.

    Controls:

    * Left mouse button: pan.
    * Right mouse button: zoom (if `camera.maintain_aspect==False`, zooms in both dimensions).
    * Middle mouse button: quick zoom.
    * Scroll: zoom.

    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # State info used during interactions. Because this is a single object,
        # there can be only one action happening at a time, which prevents
        # the controller/camera from entering weird states.
        self._action = None

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
        # -> the camera does have width/height of the viewport ...
        # -> let's revisit when we have viewport etc. figured out?

        if self._action:
            return

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

        if self._action:
            return

        if self._cameras:
            camera = self._cameras[0]

            # Get camera state
            self._action = {"name": "pan"}
            self._action["camera_state"] = camera_state = camera.get_state()
            position = camera_state["position"]

            # Get the vectors that point in the axis direction
            target = position + self._get_target_vec(camera_state)
            scene_size = viewport.logical_size
            vecx, vecy = get_screen_vectors_in_world_cords(target, scene_size, camera)

            # Store pan info
            self._action.update({"mouse_pos": pos, "vecx": vecx, "vecy": vecy})

        return self

    def pan_stop(self) -> Controller:
        """Stop the current panning operation."""
        self._action = None
        return self

    def pan_move(self, pos: Tuple[float, float]) -> Controller:
        """Handle mouse move during a panning operation. Call `pan_start()` first."""

        if self._action and self._action["name"] == "pan":
            # Get state
            original_pos = self._action["mouse_pos"]
            vecx = self._action["vecx"]
            vecy = self._action["vecy"]

            # Update
            delta = pos[0] - original_pos[0], pos[1] - original_pos[1]
            vec3 = vecy * delta[1] - vecx * delta[0]
            self._pan(vec3, self._action["camera_state"])

        return self

    def zoom(self, multiplier: float) -> Controller:
        """Zoom the view with the given multiplier.

        Note that this sets the camera's width/height for an orthographic camera,
        and distance from target for the perspective camera.
        """
        if self._action:
            return

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
        if self._action:
            return

        if self._cameras:
            self.zoom(multiplier)
            self.pan(self._get_panning_to_compensate_zoom(multiplier, pos, viewport))
        return self

    def zoom_start(self, pos: Tuple[float, float], viewport=Viewport) -> Controller:
        """Start a zoom operation.

        Note that this sets the camera's width/height for an orthographic camera,
        and distance from target for the perspective camera.
        """

        if self._action:
            return

        if self._cameras:
            camera = self._cameras[0]

            # Get camera state
            self._action = {"name": "zoom"}
            self._action["camera_state"] = camera.get_state()
            # Store pan info
            self._action.update({"mouse_pos": pos})

        return self

    def zoom_stop(self) -> Controller:
        """Stop the current zoom operation."""
        self._action = None
        return self

    def zoom_move(self, pos: Tuple[float, float]) -> Controller:
        """Handle mouse move during a zoom operation. Call `zoom_start()` first."""
        if self._action and self._action["name"] == "zoom":
            # Get state
            original_pos = self._action["mouse_pos"]
            camera_state = self._action["camera_state"]
            position = camera_state["position"]
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

        if self._action:
            return

        multiplier = self.quick_zoom_factor

        if self._cameras:
            camera = self._cameras[0]

            original_zoom = camera.zoom

            # Zoom in
            new_camera_state = {"zoom": multiplier}
            for camera in self._cameras:
                camera.set_state(new_camera_state)

            # Get state using the pan_start logic
            # Becaue the projection matrix has not been set yet, the panning will
            # be as it would normally be, so - being zoomed in - it'd be quite fast.
            # This is deliberate as it is a sign that we're indeed zoomed in.
            self.pan_start(pos, viewport)
            self._action["name"] = "quickzoom"
            self._action["original_zoom"] = original_zoom

        return self

    def quickzoom_stop(self) -> Controller:
        """Stop the current quickzoom operation."""
        if self._action and self._action["name"] == "quickzoom":
            # Restore the zoom level (maintain the changed panning)
            for camera in self._cameras:
                camera.set_state({"zoom": self._action["original_zoom"]})
        self._action = None
        return self

    def quickzoom_move(self, pos: Tuple[float, float]) -> Controller:
        """Handle mouse move during a quickzoom operation. Call `quickzoom_start()` first."""
        if self._action and self._action["name"] == "quickzoom":
            original_pos = self._action["mouse_pos"]
            delta = pos[0] - original_pos[0], pos[1] - original_pos[1]
            vec3 = -self._action["vecx"] * delta[0] + self._action["vecy"] * delta[1]
            self._pan(vec3, self._action["camera_state"])

        return self

    def adjust_fov(self, delta: float):
        """Adjust the field of view with the given delta value (Limited to [1, 179])."""

        if self._action:
            return

        if self._cameras:
            camera = self._cameras[0]

            # Get current state
            camera_state = camera.get_state()
            position = camera_state["position"]
            fov = camera_state["fov"]

            # Update fov and position
            new_fov = min(max(fov + delta, camera._fov_range[0]), camera._fov_range[1])
            pos2target1 = self._get_target_vec(camera_state, fov=fov)
            pos2target2 = self._get_target_vec(camera_state, fov=new_fov)
            new_position = position + pos2target1 - pos2target2

            # Apply to cameras
            new_camera_state = {
                **camera_state,
                "fov": new_fov,
                "position": new_position,
            }
            for camera in self._cameras:
                camera.set_state(new_camera_state)

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
            if 2 in event.buttons:
                self.zoom_move(xy)
                need_update = True
            if 3 in event.buttons:
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
