from typing import Tuple, Union

import numpy as np
import pylinalg as la

from ..utils.viewport import Viewport
from ..renderers import Renderer
from ..cameras import Camera, PerspectiveCamera
from ..cameras._perspective import fov_distance_factor


key_multiplier_factor = 10


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

    _default_controls = {}

    def __init__(
        self, camera=None, *, enabled=True, auto_update=True, register_events=None
    ):
        # Init cameras
        self._cameras = []
        if camera is not None:
            self.add_camera(camera)

        # Props
        self.enabled = enabled
        self.auto_update = auto_update

        # Init controls config
        action_names = set(
            action_tuple[0] for action_tuple in self._default_controls.values()
        )
        self._controls = Controls(*action_names)
        self._controls.update(self._default_controls)

        # State info used during interactions.
        self._actions = {}
        self._last_cam_state = {}

        # Maybe register events. The register_events arg must be a renderer or viewport.
        self._call_later = None
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
        if not self._cameras:
            self._actions = {}  # cancel any in-progress actions

    @property
    def enabled(self):
        """Whether the controller responds to events."""
        return self._enabled

    @enabled.setter
    def enabled(self, value):
        self._enabled = bool(value)
        if not self._enabled:
            self._actions = {}  # cancel any in-progress actions

    @property
    def auto_update(self):
        """Whether the controller automatically requests a new draw at the canvas."""
        return self._auto_update

    @auto_update.setter
    def auto_update(self, value):
        self._auto_update = bool(value)

    @property
    def controls(self):
        """A dictionary that maps buttons/keys to actions. Can be modified
        to configure how the controller reacts to events.
        """
        return self._controls

    def _get_target_vec(self, camera_state, **kwargs):
        """Method used by the controler implementations to determine the "target"."""
        rotation = kwargs.get("rotation", camera_state["rotation"])
        extent = 0.5 * (camera_state["width"] + camera_state["height"])
        extent = kwargs.get("extent", extent)
        fov = kwargs.get("fov", camera_state.get("fov"))

        distance = fov_distance_factor(fov) * extent
        return la.vector_apply_quaternion((0, 0, -distance), rotation)

    # def mouse_down

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
            "before_draw",
        )

    def handle_event(self, event, viewport):
        if not self.enabled:
            return
        if not self._cameras:
            return

        rect = viewport.rect
        need_update = False

        type = event.type
        if type.startswith(("pointer_", "key_", "wheel")):
            modifiers = sorted([m.lower() for m in event.modifiers])
            modifiers_prefix = "+".join(modifiers + [""])

        if type == "before_draw":
            # Do a tick, updating all actions, and using them to update the camera state.
            # Note that tick() removes actions that are done and have reached the target.
            if self._actions:
                self.tick()
                need_update = True
        elif type == "pointer_down" and viewport.is_inside(event.x, event.y):
            # Start a drag, or an action with mode push/peak/repeat
            key = modifiers_prefix + f"mouse{event.button}"
            action_tuple = self._controls.get(key)
            if action_tuple:
                need_update = True
                if action_tuple[1] == "drag":
                    # Todo: dont start drag action if another is going
                    pos = event.x, event.y
                    self._new_action_for_event(key, action_tuple, pos, pos, rect)
                else:
                    self._handle_button_down(key, action_tuple, viewport)
        elif type == "pointer_move":
            # Update all drag actions
            for action in self._actions.values():
                if action.mode == "drag" and not action.done:
                    action.set_target(np.array((event.x, event.y)))
                    need_update = True
        elif type == "pointer_up":
            # Stop all drag actions
            for action in self._actions.values():
                if action.mode == "drag":
                    action.done = True
            # End button presses, regardless of modifier state
            need_update = self._handle_button_up(f"mouse{event.button}")
        elif type == "wheel" and viewport.is_inside(event.x, event.y):
            # Wheel events. Technically there is horizontal and vertical scroll,
            # but this does not work well cross-platform, so we consider it 1D.
            key = modifiers_prefix + "wheel"
            action_tuple = self._controls.get(key)
            if action_tuple:
                need_update = True
                d = event.dy or event.dx
                pos = event.x, event.y
                action = self._actions.get(key, None)
                if action is None:
                    action = self._new_action_for_event(key, action_tuple, 0, pos, rect)
                    action.done = True
                action.increase_target(d)
        elif type == "key_down":
            # Start an action with mode push/peak/repeat
            key = modifiers_prefix + f"{event.key.lower()}"
            action_tuple = self._controls.get(key)
            if action_tuple:
                need_update = True
                self._handle_button_down(key, action_tuple, viewport)
        elif type == "key_up":
            # End key presses, regardless of modifier state
            need_update = self._handle_button_up(f"{event.key.lower()}")

        if need_update and self.auto_update:
            viewport.renderer.request_draw()

    # %% Logic used used by the public actions

    def _apply_new_camera_state(self, new_state):
        self._last_cam_state.update(new_state)

    def _update_all_cameras(self):
        for camera in self._cameras:
            camera.set_state(self._last_cam_state)

    def _create_new_action(self, action_name, offset, screen_pos, rect):
        action = Action(action_name, offset)

        action.screen_pos = screen_pos[0], screen_pos[1]
        action.rect = rect
        if not self._cameras:
            return action

        # Make sure that we have an up-to-date cam_state
        if not self._actions:
            self._last_cam_state = self._cameras[0].get_state()
        action.last_cam_state = cam_state = self._last_cam_state

        # Get vectors orthogonal to camera axii, scaled by pixel unit
        camera = self._cameras[0]
        position = cam_state["position"]
        target = position + self._get_target_vec(cam_state)
        action.vecx, action.vecy = get_screen_vectors_in_world_cords(
            target, rect[2:], camera
        )

        return action

    def apply_action(self, action, update_cameras=True):
        func_name = "_update_" + action.name
        func = getattr(self, func_name)
        func(action)
        if update_cameras:
            self._update_all_cameras()

    # %% Logic to make event input use public actions

    def tick(self):
        factor = 0.5
        # todo: take fps into account (avoid slow fps to result in extra slow control)
        to_pop = []
        for key, action in self._actions.items():
            if action.mode == "repeat" and not action.done:
                action.increase_target(key_multiplier_factor)
            action.tick(factor)
            if action.is_at_target and action.done:
                to_pop.append(key)
            self.apply_action(action, False)

        # Remove actions that are done
        for key in to_pop:
            self._actions.pop(key)

        self._update_all_cameras()

    def _new_action_for_event(self, key, action_tuple, offset, screen_pos, rect):
        if screen_pos is None:
            screen_pos = rect[0] + rect[2] / 2, rect[1] + rect[3] / 2

        action_name, mode, multiplier = action_tuple

        if isinstance(multiplier, tuple):
            multiplier = np.array(multiplier, dtype=np.float64)

        func_name = "begin_" + action_name
        func = getattr(self, func_name)
        action = func(screen_pos, rect)

        action.offset += offset
        action.done = False
        action.mode = mode
        action.multiplier = multiplier

        self._actions[key] = action
        return action
        # self._dispatch_action(action)
        # self._update_all_cameras()

    def _handle_button_down(self, key, action_tuple, viewport):
        mode = action_tuple[1]
        action = self._actions.get(key, None)
        if action is None:
            action = self._new_action_for_event(
                key, action_tuple, 0, None, viewport.rect
            )
            action.multiplier /= key_multiplier_factor
            if mode == "push":
                action.done = True
            action.set_target(key_multiplier_factor)
        if mode in ("push", "peak"):
            action.set_target(key_multiplier_factor)

    def _handle_button_up(self, button):
        need_update = False
        for key, action in self._actions.items():
            if key == button or key.endswith("+" + button):
                need_update = True
                action.done = True
                if action.mode == "peak":
                    action.set_target(action.target_value * 0)
        return need_update

    def add_default_event_handlers(self, *args):
        raise DeprecationWarning(
            "controller.add_default_event_handlers(viewport, camera) -> controller.register_events(viewport)"
        )

    def update_camera(self, *args):
        raise DeprecationWarning("controller.update_camera() is no longer necessary")


class Action:
    def __init__(self, name, offset=0.0, multiplier=1.0):
        # Clean up the offset
        if isinstance(offset, (int, float)):
            offset = float(offset)
        elif isinstance(offset, (list, tuple)):
            offset = np.array(offset, dtype=np.float64)
        elif isinstance(offset, np.ndarray):
            if offset.size == 1:
                offset = float(offset)
            else:
                offset = offset.flatten()  # makes copy

        # Derive the zero value
        if isinstance(offset, float):
            zero = 0.0
        else:
            zero = np.zeros_like(offset)

        self.name = name
        self.offset = offset
        self.last_value = zero
        self.target_value = zero
        self.current_value = zero
        self.multiplier = multiplier

    def __repr__(self):
        return f"<Action '{self.name}' {self.current_value}>"

    def set_target(self, value):
        self.target_value = value - self.offset

    def increase_target(self, value):
        self.target_value = self.target_value + value

    def tick(self, factor=1):
        new_value = (1 - factor) * self.current_value + factor * self.target_value
        dist_to_target = np.abs(self.target_value - new_value).max()
        if dist_to_target < 0.5:
            new_value = self.target_value
            # Update
        self.last_value = self.current_value
        self.current_value = new_value

    @property
    def delta(self):
        return self.multiplier * (self.current_value - self.last_value)

    @property
    def is_at_target(self):
        return np.all(self.current_value == self.target_value)


class Controls(dict):
    """Overloaded dict so we can validate when an item is set."""

    _buttons = "mouse1", "mouse2", "mouse3", "mouse4", "mouse5", "wheel"
    _buttons += "arrowleft", "arrowright", "arrowup", "arrowdown"
    _buttons += "tab", "enter", "escape", "backspace", "delete"

    _modes = "drag", "push", "peak", "repeat"

    def __init__(self, *actions):
        self._actions = tuple(actions)

    def __repr__(self):
        # Pretty print using one rule per line
        if not self:
            return "{}"
        s = "{"
        for key, action_tuple in self.items():
            s += f"\n    '{key}': {repr(action_tuple)},"
        s += "\n}"
        return s

    def __setitem__(self, key, action_tuple):
        # Check the button
        if not isinstance(key, str):
            raise TypeError("Controls key must be str")
        *modifiers, button = key.split("+")
        modifiers = sorted([m.lower() for m in modifiers])
        button = button.lower()
        for m in modifiers:
            if m not in ("shift", "control", "alt"):
                raise ValueError(f"Invalid key modifier '{m}'")
        if len(button) == 1:
            pass  # a key
        elif button not in self._buttons:
            raise ValueError(
                f"Invalid button/key '{button}', pick a char, or one of {self._buttons}"
            )
        # Check the action
        if not (isinstance(action_tuple, (list, tuple)) and len(action_tuple) == 3):
            raise TypeError("Controls action must be 3-element tuples")
        action, mode, multiplier = action_tuple
        if action not in self._actions:
            raise ValueError(f"Invalid action '{action}', pick one of {self._actions}")
        if mode not in self._modes:
            raise ValueError(f"Invalid mode '{mode}', pick one of {self._modes}")
        if mode == "drag" and not key.startswith("mouse"):
            raise ValueError("Drag mode only allowed for mouse buttons.")
        if button == "wheel" and mode != "push":
            raise ValueError("Only mode 'push' allowed with 'wheel'.")
        if isinstance(multiplier, (int, float)):
            multiplier = float(multiplier)
        elif isinstance(multiplier, (list, tuple)):
            multiplier = tuple(float(x) for x in multiplier)
        elif isinstance(multiplier, np.ndarray):
            if multiplier.size == 1:
                multiplier = float(multiplier)
            else:
                multiplier = tuple(float(x) for x in multiplier)
        # Store
        modifiers_prefix = "+".join(modifiers + [""])
        key = modifiers_prefix + button
        super().__setitem__(key, (action, mode, multiplier))

    def setdefault(self, key, default):
        if key not in self:
            self[key] = default
        return self[key]

    def update(self, e, **f):
        for k, v in e.items():
            self[k] = v
        for k, v in f.items():
            self[k] = v


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
