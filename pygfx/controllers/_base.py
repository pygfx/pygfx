from typing import Tuple, Union
from time import perf_counter

import numpy as np
import pylinalg as la

from ..cameras import Camera, PerspectiveCamera
from ..cameras._perspective import fov_distance_factor
from ..renderers import Renderer
from ..utils.viewport import Viewport


class Controller:
    """The base camera controller.

    The purpose of a controller is to provide an API to control a camera,
    and to convert user (mouse) events into camera adjustments.

    Parameters
    ----------
    camera: Camera
        The camera to control (optional). Must be a perspective or orthographic camera.
    enabled: bool
        Whether the controller is enabled (i.e. responds to events).
    damping: float
        The amount of motion damping. Zero is no damping, 10 a lot. Default 4.
    auto_update: bool
        When True (default), the controller is pretty much plug-and-play.
        For more control, it can be set to False. The controller will
        then not update the cameras automatically, and will not request
        new draws. You will then need to periodically call ``tick()``, and
        use the return value (a camera state dict).
    register_events: Renderer or Viewport
        If given and not None, will call ``.register_events()``.

    Usage
    -----

    There are multiple ways that the controller can be used.
    The easiest (and most common) approach is to use the Pygfx event system and
    make the controller listen to viewport events (using ``register_events``).

    An alternative is to feed your own events into the ``handle_event()``
    method. You'd have to mimic or use Pygfx event objects.

    The controller can also be used programmatically by calling "action methods"
    like ``pan()``, ``zoom()`` and ``rotate()``.
    """

    _default_controls = {}

    def __init__(
        self,
        camera=None,
        *,
        enabled=True,
        damping=4,
        auto_update=True,
        register_events=None,
    ):
        # Init cameras
        self._cameras = []  # tuples of camera + filters
        if camera is not None:
            self.add_camera(camera)

        self._linked_controllers = []

        # Props
        self.enabled = enabled
        self.damping = damping
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
        self._last_tick_time = 0

        # Maybe register events. The register_events arg must be a renderer or viewport.
        self._call_later = None
        if register_events is not None:
            self.register_events(register_events)

    @property
    def cameras(self):
        """A tuple with the cameras under control, in the order that they were added."""
        return tuple(cam for cam, include, exclude in self._cameras)

    def add_camera(self, camera, *, include_state=None, exclude_state=None):
        """Add a camera to control.

        The ``include_state`` and ``exclude_state`` arguments can be
        used to specify a set of camera state fields to include/exclude,
        when updating this camera. This can be used to "partially link"
        a camera. These args are None by default (i.e. no filtering).

        See ``camera.get_state()`` for available fields. Useful state
        names for the perspective and orthographhic camera are: 'x',
        'y', 'z' (or 'position' for all three), 'width', 'height',
        'rotation', 'zoom', and 'depth_range'.

        """
        if not isinstance(camera, Camera):
            raise TypeError("Controller.add_camera expects a Camera object.")
        if not isinstance(camera, PerspectiveCamera):
            raise TypeError(
                "Controller.add_camera expects a perspective or orthographic camera."
            )

        if include_state is not None:
            if not isinstance(include_state, set):
                raise TypeError("add_camera() include_state must be a set.")
            if "position" in include_state:
                include_state.discard("position")
                include_state.update({"x", "y", "z"})

        if exclude_state is not None:
            if not isinstance(exclude_state, set):
                raise TypeError("add_camera() exclude_state must be a set.")
            if "position" in exclude_state:
                exclude_state.discard("position")
                exclude_state.update({"x", "y", "z"})

        self.remove_camera(camera)
        self._cameras.append((camera, include_state, exclude_state))

    def remove_camera(self, camera):
        """Remove a camera from the list of cameras to control."""
        if not isinstance(camera, Camera):
            raise TypeError("Controller.remove_camera expects a Camera object.")
        new_cameras = []
        for cam, include_state, exclude_state in self._cameras:
            if cam is not camera:
                new_cameras.append((cam, include_state, exclude_state))
        self._cameras = new_cameras
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
    def damping(self):
        """The amount of motion damping (i.e. smoothing).
        Lower values dampen less. Typical values would be below 10.
        """
        return self._damping

    @damping.setter
    def damping(self, value):
        self._damping = max(0.0, float(value))

    @property
    def auto_update(self):
        """Whether the controller automatically requests a new draw at the canvas."""
        return self._auto_update

    @auto_update.setter
    def auto_update(self, value):
        self._auto_update = bool(value)

    @property
    def controls(self):
        """A dictionary that maps buttons/keys to actions.

        Can be modified to configure how the controller reacts to
        events. Possible keys include 'mouse1' to 'mouse5', single
        characters for the corresponding key presses, and 'arrowleft',
        'arrowright', 'arrowup', 'arrowdown', 'tab', 'enter', 'escape',
        'backspace', and 'delete'.

        Each action value is a tuple with 3 fields:

        * The `action name`, e.g. 'pan', see ``controller.controls.action_names``
          for the possible names.
        * The `mode`: 'drag', 'push', 'peek', 'repeat'. Drag represents mouse drag,
          push means the action is performed as the key is pushed, peek
          means that the action is undone once the key is released, and repeat
          means that while the key is held down, the value updates with the given
          amount per second.
        * The `multiplier` is the value that the set value from the event is
          multiplied with.

        """
        return self._controls

    @controls.setter
    def controls(self, value):
        self._controls.clear()
        self._controls.update(value)

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
            "before_render",
        )

    def tick(self):
        """Should be called periodically to keep the camera up to date.

        If ``auto_update`` is True, this is done automatically. Returns
        a dict with the new camera state, or None if there are no
        running actions.
        """

        if not self._actions:
            return None

        # Get elapsed time since last frame
        now = perf_counter()
        elapsed_time = now - self._last_tick_time
        self._last_tick_time = now
        # Determine damping/smoothing factor to update action values.
        # In the formula below the mul with elapsed_time is equivalent
        # to a division by fps. The mul with 50 is just so typical
        # values for damping lay between 0..10 :)
        factor = 1
        if self.damping > 0:
            factor = min(1, 50 * elapsed_time / self.damping)

        to_pop = []
        for key, action in self._actions.items():
            if action.mode == "repeat":
                action.increase_target(elapsed_time)
            action.tick(factor)
            if action.is_at_target and action.done:
                to_pop.append(key)
            self._apply_action(action)

        # Remove actions that are done
        for key in to_pop:
            self._actions.pop(key)

        return self._update_cameras()

    def _get_target_vec(self, camera_state, **kwargs):
        """Method used by the controller implementations to determine the "target"."""
        rotation = kwargs.get("rotation", camera_state["rotation"])
        extent = 0.5 * (camera_state["width"] + camera_state["height"])
        extent = kwargs.get("extent", extent)
        fov = kwargs.get("fov", camera_state.get("fov"))

        distance = fov_distance_factor(fov) * extent
        return la.vec_transform_quat((0, 0, -distance), rotation)

    def _get_camera_vecs(self, rect):
        """Get vectors orthogonal to camera axii."""
        if not self._cameras:
            raise ValueError("No cameras attached!")

        camera, _, _ = self._cameras[0]
        cam_state = self._get_camera_state()
        target = cam_state["position"] + self._get_target_vec(cam_state)
        vecx, vecy = get_screen_vectors_in_world_cords(target, rect[2:], camera)

        return vecx, vecy

    def _get_camera_state(self):
        """Gets the first camera's state, or the internal state if
        the controller is currently interacting.
        """
        if self._actions:
            return self._last_cam_state.copy()
        elif self._cameras:
            return self._cameras[0][0].get_state()
        else:
            raise ValueError("No cameras attached!")

    def _set_camera_state(self, new_state):
        """Set the internal camera state. Camera state can be updated multiple times
        before updating the cameras.
        """
        # Update the state dict, turn arrays into tuples, because
        # downstream code might expect that.
        for k, v in new_state.items():
            if isinstance(v, np.ndarray):
                v = tuple(v)
            self._last_cam_state[k] = v

    def _update_cameras(self):
        """Update the cameras using the internally stored state. Should only
        be called by code that knows that internally stored state is valid.
        """
        if self._auto_update:
            for camera, include_state, exclude_state in self._cameras:
                state = self._last_cam_state
                if include_state or exclude_state:
                    if "position" in state:
                        state = state.copy()
                        state["x"], state["y"], state["z"] = state.pop("position")
                    if include_state is not None:
                        state = {k: v for k, v in state.items() if k in include_state}
                    if exclude_state is not None:
                        state = {
                            k: v for k, v in state.items() if k not in exclude_state
                        }
                camera.set_state(state)
        return self._last_cam_state

    def add_default_event_handlers(self, *args):
        raise DeprecationWarning(
            "controller.add_default_event_handlers(viewport, camera) -> controller.register_events(viewport)"
        )

    def update_camera(self, *args):
        raise DeprecationWarning("controller.update_camera() is no longer necessary")

    # %% Builtin event handling

    def handle_event(self, event, viewport):
        if not self.enabled:
            return
        if not self._cameras:
            return

        rect = viewport.rect
        need_update = False

        type = event.type
        if type.startswith(("pointer_", "key_", "wheel")):
            modifiers = {m.lower() for m in event.modifiers}
            if type.startswith("key_"):
                modifiers.discard(event.key.lower())
            modifiers_prefix = "+".join(sorted(modifiers) + [""])

        if type == "before_render":
            # Do a tick, updating all actions, and using them to update the camera state.
            # Note that tick() removes actions that are done and have reached the target.
            if self._auto_update and self._actions:
                self.tick()
                need_update = True
        elif type == "pointer_down" and viewport.is_inside(event.x, event.y):
            # Start a drag, or an action with mode push/peek/repeat
            key = modifiers_prefix + f"mouse{event.button}"
            action_tuple = self._controls.get(key)
            if action_tuple:
                need_update = True
                if action_tuple[1] == "drag":
                    # Dont start a new drag if there is one going
                    if not any(
                        (a.mode == "drag" and not a.done)
                        for a in self._actions.values()
                    ):
                        pos = event.x, event.y
                        self._create_action(key, action_tuple, pos, pos, rect)
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
                    action = self._create_action(key, action_tuple, 0, pos, rect)
                    action.done = True
                action.increase_target(d)
        elif type == "key_down":
            # Start an action with mode push/peek/repeat
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

    def _handle_button_down(self, key, action_tuple, viewport):
        """Common code to handle key/mouse button presses."""
        mode = action_tuple[1]
        action = self._actions.get(key, None)
        if action is None:
            action = self._create_action(key, action_tuple, 0, None, viewport.rect)
            action.snap_distance = 0.01
        action.done = mode == "push"
        if mode in ("push", "peek"):
            action.set_target(1)

    def _handle_button_up(self, button):
        """Common code to handle key/mouse button releases."""
        need_update = False
        for key, action in self._actions.items():
            if key == button or key.endswith("+" + button):
                need_update = True
                action.done = True
                if action.mode == "peek":
                    action.set_target(action.target_value * 0)
        return need_update

    def _create_action(self, key, action_tuple, offset, screen_pos, rect):
        """Creates an action object, which helps keep track of the operation."""

        key = key or str(perf_counter())
        if screen_pos is None:
            screen_pos = rect[0] + rect[2] / 2, rect[1] + rect[3] / 2

        # Get vectors orthogonal to camera axii, scaled by pixel unit
        vecx, vecy = self._get_camera_vecs(rect)

        # Make sure that we have an up-to-date cam_state
        if not self._actions:
            self._last_cam_state = self._cameras[0][0].get_state()
            self._last_tick_time = perf_counter()

        # Create action
        kwargs = dict(rect=rect, screen_pos=screen_pos, vecx=vecx, vecy=vecy)
        action = Action(action_tuple, offset, kwargs)

        self._actions[key] = action
        return action

    def _apply_action(self, action):
        """Apply the action by calling the appropriate update method."""
        # Get function to call
        func = getattr(self, "_update_" + action.name)

        # Collect the kwargs that the function needs
        code = func.__func__.__code__
        assert code.co_argcount == 2  # self and delta
        assert code.co_posonlyargcount == 0
        kwargnames = code.co_varnames[2 : 2 + code.co_kwonlyargcount]
        kwargs = {k: action.kwargs[k] for k in kwargnames}

        # Call it!
        func(action.delta, **kwargs)

    # %% Actions on the base class

    def quickzoom(self, delta: float, *, animate=False):
        """Zoom the view using the camera's zoom property. This is intended
        for temporary zoom operations.

        If animate is True, the motion is damped. This requires the
        controller to receive events from the renderer/viewport.
        """

        if animate:
            action_tuple = ("quickzoom", "push", 1.0)
            action = self._create_action(None, action_tuple, 0.0, None, (0, 0, 1, 1))
            action.set_target(delta)
            action.done = True
        elif self._cameras:
            self._update_quickzoom(delta)
            return self._update_cameras()

    def _update_quickzoom(self, delta):
        assert isinstance(delta, (int, float))
        zoom = self._get_camera_state()["zoom"]
        new_cam_state = {"zoom": zoom * 2**delta}
        self._set_camera_state(new_cam_state)

    def update_fov(self, delta, *, animate):
        """Adjust the field of view with the given delta value (Limited to [1, 179]).

        If animate is True, the motion is damped. This requires the
        controller to receive events from the renderer/viewport.
        """

        if animate:
            action_tuple = ("fov", "push", 1.0)
            action = self._create_action(None, action_tuple, 0.0, None, (0, 0, 1, 1))
            action.set_target(delta)
            action.done = True
        elif self._cameras:
            self._update_fov(delta)
            return self._update_cameras()

    def _update_fov(self, delta: float):
        fov_range = self._cameras[0][0]._fov_range

        # Get current state
        cam_state = self._get_camera_state()
        position = cam_state["position"]
        fov = cam_state["fov"]

        # Update fov and position
        new_fov = min(max(fov + delta, fov_range[0]), fov_range[1])
        pos2target1 = self._get_target_vec(cam_state, fov=fov)
        pos2target2 = self._get_target_vec(cam_state, fov=new_fov)
        new_position = position + pos2target1 - pos2target2

        self._set_camera_state({"fov": new_fov, "position": new_position})


class Action:
    """Simple value to represent a value to change an action with."""

    def __init__(self, action_tuple, offset=0.0, kwargs=None):
        action_name, mode, multiplier = action_tuple

        # The name of the action, used to dispatch to controller._update_xx()
        self.name = action_name

        # The offset defines the dimension of the input
        self.offset = self._clean_up_value(offset)

        # The multiplier defines the dimension of the output.
        # If it's less than the input, the first input dimension(s) are dropped.
        self.multiplier = self._clean_up_value(multiplier)

        # Derive the zero value
        if isinstance(offset, float):
            zero = 0.0
        else:
            zero = np.zeros_like(offset)

        # Init the values
        self.last_value = zero
        self.target_value = zero
        self.current_value = zero
        self.repeat_multiplier = 0

        self.snap_distance = 0.5
        self.done = False
        self.mode = mode
        self.kwargs = kwargs or {}

    def _clean_up_value(self, value):
        # Turns value into either a float or a flat nd float array
        if isinstance(value, (int, float)):
            return float(value)
        elif isinstance(value, (list, tuple)):
            return np.array(value, dtype=np.float64)
        elif isinstance(value, np.ndarray):
            if value.size == 1:
                return float(value)
            else:
                return value.flatten()  # makes copy

    def __repr__(self):
        return f"<Action '{self.name}' {self.current_value}>"

    def set_target(self, value):
        self.target_value = value - self.offset

    def increase_target(self, value):
        if self.mode == "repeat":
            # We increase/decrease a multiplier linearly
            lag_time = 1.5  # seconds
            if not self.done:
                self.repeat_multiplier = min(
                    1.0, self.repeat_multiplier + value / lag_time
                )
            else:
                self.repeat_multiplier = max(
                    0.0, self.repeat_multiplier - value / lag_time
                )
            self.target_value = self.target_value + value * self.repeat_multiplier
        else:
            self.target_value = self.target_value + value

    def tick(self, factor=1):
        # Update value
        new_value = (1 - factor) * self.current_value + factor * self.target_value
        self.last_value = self.current_value
        self.current_value = new_value
        #
        dist_to_target = np.abs(self.target_value - self.current_value).max()
        if dist_to_target < self.snap_distance:
            self.current_value = self.target_value * 1.0  # make a copy if array

    @property
    def delta(self):
        """Get the delta value, multiplied with the multiplier."""
        delta = self.multiplier * (self.current_value - self.last_value)
        if not isinstance(delta, float):
            delta = tuple(delta)
            if isinstance(self.multiplier, float):
                delta = delta[-1]
            elif self.multiplier.size < len(delta):
                delta = delta[-self.multiplier.size :]
        return delta

    @property
    def is_at_target(self):
        if self.mode == "repeat":
            if self.repeat_multiplier:
                return False
        return np.all(self.current_value == self.target_value)


class Controls(dict):
    """Overloaded dict so we can validate when an item is set."""

    _buttons = "mouse1", "mouse2", "mouse3", "mouse4", "mouse5", "wheel"
    _buttons += "arrowleft", "arrowright", "arrowup", "arrowdown"
    _buttons += "tab", "enter", "escape", "backspace", "delete"
    _buttons += "shift", "control"

    _modes = "drag", "push", "peek", "repeat"

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

    @property
    def action_names(self):
        """The possible action names."""
        return self._actions

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
        if mode == "drag" and not button.startswith("mouse"):
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
        # Overloaded to make use of our implementation of __setitem__
        if key not in self:
            self[key] = default
        return self[key]

    def update(self, e, **f):
        # Overloaded to make use of our implementation of __setitem__
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
    camera_world = camera.world.matrix
    camera_world_inverse = camera.world.inverse_matrix
    camera_projection = camera.projection_matrix
    camera_projection_inverse = camera.projection_matrix_inverse

    # Get center location on screen
    center_ndc = la.vec_transform(
        la.vec_transform(center_world, camera_world_inverse), camera_projection
    )

    # Step 1 NDC unit in x and y, and convert these positions back to world
    posx_ndc = center_ndc + (1, 0, 0)
    posy_ndc = center_ndc + (0, 1, 0)
    posx_world = la.vec_transform(
        la.vec_transform(posx_ndc, camera_projection_inverse),
        camera_world,
    )
    posy_world = la.vec_transform(
        la.vec_transform(posy_ndc, camera_projection_inverse),
        camera_world,
    )

    # Calculate the vectors, and scale to logical pixels.
    vecx_world = posx_world - center_world
    vecy_world = posy_world - center_world
    return vecx_world * 2.0 / scene_size[0], vecy_world * 2.0 / scene_size[1]
