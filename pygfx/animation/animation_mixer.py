import numpy as np
from .animation_action import AnimationAction
from ..objects import RootEventHandler, Group, Mesh


class AnimationMixer(RootEventHandler):
    """The AnimationMixer serves as a player for animations within a scene, managing the playback state of all animations.

    It handles blending and playing multiple animation actions simultaneously, combining one or more AnimationAction objects
    and controlling transitions and smooth blending between different animations based on time and weight.
    """

    def __init__(self):
        # todo: separate the animation data from model objects, bind them together here in the mixer with a root object when actions are created
        # self._root = root
        super().__init__()
        self._actions = {}
        self._time = 0
        self._time_scale = 1.0

        self.__activated_actions = []

        self.__property_accu_cache = {}  # Cache for the accumulated target value for active actions

        self.__property_ori_cache = {}  # Cache for the original target value

    @property
    def time(self):
        """The global mixer time (in seconds; starting with 0 on the mixer's creation)."""
        return self._time

    @property
    def time_scale(self):
        """The time scale of the mixer. You can use this to control the general speed of all animations.
        Default is 1.0.
        """
        return self._time_scale

    @time_scale.setter
    def time_scale(self, value):
        self._time_scale = value

    def clip_action(self, clip):
        """Create an AnimationAction from an AnimationClip.
        Calling this method several times with the same clip and root parameters always returns the same clip instance.
        """

        if clip in self._actions:
            return self._actions[clip]

        action = AnimationAction(self, clip)
        self._actions[clip] = action

        # cache the original value
        for t in action._clip.tracks:
            key = (t.target, t.path)
            if key not in self.__property_ori_cache:
                ori_value = self._get_path_value(t.target, t.path).copy()
                if isinstance(ori_value, list):
                    ori_value = np.array(ori_value)
                self.__property_ori_cache[key] = ori_value

        return action

    def _activate_action(self, action):
        if not self._is_active_action(action):
            self.__activated_actions.append(action)

    def _deactivate_action(self, action):
        if self._is_active_action(action):
            self.__activated_actions.remove(action)

        # restore the original value
        for t in action._clip.tracks:
            key = (t.target, t.path)
            if key in self.__property_ori_cache:
                self._set_path_value(t.target, t.path, self.__property_ori_cache[key])

    def _is_active_action(self, action):
        return action in self.__activated_actions

    def set_time(self, time):
        """Set the time of the mixer."""
        self._time = 0
        for action in self.__activated_actions:
            action.time = 0

        self.update(time)

    def update(self, dt):
        """Update the mixer with the time delta."""
        self.__property_accu_cache.clear()
        dt = dt * self._time_scale
        self._time += dt
        for action in self.__activated_actions:
            action._update(dt)

        # apply the accumulated value
        components_map = ["position", "rotation", "scale"]
        for target, paths in self.__property_accu_cache.items():
            components = [None, None, None]
            for path, (accu_value, accu_weight) in paths.items():
                if accu_weight < 1:
                    ori_value = self.__property_ori_cache[(target, path)]
                    if path == "rotation":
                        accu_value = self._mix_slerp(
                            accu_value, ori_value, 1 - accu_weight
                        )
                    else:
                        # translation\scale\weights
                        accu_value = self._mix_lerp(
                            accu_value, ori_value, 1 - accu_weight
                        )

                if path in components_map:
                    components[components_map.index(path)] = accu_value
                else:
                    self._set_path_value(target, path, accu_value)

            if any(c is not None for c in components):
                self._set_path_value(target, "components", components)

    def _accumulate(self, target, path, value, weight):
        accu_cache = self.__property_accu_cache
        if target not in accu_cache:
            accu_cache[target] = {}
        if path not in accu_cache[target]:
            accu_cache[target][path] = (value, weight)
        else:
            current_value, current_weight = accu_cache[target][path]
            current_weight += weight
            mix = weight / current_weight

            if path == "rotation":
                current_value = self._mix_slerp(current_value, value, mix)
            else:
                # translation\scale\weights
                current_value = self._mix_lerp(current_value, value, mix)

            accu_cache[target][path] = (current_value, current_weight)

    def _mix_lerp(self, a, b, t):
        return a * (1 - t) + b * t

    def _mix_slerp(self, a, b, t):
        dot = np.dot(a, b)
        if dot < 0:
            a = -a
            dot = -dot
        if dot > 0.99:
            q = self._mix_lerp(a, b, t)
            return q / np.linalg.norm(q)
        theta = np.arccos(dot)
        sin_theta = np.sin(theta)

        return (np.sin((1 - t) * theta) / sin_theta) * a + (
            np.sin(t * theta) / sin_theta
        ) * b

    def _mix_select(self, a, b, t):
        return a if t < 0.5 else b

    def _get_path_value(self, target, path):
        if path == "scale":
            return target.local.scale
        elif path == "translation":
            return target.local.position
        elif path == "rotation":
            return target.local.rotation
        elif path == "components":
            return target.local.components
        elif path == "weights":
            if isinstance(target, Mesh):
                return target.morph_target_influences
            elif isinstance(target, Group):
                for c in target.children:
                    if isinstance(c, Mesh):
                        return c.morph_target_influences

    def _set_path_value(self, target, path, value):
        if path == "scale":
            target.local.scale = value
        elif path == "translation":
            target.local.position = value
        elif path == "rotation":
            target.local.rotation = value
        elif path == "components":
            target.local.components = value
        elif path == "weights":
            if isinstance(target, Mesh):
                target.morph_target_influences = value
            elif isinstance(target, Group):
                for c in target.children:
                    if isinstance(c, Mesh):
                        c.morph_target_influences = value
