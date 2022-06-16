import weakref


class Environment:
    """Object that represents the state-type of the "environment".
    With environment we mean not the wobject (and associates material,
    geometry and resources), nor the global stuff (like stdinfo and
    glyph atlas), but the stuff that affects the rendering, like
    renderer state, and lights.
    """

    def __init__(self, renderer_state_hash, scene_state_hash):

        self._renderers = weakref.WeakSet()
        self._scenes = weakref.WeakSet()
        self._renderer_state_hash = renderer_state_hash
        self._scene_state_hash = scene_state_hash

    @property
    def hash(self):
        return self._renderer_state_hash, self._scene_state_hash

    def register(self, renderer, scene, state):
        self._renderers.add(renderer)
        self._scenes.add(scene)

        # Store stuff relevant to this environment
        self.blender = state["blender"]

    def check_unused(self, renderer, scene, renderer_state_hash, scene_state_hash):
        renderers = set(self._renderers)
        scenes = set(self._scenes)

        if renderer in renderers:
            if renderer_state_hash != self._renderer_state_hash:
                self._renderers.discard(renderer)

        if scene in scenes:
            if scene_state_hash != self._scene_state_hash:
                self._scenes.discard(scene)

        if not renderers or not scenes:
            return True


class GlobalEnvironmentManager:
    def __init__(self):
        self.envs = {}  # hash -> Environment
        self._containers = weakref.WeakSet()

    def get_environment(self, renderer, scene):

        renderer_state_hash, scene_state_hash, state = get_hash_and_state(
            renderer, scene
        )
        env_hash = (renderer_state_hash, scene_state_hash)

        if env_hash in self.envs:
            env = self.envs[env_hash]
        else:
            env = Environment(renderer_state_hash, scene_state_hash)
            self.envs[env_hash] = env

        env.register(renderer, scene, state)

        # Cleanup. We remove all environments of which the associated renderer
        # or scene no longer exists, or their states have changed
        hashes_to_drop = []
        for env_hash, env in self.envs.items():
            if env.check_unused(renderer, scene, renderer_state_hash, scene_state_hash):
                hashes_to_drop.append(env_hash)
        for env_hash in hashes_to_drop:
            self.envs.pop(env_hash)

        return env


environment_manager = GlobalEnvironmentManager()
get_environment = environment_manager.get_environment


def get_hash_and_state(renderer, scene):

    state = {}

    # For renderer

    state["blender"] = renderer._blender
    renderer_hash = renderer.blend_mode

    # For scene

    lights = []

    def visit(ob):
        pass
        # if isinstance(ob, Light):
        #     lights.append(ob)

    scene.traverse(visit)

    state["lights"] = lights
    scene_hash = (len(lights),)

    return renderer_hash, scene_hash, state
