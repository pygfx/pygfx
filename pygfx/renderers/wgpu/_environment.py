"""
The environment object defines details on how objects are rendered,
related to the environment.
"""

import weakref


class Environment:
    """Object that represents the state of the "environment". With
    environment we mean the stuff - other than the wobject itself -
    that affects the rendering, like renderer state, and lights.

    An environment object represents a "state type". It's attributes
    will be changed by the renderer for each draw, but these changes
    will be compatible with the source and pipeline. As an example, the
    number of lights affects the format of the uniform, so it's part
    of the hash, but the color of the lights do not matter for the
    pipeline.
    """

    def __init__(self, renderer_state_hash, scene_state_hash):
        # The hash consists of two parts. It does not change.
        self._renderer_state_hash = renderer_state_hash
        self._scene_state_hash = scene_state_hash
        # Keep track of all renders and scenes that make use of this
        # environment, so that we can detect that the env has become
        # inactive.
        self._renderers = weakref.WeakSet()
        self._scenes = weakref.WeakSet()
        # keep track of all pipeline containers that have objects for this
        # environment, so that we can remove those objects when this env
        # becomes inactive.
        self._pipeline_containers = weakref.WeakSet()

        # Something like this
        # self.uniform_buffer = Buffer(array_from_shadertype(stdinfo_uniform_type))

    @property
    def hash(self):
        """The full hash for this environment."""
        return self._renderer_state_hash, self._scene_state_hash

    def update(self, renderer, scene, blender=None, lights=None):
        """Register a renderer and scene to use this environment,
        and update it with the given state.
        """
        # Register
        self._renderers.add(renderer)
        self._scenes.add(scene)

        # Update
        self.blender = blender
        # Note: when we implement lights, this is where we'd update the uniform(s)
        # self.uniform_buffer.data[xx] = yy

    def register_pipeline_container(self, pipeline_container):
        """Allow pipeline containers to register, so that their
        env-specific wgpu objects can be removed.
        """
        self._pipeline_containers.add(pipeline_container)

    def check_inactive(self, renderer, scene, renderer_state_hash, scene_state_hash):
        """So some clean-up for the given renderer and scene,"""
        renderers = set(self._renderers)
        scenes = set(self._scenes)

        if renderer in renderers:
            if renderer_state_hash != self._renderer_state_hash:
                self._renderers.discard(renderer)
                renderers.discard(renderer)

        if scene in scenes:
            if scene_state_hash != self._scene_state_hash:
                self._scenes.discard(scene)
                scenes.discard(renderer)

        if not renderers or not scenes:
            self.clear()
            return True

    def clear(self):
        """Remove all wgpu objects associated with this environment."""
        for pipeline_container in self._pipeline_containers:
            pipeline_container.remove_env_hash(self.hash)
        self._pipeline_containers.clear()
        self._renderers.clear()
        self._scenes.clear()


class GlobalEnvironmentManager:
    """A little class to manage the different environments."""

    def __init__(self):
        self.environments = {}  # hash -> Environment

    def get_environment(self, renderer, scene):
        """The main entrypoint. The renderer uses this to obtain an
        environment object.
        """
        renderer_state_hash, scene_state_hash, state = get_hash_and_state(
            renderer, scene
        )
        env_hash = (renderer_state_hash, scene_state_hash)

        # Re-use or create an environment
        if env_hash in self.environments:
            env = self.environments[env_hash]
        else:
            env = Environment(renderer_state_hash, scene_state_hash)
            assert env.hash == env_hash
            self.environments[env_hash] = env

        # Update the environment
        env.update(renderer, scene, **state)

        # Cleanup
        self._cleanup(renderer, scene, renderer_state_hash, scene_state_hash)

        return env

    def _cleanup(self, renderer, scene, renderer_state_hash, scene_state_hash):
        """ " Remove all environments of which the associated renderer
        or scene no longer exists, or their states have changed.
        """
        hashes_to_drop = []
        for env_hash, env in self.environments.items():
            if env.check_inactive(
                renderer, scene, renderer_state_hash, scene_state_hash
            ):
                hashes_to_drop.append(env_hash)
        for env_hash in hashes_to_drop:
            self.environments.pop(env_hash)


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
