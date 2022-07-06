"""
The environment object defines details on how objects are rendered,
related to the environment.
"""

import weakref
import numpy as np

from ...utils.trackable import Trackable
from ...objects import Light, PointLight, DirectionalLight, SpotLight, AmbientLight
from ...resources import Buffer
from ...utils import array_from_shadertype, Color
from ...linalg import Matrix4, Vector3


class Environment(Trackable):
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

    _tmp_vector = Vector3()

    def __init__(self, renderer_state_hash, scene_state_hash):
        super().__init__()
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

        # Lights
        # seems don't need _store?
        self.ambient_lights_buffer = Buffer(array_from_shadertype(AmbientLight().uniform_type))

        # TODO: Get the light numbers in some other way, not hash?
        self.point_lights_num, self.dir_lights_num, self.spot_lights_num = scene_state_hash

        self.point_lights_buffer = None
        if self.point_lights_num>0:
            self.point_lights_buffer = Buffer(array_from_shadertype(PointLight().uniform_type, self.point_lights_num))

        self.directional_lights_buffer = None
        if self.dir_lights_num>0:
            self.directional_lights_buffer = Buffer(array_from_shadertype(DirectionalLight().uniform_type, self.dir_lights_num))

        self.spot_lights_buffer = None
        if self.spot_lights_num>0:
            self.spot_lights_buffer = Buffer(array_from_shadertype(SpotLight().uniform_type, self.spot_lights_num))


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
        if lights:
            self._setup_lights(lights)

    def _setup_lights(self, lights):

        ambient_lights_buffer = self.ambient_lights_buffer
        if not np.all(
            ambient_lights_buffer.data["color"][:3] == lights["ambient"]
        ):
            ambient_lights_buffer.data["color"].flat = lights["ambient"]
            ambient_lights_buffer.update_range(0, 1)

        # TODO: Use light attributes directly to setup final scene lights_buffer?

        # directional lights
        if self.dir_lights_num>0:
            dir_lights_buffer = self.directional_lights_buffer
            directional_lights = lights["directional_lights"]
            for i, light in enumerate(directional_lights):
                direction = self._tmp_vector.sub_vectors(
                    light.target.get_world_position(), light.get_world_position()
                ).normalize()
                light.uniform_buffer.data["direction"].flat = direction.to_array()

                if dir_lights_buffer.data[i] != light.uniform_buffer.data:
                    dir_lights_buffer.data[i] = light.uniform_buffer.data
                    dir_lights_buffer.update_range(i, 1)
                    light.uniform_buffer._pending_uploads = []

        # point lights
        if self.point_lights_num>0:
            point_lights_buffer = self.point_lights_buffer
            point_lights = lights["point_lights"]
            for i, light in enumerate(point_lights):
                if point_lights_buffer.data[i] != light.uniform_buffer.data:
                    point_lights_buffer.data[i] = light.uniform_buffer.data
                    point_lights_buffer.update_range(i, 1)
                    light.uniform_buffer._pending_uploads = []

        # spot lights
        if self.spot_lights_num>0:
            spot_lights_buffer = self.spot_lights_buffer
            spot_lights = lights["spot_lights"]
            for i, light in enumerate(spot_lights):
                direction = self._tmp_vector.sub_vectors(
                    light.target.get_world_position(), light.get_world_position()
                ).normalize()
                light.uniform_buffer.data["direction"].flat = direction.to_array()
                
                if spot_lights_buffer.data[i] != light.uniform_buffer.data:
                    spot_lights_buffer.data[i] = light.uniform_buffer.data
                    spot_lights_buffer.update_range(i, 1)
                    light.uniform_buffer._pending_uploads = []


    def register_pipeline_container(self, pipeline_container):
        """Allow pipeline containers to register, so that their
        env-specific wgpu objects can be removed.
        """
        self._pipeline_containers.add(pipeline_container)

    def check_inactive(self, renderer, scene, renderer_state_hash, scene_state_hash):
        """Do some clean-up for the given renderer and scene."""
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
        """Remove all environments of which the associated renderer
        or scene no longer exist, or their states have changed.
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

    point_lights = []
    directional_lights = []
    spot_lights = []
    ambient_color = [0, 0, 0]

    def visit(ob):
        if isinstance(ob, PointLight):
            point_lights.append(ob)
        elif isinstance(ob, DirectionalLight):
            directional_lights.append(ob)
        elif isinstance(ob, SpotLight):
            spot_lights.append(ob)
        elif isinstance(ob, AmbientLight):
            ambient_color[0] += ob.uniform_buffer.data["color"][0]
            ambient_color[1] += ob.uniform_buffer.data["color"][1]
            ambient_color[2] += ob.uniform_buffer.data["color"][2]

    scene.traverse(visit, True)

    state["lights"] = {
        "point_lights": point_lights,
        "directional_lights": directional_lights,
        "spot_lights": spot_lights,
        "ambient": ambient_color
    }

    scene_hash = (len(point_lights), len(directional_lights), len(spot_lights),)

    return renderer_hash, scene_hash, state
