"""
The environment object defines details on how objects are rendered,
related to the environment.
"""

import weakref
import numpy as np
import wgpu

from ...utils.trackable import Trackable
from ...objects import (
    PointLight,
    DirectionalLight,
    SpotLight,
    AmbientLight,
)
from ...resources import Buffer
from ...utils import array_from_shadertype
from ._pipeline import Binding
from ._update import update_buffer
from ._utils import generate_uniform_struct


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

    _ambient_uniform_type = AmbientLight().uniform_type
    _point_uniform_type = PointLight().uniform_type
    _dir_uniform_type = DirectionalLight().uniform_type
    _spot_uniform_type = SpotLight().uniform_type

    def __init__(self, renderer_state_hash, scene_state_hash, device):
        super().__init__()
        # The hash consists of two parts. It does not change.
        self._renderer_state_hash = renderer_state_hash
        self._scene_state_hash = scene_state_hash

        self.device = device
        # Keep track of all renders and scenes that make use of this
        # environment, so that we can detect that the env has become
        # inactive.
        self._renderers = weakref.WeakSet()
        self._scenes = weakref.WeakSet()
        # keep track of all pipeline containers that have objects for this
        # environment, so that we can remove those objects when this env
        # becomes inactive.
        self._pipeline_containers = weakref.WeakSet()

        self.wgpu_bind_group = None

        # self.resources = None
        # TODO: make this configurable
        self.shadow_map_size = (1024, 1024)

        self.bindings = []

        # Lights
        self._setup_light_resources()

        if self.bindings:
            self._collect_resources()

    def _setup_light_resources(self):

        self.ambient_lights_buffer = Buffer(
            array_from_shadertype(self._ambient_uniform_type)
        )

        (
            self.point_lights_num,
            self.dir_lights_num,
            self.spot_lights_num,
        ) = self._scene_state_hash

        self.bindings.append(
            Binding(
                "u_ambient_light",
                "buffer/uniform",
                self.ambient_lights_buffer,
                structname="AmbientLight",
            )
        )

        self.directional_lights_buffer = None
        self.directional_lights_shadow_texture = None
        if self.dir_lights_num > 0:
            self.directional_lights_buffer = Buffer(
                array_from_shadertype(self._dir_uniform_type, self.dir_lights_num)
            )
            self.bindings.append(
                Binding(
                    "u_directional_lights",
                    "buffer/uniform",
                    self.directional_lights_buffer,
                    structname="DirectionalLight",
                )
            )

            self.directional_lights_shadow_texture = self.device.create_texture(
                size=(
                    self.shadow_map_size[0],
                    self.shadow_map_size[1],
                    self.dir_lights_num,
                ),
                usage=wgpu.TextureUsage.RENDER_ATTACHMENT
                | wgpu.TextureUsage.TEXTURE_BINDING,
                format="depth32float",
            )

            self.bindings.append(
                Binding(
                    "u_shadow_map_dir_light",
                    "shadow_texture/2d-array",
                    self.directional_lights_shadow_texture.create_view(
                        dimension="2d-array"
                    ),
                )
            )

        self.point_lights_buffer = None
        self.point_lights_shadow_texture = None
        if self.point_lights_num > 0:
            self.point_lights_buffer = Buffer(
                array_from_shadertype(self._point_uniform_type, self.point_lights_num)
            )
            self.bindings.append(
                Binding(
                    "u_point_lights",
                    "buffer/uniform",
                    self.point_lights_buffer,
                    structname="PointLight",
                )
            )

            self.point_lights_shadow_texture = self.device.create_texture(
                size=(
                    self.shadow_map_size[0],
                    self.shadow_map_size[1],
                    self.point_lights_num * 6,
                ),
                usage=wgpu.TextureUsage.RENDER_ATTACHMENT
                | wgpu.TextureUsage.TEXTURE_BINDING,
                format="depth32float",
            )

            self.bindings.append(
                Binding(
                    "u_shadow_map_point_light",
                    "shadow_texture/cube-array",
                    self.point_lights_shadow_texture.create_view(
                        dimension="cube-array"
                    ),
                )
            )

        self.spot_lights_buffer = None
        self.spot_lights_shadow_texture = None
        if self.spot_lights_num > 0:
            self.spot_lights_buffer = Buffer(
                array_from_shadertype(self._spot_uniform_type, self.spot_lights_num)
            )
            self.bindings.append(
                Binding(
                    "u_spot_lights",
                    "buffer/uniform",
                    self.spot_lights_buffer,
                    structname="SpotLight",
                )
            )

            self.spot_lights_shadow_texture = self.device.create_texture(
                size=(
                    self.shadow_map_size[0],
                    self.shadow_map_size[1],
                    self.spot_lights_num,
                ),
                usage=wgpu.TextureUsage.RENDER_ATTACHMENT
                | wgpu.TextureUsage.TEXTURE_BINDING,
                format="depth32float",
            )

            self.bindings.append(
                Binding(
                    "u_shadow_map_spot_light",
                    "shadow_texture/2d-array",
                    self.spot_lights_shadow_texture.create_view(dimension="2d-array"),
                )
            )

        if self.dir_lights_num + self.point_lights_num + self.spot_lights_num > 0:
            self.shadow_sampler = self.device.create_sampler(
                mag_filter=wgpu.FilterMode.linear,
                min_filter=wgpu.FilterMode.linear,
                compare=wgpu.CompareFunction.less_equal,
            )

            self.bindings.append(
                Binding(
                    f"u_shadow_sampler",
                    "shadow_sampler/comparison",
                    self.shadow_sampler,
                )
            )


    def _collect_resources(self):
        bg_descriptor = []
        bg_layout_descriptor = []

        for index, binding in enumerate(self.bindings):

            if binding.type.startswith("buffer/"):
                binding.resource._wgpu_usage |= wgpu.BufferUsage.UNIFORM
                update_buffer(self.device, binding.resource)

            binding_des, binding_layout_des = binding.get_bind_group_descriptors(index)

            bg_descriptor.append(binding_des)
            bg_layout_descriptor.append(binding_layout_des)

        bind_group_layout = self.device.create_bind_group_layout(
            entries=bg_layout_descriptor
        )

        bind_group = self.device.create_bind_group(
            layout=bind_group_layout, entries=bg_descriptor
        )
        self.wgpu_bind_group = (bind_group_layout, bind_group)

    def get_light_structs_code(self):
        light_struct = []
        for binding in self.bindings:
            if binding.type.startswith("buffer/"):
                struct_code = generate_uniform_struct(
                    binding.resource.data.dtype, binding.structname
                )
                light_struct.append(struct_code)

        return "\n".join(light_struct)

    def _define_shadow_texture(self, bind_group_index, index, binding):
        texture_view = binding.resource  # wgpu.TextureView

        dim = "2d"
        if texture_view.size[2] == 1:
            dim = "2d"
        elif texture_view.size[2] == 6:
            dim = "cube"

        code = f"""
        @group({ bind_group_index }) @binding({index})
        var {binding.name}: texture_depth_{dim}_array;
        """.rstrip()
        return code

    def _define_shadow_sampler(self, bind_group_index, index, binding):
        code = f"""
        @group({ bind_group_index }) @binding({index})
        var {binding.name}: sampler_comparison;
        """.rstrip()
        return code

    def get_light_vars_code(self, bind_group_index):
        codes = []
        for slot, binding in enumerate(self.bindings):
            if binding.type.startswith("buffer/"):
                uniform_type = (
                    f"array<{binding.structname}, {binding.resource.data.shape[0]}>"  # array of struct
                    if isinstance(binding.resource, Buffer)
                    and binding.resource.data.shape  # Buffer.items > 1
                    else binding.structname
                )

                code = f"""
            @group({bind_group_index}) @binding({slot})
            var<uniform> {binding.name}: {uniform_type};
            """.rstrip()
                codes.append(code)
            elif binding.type.startswith("shadow_texture/"):
                code = self._define_shadow_texture(bind_group_index, slot, binding)
                codes.append(code)
            elif binding.type.startswith("shadow_sampler/"):
                code = self._define_shadow_sampler(bind_group_index, slot, binding)
                codes.append(code)

        return "\n".join(codes)

    def get_shader_kwargs(self, bind_group_index=1):
        args = {
            "num_dir_lights": self.dir_lights_num,
            "num_spot_lights": self.spot_lights_num,
            "num_point_lights": self.point_lights_num,
            "light_structs": self.get_light_structs_code(),
            "light_vars": self.get_light_vars_code(bind_group_index),
        }
        return args

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
        self.lights = lights

        if lights:
            self._update_light_buffers(lights)

    def _update_light_buffers(self, lights):
        """Update the contents of the uniform buffers for the lights, and create texture views if needed."""

        device = self.device

        ambient_lights_buffer = self.ambient_lights_buffer
        if not np.all(ambient_lights_buffer.data["color"][:3] == lights["ambient"]):
            ambient_lights_buffer.data["color"].flat = lights["ambient"]
            ambient_lights_buffer.update_range(0, 1)

        # We only need to update buffers once before each draw
        update_buffer(device, ambient_lights_buffer)

        # TODO: Use light attributes directly to setup final lights_buffer?

        # directional lights
        if self.dir_lights_num > 0:
            dir_lights_buffer = self.directional_lights_buffer
            directional_lights = lights["directional_lights"]
            for i, light in enumerate(directional_lights):
                light.update_uniform_buffer()
                if light.cast_shadow:
                    light.shadow.update_uniform_buffers(light)

                    if light.shadow.map is None or light.shadow.map_index != i:

                        light.shadow.map = (
                            self.directional_lights_shadow_texture.create_view(
                                base_array_layer=i
                            )
                        )
                        light.shadow.map_index = i

                if dir_lights_buffer.data[i] != light.uniform_buffer.data:
                    dir_lights_buffer.data[i] = light.uniform_buffer.data
                    dir_lights_buffer.update_range(i, 1)
                    light.uniform_buffer._pending_uploads = []

            update_buffer(device, dir_lights_buffer)

        # point lights
        if self.point_lights_num > 0:
            point_lights_buffer = self.point_lights_buffer
            point_lights = lights["point_lights"]
            for i, light in enumerate(point_lights):
                light.update_uniform_buffer()
                if light.cast_shadow:
                    light.shadow.update_uniform_buffers(light)

                    if light.shadow.map is None or light.shadow.map_index != i:
                        light.shadow.map = []
                        for face in range(6):
                            light.shadow.map.append(
                                self.point_lights_shadow_texture.create_view(
                                    base_array_layer=i * 6 + face
                                )
                            )

                        light.shadow.map_index = i

                if point_lights_buffer.data[i] != light.uniform_buffer.data:
                    point_lights_buffer.data[i] = light.uniform_buffer.data
                    point_lights_buffer.update_range(i, 1)
                    light.uniform_buffer._pending_uploads = []

            update_buffer(device, point_lights_buffer)

        # spot lights
        if self.spot_lights_num > 0:
            spot_lights_buffer = self.spot_lights_buffer
            spot_lights = lights["spot_lights"]
            for i, light in enumerate(spot_lights):
                light.update_uniform_buffer()
                if light.cast_shadow:
                    light.shadow.update_uniform_buffers(light)

                    if light.shadow.map is None or light.shadow.map_index != i:

                        light.shadow.map = self.spot_lights_shadow_texture.create_view(
                            base_array_layer=i
                        )
                        light.shadow.map_index = i

                if spot_lights_buffer.data[i] != light.uniform_buffer.data:
                    spot_lights_buffer.data[i] = light.uniform_buffer.data
                    spot_lights_buffer.update_range(i, 1)
                    light.uniform_buffer._pending_uploads = []

            update_buffer(device, spot_lights_buffer)

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
            env = Environment(renderer_state_hash, scene_state_hash, renderer.device)
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
        "ambient": ambient_color,
    }

    scene_hash = (
        len(point_lights),
        len(directional_lights),
        len(spot_lights),
    )

    return renderer_hash, scene_hash, state
