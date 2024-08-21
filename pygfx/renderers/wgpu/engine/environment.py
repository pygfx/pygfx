"""
The environment object defines details on how objects are rendered,
related to the environment.
"""

import weakref
import numpy as np
import wgpu

from ....utils.trackable import Trackable
from ....objects import (
    PointLight,
    DirectionalLight,
    SpotLight,
    AmbientLight,
)
from ....resources import Buffer
from ....utils import array_from_shadertype

from .pipeline import Binding
from .utils import generate_uniform_struct
from .shared import get_shared


class Environment(Trackable):
    """Object for internal use that represents the state of the
    "environment". With environment we mean the stuff - other than the
    wobject itself - that affects the rendering, like renderer state,
    and lights.

    An environment object represents a "state type". It's attributes
    will be changed by the renderer for each draw, but these changes
    will be compatible with the source and pipeline. As an example, the
    number of lights affects the format of the uniform, so it's part
    of the hash, but the color of the lights do not matter for the
    pipeline.
    """

    _ambient_uniform_type = AmbientLight.uniform_type
    _point_uniform_type = PointLight.uniform_type
    _dir_uniform_type = DirectionalLight.uniform_type
    _spot_uniform_type = SpotLight.uniform_type

    def __init__(self, renderer_state_hash, scene_state_hash):
        super().__init__()
        # The hash consists of two parts. It does not change.
        self._renderer_state_hash = renderer_state_hash
        self._scene_state_hash = scene_state_hash

        self.device = get_shared().device

        # Keep track of all renders and scenes that make use of this
        # environment, so that we can detect that the env has become
        # inactive.
        self._renderers = weakref.WeakSet()
        self._scenes = weakref.WeakSet()

        # keep track of all pipeline containers that have objects for this
        # environment, so that we can remove those objects when this env
        # becomes inactive.
        self._pipeline_containers = weakref.WeakSet()

        # Note: we could make this configurable. Would probably have to be a global setting.
        self.shadow_map_size = (1024, 1024)

        # List of binding objects
        self.bindings = []

        # The wgpu bind group to collect the bindings in
        self.wgpu_bind_group = None

        # Init
        self._setup_light_resources()
        if self.bindings:
            self.wgpu_bind_group = self._collect_bindings()

    def _setup_light_resources(self):
        (
            self.point_lights_num,
            self.dir_lights_num,
            self.spot_lights_num,
        ) = self._scene_state_hash

        # The ambient light binding is easy, because ambient lights can be combined on CPU

        self.ambient_lights_buffer = Buffer(
            array_from_shadertype(self._ambient_uniform_type), force_contiguous=True
        )
        self.bindings.append(
            Binding(
                "u_ambient_light",
                "buffer/uniform",
                self.ambient_lights_buffer,
                structname="AmbientLight",
            )
        )

        # A bit more work for the directional lights

        self.directional_lights_buffer = None
        self.directional_lights_shadow_texture = None
        if self.dir_lights_num > 0:
            self.directional_lights_buffer = Buffer(
                array_from_shadertype(self._dir_uniform_type, self.dir_lights_num),
                force_contiguous=True,
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
            self.directional_lights_shadow_texture_views = [
                self.directional_lights_shadow_texture.create_view(base_array_layer=i)
                for i in range(self.dir_lights_num)
            ]
            self.bindings.append(
                Binding(
                    "u_shadow_map_dir_light",
                    "shadow_texture/2d-array",
                    self.directional_lights_shadow_texture.create_view(
                        dimension="2d-array"
                    ),
                )
            )

        # Similar logic for the point lights

        self.point_lights_buffer = None
        self.point_lights_shadow_texture = None
        if self.point_lights_num > 0:
            self.point_lights_buffer = Buffer(
                array_from_shadertype(self._point_uniform_type, self.point_lights_num),
                force_contiguous=True,
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
            self.point_lights_shadow_texture_views = [
                self.point_lights_shadow_texture.create_view(base_array_layer=i)
                for i in range(self.point_lights_num * 6)
            ]
            self.bindings.append(
                Binding(
                    "u_shadow_map_point_light",
                    "shadow_texture/cube-array",
                    self.point_lights_shadow_texture.create_view(
                        dimension="cube-array"
                    ),
                )
            )

        # And the spot lights

        self.spot_lights_buffer = None
        self.spot_lights_shadow_texture = None
        if self.spot_lights_num > 0:
            self.spot_lights_buffer = Buffer(
                array_from_shadertype(self._spot_uniform_type, self.spot_lights_num),
                force_contiguous=True,
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
            self.spot_lights_shadow_texture_views = [
                self.spot_lights_shadow_texture.create_view(base_array_layer=i)
                for i in range(self.spot_lights_num)
            ]
            self.bindings.append(
                Binding(
                    "u_shadow_map_spot_light",
                    "shadow_texture/2d-array",
                    self.spot_lights_shadow_texture.create_view(dimension="2d-array"),
                )
            )

        # We only need a sampled if we have shadow-casting lights

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

    def _collect_bindings(self):
        """Group the bindings into a wgpu bind group."""
        bg_descriptor = []
        bg_layout_descriptor = []
        device = self.device

        for index, binding in enumerate(self.bindings):
            if binding.type.startswith("buffer/"):
                binding.resource._wgpu_usage |= wgpu.BufferUsage.UNIFORM
            binding_des, binding_layout_des = binding.get_bind_group_descriptors(index)
            bg_descriptor.append(binding_des)
            bg_layout_descriptor.append(binding_layout_des)

        bind_group_layout = device.create_bind_group_layout(
            entries=bg_layout_descriptor
        )
        bind_group = device.create_bind_group(
            layout=bind_group_layout, entries=bg_descriptor
        )
        return bind_group_layout, bind_group

    def _get_light_structs_code(self):
        """Generate the wgsl that defines the light structs."""
        light_struct = []
        for binding in self.bindings:
            if binding.type.startswith("buffer/"):
                struct_code = generate_uniform_struct(
                    binding.resource.data.dtype, binding.structname
                )
                light_struct.append(struct_code)
        return "\n".join(light_struct)

    def _get_light_vars_code(self, bind_group_index):
        """Generate the wgsl that defines the uniforms, textures and sampler bindings."""
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
                dim = binding.type.split("/")[-1].replace("-", "_")
                code = f"""
                @group({bind_group_index}) @binding({slot})
                var {binding.name}: texture_depth_{dim};
                """.rstrip()
                codes.append(code)
            elif binding.type.startswith("shadow_sampler/"):
                code = f"""
                @group({bind_group_index}) @binding({slot})
                var {binding.name}: sampler_comparison;
                """.rstrip()
                codes.append(code)

        return "\n".join(codes)

    def get_shader_kwargs(self, bind_group_index=1):
        """Get shader template kwargs specific to the environment.
        Used by the pipeline to complete the shader.
        """
        light_definitions = self._get_light_structs_code()
        light_definitions += self._get_light_vars_code(bind_group_index)
        args = {
            "num_dir_lights": self.dir_lights_num,
            "num_spot_lights": self.spot_lights_num,
            "num_point_lights": self.point_lights_num,
            "light_definitions": light_definitions,
        }
        return args

    @property
    def hash(self):
        """The full hash for this environment."""
        return self._renderer_state_hash, self._scene_state_hash

    def update(self, renderer, scene, blender=None, lights=None):
        """Update the content of the environment object.

        An environment is shared between all renderers/scenes for which
        the environment hash is a match (i.e. the number of light
        matches). This method registers the renderer and scene, and
        updates the uniform buffers.
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
        """Update the contents of the uniform buffers for the lights,
        and create texture views if needed.
        """

        # Update ambient buffer

        ambient_lights_buffer = self.ambient_lights_buffer
        if not np.all(ambient_lights_buffer.data["color"][:3] == lights["ambient"]):
            ambient_lights_buffer.data["color"].flat = lights["ambient"]
            ambient_lights_buffer.update_range(0, 1)

        # We update the uniform buffers of the lights below. These buffers
        # are not actually used directly but copied to the environment's buffer.
        # Seems like a detour, but kindof the simplest solution still.

        # Update directional light buffers

        if self.dir_lights_num > 0:
            dir_lights_buffer = self.directional_lights_buffer
            directional_lights = lights["directional_lights"]
            for i, light in enumerate(directional_lights):
                light._gfx_update_uniform_buffer()
                if light.cast_shadow:
                    light.shadow._gfx_update_uniform_buffer(light)
                    light.shadow._wgpu_tex_view = (
                        self.directional_lights_shadow_texture_views[i]
                    )
                if dir_lights_buffer.data[i] != light.uniform_buffer.data:
                    dir_lights_buffer.data[i] = light.uniform_buffer.data
                    dir_lights_buffer.update_range(i, 1)

        # Update point light buffers
        if self.point_lights_num > 0:
            point_lights_buffer = self.point_lights_buffer
            point_lights = lights["point_lights"]
            for i, light in enumerate(point_lights):
                light._gfx_update_uniform_buffer()
                if light.cast_shadow:
                    light.shadow._gfx_update_uniform_buffer(light)
                    light.shadow._wgpu_tex_view = [
                        self.point_lights_shadow_texture_views[i * 6 + face]
                        for face in range(6)
                    ]
                if point_lights_buffer.data[i] != light.uniform_buffer.data:
                    point_lights_buffer.data[i] = light.uniform_buffer.data
                    point_lights_buffer.update_range(i, 1)

        if self.spot_lights_num > 0:
            spot_lights_buffer = self.spot_lights_buffer
            spot_lights = lights["spot_lights"]
            for i, light in enumerate(spot_lights):
                light._gfx_update_uniform_buffer()
                if light.cast_shadow:
                    light.shadow._gfx_update_uniform_buffer(light)
                    light.shadow._wgpu_tex_view = self.spot_lights_shadow_texture_views[
                        i
                    ]
                if spot_lights_buffer.data[i] != light.uniform_buffer.data:
                    spot_lights_buffer.data[i] = light.uniform_buffer.data
                    spot_lights_buffer.update_range(i, 1)

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
            r, g, b = ob.color.to_physical()
            ambient_color[0] += r * ob.intensity
            ambient_color[1] += g * ob.intensity
            ambient_color[2] += b * ob.intensity

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
