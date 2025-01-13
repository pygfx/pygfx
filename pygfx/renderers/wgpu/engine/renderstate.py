"""
The renderstate object defines details on how objects are rendered,
related to statefull stuff like lights in the scene and blend mode.
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


# This cache is only used so renderstate objects can be re-used.
# The instances must be stored somewhere to prevent them from being deleted.
_renderstate_instance_cache = weakref.WeakValueDictionary()


def get_renderstate(scene, blender):
    """Get the renderstate object for the given scene and blender."""

    # Convert args to states
    light_state = _scene_to_light_state(scene)
    blend_state = (blender,)
    combined_state = light_state, blend_state

    # Create renderstate object and prepare it with the current state
    hash = CombinedRenderState.state_to_hash(*combined_state)
    ob = CombinedRenderState.obtain_from_hash(hash)
    ob.prepare_for_draw(*combined_state)

    return ob


def _scene_to_light_state(scene):
    """Convert scene object to light state.
    If we introduce more renderstates based on the scene, we should probably update this function,
    so we need to traverse it just once.
    """
    point_lights = []
    directional_lights = []
    spot_lights = []
    ambient_color = [0, 0, 0]

    # todo: would be nice to have a flag to indicate on each wobject when was the last time that its downstream graph changed, so we can usually skip this step!
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

    light_state = (
        point_lights,
        directional_lights,
        spot_lights,
        ambient_color,
    )
    return light_state


# ----------


class BaseRenderState:
    """Object for internal use, that represents a certain state related to
    rendering, that affects the shader and/or pipeline.

    We distinguish the following three concepts: args -> state -> hash.

    The args is the incoming argument, e.g. the scene object. The state
    is derived from that, e.g. lists of light objects.
    """

    # todo: must this be a Trackable?

    _hash = None

    @classmethod
    def obtain_from_hash(cls, hash):
        # Could do this in an __init__, but I like how this is more explicit
        ob = _renderstate_instance_cache.get(hash, None)
        if ob is None:
            ob = cls(*hash)
            ob._hash = hash
            _renderstate_instance_cache[hash] = ob
        return ob

    @property
    def hash(self):
        return self._hash

    # subclasses must implement

    @classmethod
    def state_to_hash(cls, *state):
        """Convert state-tuple to a hash-tuple."""
        raise NotImplementedError()

    def __init__(self, *hash):
        raise NotImplementedError()

    def prepare_for_draw(self, *state):
        raise NotImplementedError()

    def get_shader_kwargs(self, bind_group_index=1):
        raise NotImplementedError()


class LightRenderState(BaseRenderState):
    """Represents the rendering state for a specific number of lights (of each type)."""

    @classmethod
    def state_to_hash(
        cls, point_lights, directional_lights, spot_lights, ambient_color
    ):
        return (len(point_lights), len(directional_lights), len(spot_lights))

    def __init__(self, point_lights_count, dir_lights_count, spot_lights_count):
        self.device = get_shared().device

        # Store counts
        self.point_lights_count = point_lights_count
        self.dir_lights_count = dir_lights_count
        self.spot_lights_count = spot_lights_count

        # Note: we could make this configurable. Would probably have to be a global setting.
        self.shadow_map_size = (1024, 1024)

        # Create list of binding objects
        self.bindings = []
        self._setup_light_resources()

    def _setup_light_resources(self):
        # The ambient light binding is easy, because ambient lights can be combined on CPU

        self.ambient_lights_buffer = Buffer(
            array_from_shadertype(AmbientLight.uniform_type), force_contiguous=True
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
        if self.dir_lights_count > 0:
            self.directional_lights_buffer = Buffer(
                array_from_shadertype(
                    DirectionalLight.uniform_type, self.dir_lights_count
                ),
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
                    self.dir_lights_count,
                ),
                usage=wgpu.TextureUsage.RENDER_ATTACHMENT
                | wgpu.TextureUsage.TEXTURE_BINDING,
                format="depth32float",
            )
            self.directional_lights_shadow_texture_views = [
                self.directional_lights_shadow_texture.create_view(base_array_layer=i)
                for i in range(self.dir_lights_count)
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
        if self.point_lights_count > 0:
            self.point_lights_buffer = Buffer(
                array_from_shadertype(PointLight.uniform_type, self.point_lights_count),
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
                    self.point_lights_count * 6,
                ),
                usage=wgpu.TextureUsage.RENDER_ATTACHMENT
                | wgpu.TextureUsage.TEXTURE_BINDING,
                format="depth32float",
            )
            self.point_lights_shadow_texture_views = [
                self.point_lights_shadow_texture.create_view(base_array_layer=i)
                for i in range(self.point_lights_count * 6)
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
        if self.spot_lights_count > 0:
            self.spot_lights_buffer = Buffer(
                array_from_shadertype(SpotLight.uniform_type, self.spot_lights_count),
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
                    self.spot_lights_count,
                ),
                usage=wgpu.TextureUsage.RENDER_ATTACHMENT
                | wgpu.TextureUsage.TEXTURE_BINDING,
                format="depth32float",
            )
            self.spot_lights_shadow_texture_views = [
                self.spot_lights_shadow_texture.create_view(base_array_layer=i)
                for i in range(self.spot_lights_count)
            ]
            self.bindings.append(
                Binding(
                    "u_shadow_map_spot_light",
                    "shadow_texture/2d-array",
                    self.spot_lights_shadow_texture.create_view(dimension="2d-array"),
                )
            )

        # We only need a sampler if we have shadow-casting lights

        if self.dir_lights_count + self.point_lights_count + self.spot_lights_count > 0:
            self.shadow_sampler = self.device.create_sampler(
                mag_filter=wgpu.FilterMode.linear,
                min_filter=wgpu.FilterMode.linear,
                compare=wgpu.CompareFunction.less_equal,
            )
            self.bindings.append(
                Binding(
                    "u_shadow_sampler",
                    "shadow_sampler/comparison",
                    self.shadow_sampler,
                )
            )

    def get_shader_kwargs(self, bind_group_index=1):
        """Get shader template kwargs specific to the renderstate.
        Used by the pipeline to complete the shader.
        """
        light_definitions = self._get_light_structs_code()
        light_definitions += self._get_light_vars_code(bind_group_index)
        kwargs = {
            "num_dir_lights": self.dir_lights_count,
            "num_spot_lights": self.spot_lights_count,
            "num_point_lights": self.point_lights_count,
            "light_definitions": light_definitions,
        }
        return kwargs

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

    def prepare_for_draw(
        self, point_lights, directional_lights, spot_lights, ambient_color
    ):
        """Update the contents of the uniform buffers for the lights,
        and create texture views if needed.
        """

        # Make light info available to other parts of the pygfx engine
        self.lights = {
            "point_lights": point_lights,
            "directional_lights": directional_lights,
            "spot_lights": spot_lights,
            "ambient_color": ambient_color,
        }

        # Update ambient buffer

        ambient_lights_buffer = self.ambient_lights_buffer
        if not np.all(ambient_lights_buffer.data["color"][:3] == ambient_color):
            ambient_lights_buffer.data["color"].flat = ambient_color
            ambient_lights_buffer.update_range(0, 1)

        # We update the uniform buffers of the lights below. These buffers
        # are not actually used directly but copied to the renderstate's buffer.
        # Seems like a detour, but kindof the simplest solution still.

        # Update directional light buffers

        if self.dir_lights_count > 0:
            dir_lights_buffer = self.directional_lights_buffer
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
        if self.point_lights_count > 0:
            point_lights_buffer = self.point_lights_buffer
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

        if self.spot_lights_count > 0:
            spot_lights_buffer = self.spot_lights_buffer
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


class BlendRenderState(BaseRenderState):
    """Represents the rendering state for a specific blend mode."""

    @classmethod
    def state_to_hash(cls, blender):
        return (blender.name,)

    def __init__(self, blend_mode):
        self.blend_mode = blend_mode
        self.bindings = []

    def prepare_for_draw(self, blender):
        # Make blender available for other parts or the pygfx engine
        self.blender = blender

    def get_shader_kwargs(self, bind_group_index=1):
        return {"blend_mode": self.blend_mode}


class CombinedRenderState(BaseRenderState):
    """Represents the combined rendering state."""

    @classmethod
    def state_to_hash(cls, light_state, blend_state):
        light_hash = LightRenderState.state_to_hash(*light_state)
        blend_hash = BlendRenderState.state_to_hash(*blend_state)

        return light_hash, blend_hash

    def __init__(self, light_hash, blend_hash):
        self.device = get_shared().device

        self.light_renderstate = LightRenderState.obtain_from_hash(light_hash)
        self.blend_renderstate = BlendRenderState.obtain_from_hash(blend_hash)

        self.bindings = []
        self.bindings += self.light_renderstate.bindings
        self.bindings += self.blend_renderstate.bindings

        self.wgpu_bind_group_layout = None
        self.wgpu_bind_group = None
        if self.bindings:
            self.wgpu_bind_group_layout, self.wgpu_bind_group = self._collect_bindings()

    def prepare_for_draw(self, light_state, blend_state):
        # Delegate
        self.light_renderstate.prepare_for_draw(*light_state)
        self.blend_renderstate.prepare_for_draw(*blend_state)

        # Make state available for other parts or the pygfx engine
        self.lights = self.light_renderstate.lights
        self.blender = self.blend_renderstate.blender

    def get_shader_kwargs(self, bind_group_index=1):
        kwargs = {}
        kwargs.update(self.light_renderstate.get_shader_kwargs(bind_group_index))
        kwargs.update(self.blend_renderstate.get_shader_kwargs(bind_group_index))
        return kwargs

    # Stuff specific for the combined renderstate

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

    # def register_pipeline_container(self, pipeline_container):
    #     """Allow pipeline containers to register, so that their
    #     env-specific wgpu objects can be removed.
    #     """
    #     self._pipeline_containers.add(pipeline_container)

    # def check_inactive(self, renderer, scene, renderer_state_hash, scene_state_hash):
    #     """Do some clean-up for the given renderer and scene."""
    #     renderers = set(self._renderers)
    #     scenes = set(self._scenes)

    #     if renderer in renderers:
    #         if renderer_state_hash != self._renderer_state_hash:
    #             self._renderers.discard(renderer)
    #             renderers.discard(renderer)

    #     if scene in scenes:
    #         if scene_state_hash != self._scene_state_hash:
    #             self._scenes.discard(scene)
    #             scenes.discard(renderer)

    #     if not renderers or not scenes:
    #         self.clear()
    #         return True

    # def clear(self):
    #     """Remove all wgpu objects associated with this environment."""
    #     for pipeline_container in self._pipeline_containers:
    #         pipeline_container.remove_env_hash(self.hash)
    #     self._pipeline_containers.clear()
    #     self._renderers.clear()
    #     self._scenes.clear()


# class GlobalEnvironmentManager:
#     """A little class to manage the different environments."""

#     def __init__(self):
#         self.environments = {}  # hash -> Environment

#     def get_environment(self, renderer, scene):
#         """The main entrypoint. The renderer uses this to obtain an
#         environment object.
#         """
#         renderer_state_hash, scene_state_hash, state = get_hash_and_state(
#             renderer, scene
#         )
#         env_hash = (renderer_state_hash, scene_state_hash)

#         # Re-use or create an environment
#         if env_hash in self.environments:
#             env = self.environments[env_hash]
#         else:
#             env = Environment(renderer_state_hash, scene_state_hash)
#             assert env.hash == env_hash
#             self.environments[env_hash] = env

#         # Update the environment
#         env.update(renderer, scene, **state)

#         # Cleanup
#         self._cleanup(renderer, scene, renderer_state_hash, scene_state_hash)

#         return env

#     def _cleanup(self, renderer, scene, renderer_state_hash, scene_state_hash):
#         """Remove all environments of which the associated renderer
#         or scene no longer exist, or their states have changed.
#         """
#         hashes_to_drop = []
#         for env_hash, env in self.environments.items():
#             if env.check_inactive(
#                 renderer, scene, renderer_state_hash, scene_state_hash
#             ):
#                 hashes_to_drop.append(env_hash)
#         for env_hash in hashes_to_drop:
#             self.environments.pop(env_hash)


# environment_manager = GlobalEnvironmentManager()
# get_environment = environment_manager.get_environment


# def get_hash_and_state(renderer, scene):
#     state = {}

#     # For renderer

#     state["blender"] = renderer._blender
#     renderer_hash = renderer.blend_mode

#     # For scene

#     point_lights = []
#     directional_lights = []
#     spot_lights = []
#     ambient_color = [0, 0, 0]

#     def visit(ob):
#         if isinstance(ob, PointLight):
#             point_lights.append(ob)
#         elif isinstance(ob, DirectionalLight):
#             directional_lights.append(ob)
#         elif isinstance(ob, SpotLight):
#             spot_lights.append(ob)
#         elif isinstance(ob, AmbientLight):
#             r, g, b = ob.color.to_physical()
#             ambient_color[0] += r * ob.intensity
#             ambient_color[1] += g * ob.intensity
#             ambient_color[2] += b * ob.intensity

#     scene.traverse(visit, True)

#     state["lights"] = {
#         "point_lights": point_lights,
#         "directional_lights": directional_lights,
#         "spot_lights": spot_lights,
#         "ambient": ambient_color,
#     }

#     scene_hash = (
#         len(point_lights),
#         len(directional_lights),
#         len(spot_lights),
#     )

#     return renderer_hash, scene_hash, state
