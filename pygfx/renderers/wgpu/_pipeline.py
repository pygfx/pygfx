"""
This module implements the PipelineContainer (for compute and render pipelines).
This object is responsible for creating the native wgpu objects and doing the
actual dispatching / drawing.
"""

import wgpu

from ...resources import Buffer
from ...utils import logger
from ._shader import BaseShader
from ._utils import to_texture_format, GfxSampler, GfxTextureView
from ._utils import GpuCache, hash_from_value
from ._update import ensure_wgpu_object, ALTTEXFORMAT
from ._shared import get_shared
from . import registry


visibility_render = wgpu.ShaderStage.VERTEX | wgpu.ShaderStage.FRAGMENT
visibility_all = (
    wgpu.ShaderStage.VERTEX | wgpu.ShaderStage.FRAGMENT | wgpu.ShaderStage.COMPUTE
)


# These caches enables sharing gpu resources for similar objects. It
# makes creating such objects faster (i.e. faster startup). It also
# saves gpu resources. It does not necessarily make the visualization
# faster.
LAYOUT_CACHE = GpuCache("layouts")
BINDING_CACHE = GpuCache("bindings")
SHADER_CACHE = GpuCache("shader_modules")
PIPELINE_CACHE = GpuCache("pipelines")


def get_cached_bind_group_layout(device, *args):
    key = "bind_group_layout", hash_from_value(args)
    result = LAYOUT_CACHE.get(key)
    if result is None:
        (entries,) = args
        result = device.create_bind_group_layout(entries=entries)

        LAYOUT_CACHE.set(key, result)

    return result


def get_cached_pipeline_layout(device, *args):
    key = "pipeline_layout", hash_from_value(args)
    result = LAYOUT_CACHE.get(key)
    if result is None:
        (bind_group_layouts,) = args
        result = device.create_pipeline_layout(bind_group_layouts=bind_group_layouts)

        LAYOUT_CACHE.set(key, result)

    return result


def get_cached_bind_group(device, *args):
    key = "bind_group", hash_from_value(args)
    result = BINDING_CACHE.get(key)
    if result is None:
        layout, entries = args
        result = device.create_bind_group(layout=layout, entries=entries)

        BINDING_CACHE.set(key, result)

    return result


def get_cached_shader_module(device, shader, shader_kwargs):
    # Using a key that *defines* the wgsl - rather than the wgsl itself
    # - avoids the templating to be applied on a cache hit, which safes
    # considerable time!
    key = "shader", shader.hash, hash_from_value(shader_kwargs)

    result = SHADER_CACHE.get(key)
    if result is None:
        wgsl = shader.generate_wgsl(**shader_kwargs)
        result = device.create_shader_module(code=wgsl)

        SHADER_CACHE.set(key, result)

    return result


def get_cached_compute_pipeline(device, *args):
    key = "compute_pipeline", hash_from_value(args)
    result = PIPELINE_CACHE.get(key)
    if result is None:
        pipeline_layout, shader_module = args

        result = device.create_compute_pipeline(
            layout=pipeline_layout,
            compute={"module": shader_module, "entry_point": "main"},
        )

        PIPELINE_CACHE.set(key, result)

    return result


def get_cached_render_pipeline(device, *args):
    key = "render_pipeline", hash_from_value(args)
    result = PIPELINE_CACHE.get(key)
    if result is None:
        (
            pipeline_layout,
            shader_module,
            primitive_topology,
            strip_index_format,
            cull_mode,
            depth_descriptor,
            color_descriptors,
        ) = args

        result = device.create_render_pipeline(
            layout=pipeline_layout,
            vertex={
                "module": shader_module,
                "entry_point": "vs_main",
                "buffers": [],
            },
            primitive={
                "topology": primitive_topology,
                "strip_index_format": strip_index_format,
                "front_face": wgpu.FrontFace.ccw,
                "cull_mode": cull_mode,
            },
            depth_stencil=depth_descriptor,
            multisample={
                "count": 1,
                "mask": 0xFFFFFFFF,
                "alpha_to_coverage_enabled": False,
            },
            fragment={
                "module": shader_module,
                "entry_point": "fs_main",
                "targets": color_descriptors,
            },
        )

        PIPELINE_CACHE.set(key, result)

    return result


def get_pipeline_container_group(wobject, environment):
    """Update the GPU objects associated with the given wobject. Returns
    quickly if no changes are needed. Only this function is used by the
    renderer.
    """

    # Get pipeline_container
    try:
        pipeline_container_group = wobject._wgpu_pipeline_container_group
    except AttributeError:
        pipeline_container_group = PipelineContainerGroup()
        wobject._wgpu_pipeline_container_group = pipeline_container_group
        changed_labels = {"create"}
    else:
        # Get whether the object has changes
        changed_labels = wobject.tracker.pop_changed()

    # Update if necessary - this part is defined to be fast if there are no changes
    pipeline_container_group.update(wobject, environment, changed_labels)

    # Return the pipeline container group
    return pipeline_container_group


class Binding:
    """Simple object to hold together some information about a binding, for internal use.

    Parameters:
        name: the name in wgsl
        type: "buffer/subtype", "sampler/subtype", "texture/subtype", "storage_texture/subtype".
            The subtype depends on the type:
            BufferBindingType, SamplerBindingType, TextureSampleType, or StorageTextureAccess.
        resource: the Buffer, GfxTextureView, or GfxSampler object.
        visibility: wgpu.ShaderStage flag
        structname: the custom wgsl struct name, if any. otherwise will auto-generate.

    Some tips on terminology:

    * A "binding" is a generic term for an object that defines how a
        resource (buffer or texture) is bound to a shader. In this subpackage it
        likely means a Binding object like this.
    * This binding can be represented with a binding_descriptor and
        binding_layout_descriptor. These are dicts to be passed to wgpu.
    * A list of these binding_layout_descriptor's can be passed to create_bind_group_layout.
    * A list of these binding_layout's can be passed to create_bind_group.
    * Multiple bind_group_layout's can be combined into a pipeline_layout.

    """

    def __init__(
        self, name, type, resource, visibility=visibility_render, structname=None
    ):
        if isinstance(visibility, str):
            visibility = getattr(wgpu.ShaderStage, visibility)
        self.name = name
        self.type = type
        self.resource = resource
        self.visibility = visibility

        self.structname = structname

    def _require_usage_flags(self, resource, usage_flags):
        if resource._wgpu_object is None:
            resource._wgpu_usage |= usage_flags
        elif not (resource._wgpu_usage & usage_flags):
            logger.warning(
                "{resource} requires usage {usage_flags}, but has already been created."
            )

    def get_bind_group_descriptors(self, slot):
        """Get the descriptors (dicts) for creating a binding_descriptor
        and binding_layout_descriptor. A list of these descriptors are
        combined into a bind_group and bind_group_layout.
        """
        resource = self.resource
        subtype = self.type.partition("/")[2]

        if self.type.startswith("buffer/"):
            assert isinstance(resource, Buffer)
            usage_flags = 0
            if "uniform" in self.type:
                usage_flags |= wgpu.BufferUsage.UNIFORM
            elif "storage" in self.type:
                usage_flags |= wgpu.BufferUsage.STORAGE
                if "indices" in self.name:
                    usage_flags |= wgpu.BufferUsage.INDEX
                else:
                    usage_flags |= wgpu.BufferUsage.VERTEX
            self._require_usage_flags(resource, usage_flags)
            binding = {
                "binding": slot,
                "resource": {
                    "buffer": ensure_wgpu_object(resource),
                    "offset": 0,
                    "size": resource.nbytes,
                },
            }
            # Note: we set min_binding_size, rather than relying on the default (None),
            # otherwise the layout for different buffers look equal, and are thus
            # re-used (using caching), but they won't be compatible.
            binding_layout = {
                "binding": slot,
                "visibility": self.visibility,
                "buffer": {
                    "type": getattr(wgpu.BufferBindingType, subtype),
                    "has_dynamic_offset": False,
                    "min_binding_size": resource.itemsize,
                },
            }
        elif self.type.startswith("sampler/"):
            assert isinstance(resource, GfxSampler)
            binding = {
                "binding": slot,
                "resource": ensure_wgpu_object(resource),
            }
            binding_layout = {
                "binding": slot,
                "visibility": self.visibility,
                "sampler": {
                    "type": getattr(wgpu.SamplerBindingType, subtype),
                },
            }
        elif self.type.startswith("texture/"):
            assert isinstance(resource, GfxTextureView)
            self._require_usage_flags(
                resource.texture, wgpu.TextureUsage.TEXTURE_BINDING
            )
            binding = {
                "binding": slot,
                "resource": ensure_wgpu_object(resource),
            }
            dim = resource.view_dim
            dim = getattr(wgpu.TextureViewDimension, dim, dim)
            sample_type = getattr(wgpu.TextureSampleType, subtype, subtype)
            # Derive sample type from texture
            if sample_type == "auto":
                fmt = to_texture_format(resource.format)
                fmt = ALTTEXFORMAT.get(fmt, [fmt])[0]
                if "float" in fmt or "norm" in fmt:
                    sample_type = wgpu.TextureSampleType.float
                    # For float32 wgpu does not allow the sampler to be filterable,
                    # except when the native-only feature
                    # TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES is set,
                    # which wgpu-py does by default.
                    # if "32float" in fmt:
                    #     sample_type = wgpu.TextureSampleType.unfilterable_float
                elif "uint" in fmt:
                    sample_type = wgpu.TextureSampleType.uint
                elif "sint" in fmt:
                    sample_type = wgpu.TextureSampleType.sint
                elif "depth" in fmt:
                    sample_type = wgpu.TextureSampleType.depth
                else:
                    raise ValueError("Could not determine texture sample type.")
            binding_layout = {
                "binding": slot,
                "visibility": self.visibility,
                "texture": {
                    "sample_type": sample_type,
                    "view_dimension": dim,
                    "multisampled": False,
                },
            }
        elif self.type.startswith("storage_texture/"):
            assert isinstance(resource, GfxTextureView)
            self._require_usage_flags(
                resource.texture, wgpu.TextureUsage.STORAGE_BINDING
            )
            binding = {
                "binding": slot,
                "resource": ensure_wgpu_object(resource),
            }
            dim = resource.view_dim
            dim = getattr(wgpu.TextureViewDimension, dim, dim)
            fmt = to_texture_format(resource.format)
            fmt = ALTTEXFORMAT.get(fmt, [fmt])[0]
            binding_layout = {
                "binding": slot,
                "visibility": self.visibility,
                "storage_texture": {
                    "access": getattr(wgpu.StorageTextureAccess, subtype),
                    "format": fmt,
                    "view_dimension": dim,
                },
            }
        elif self.type.startswith("shadow_texture/"):
            # a shadow_texture's resource is wgpu.GPUTextureView
            assert isinstance(resource, wgpu.GPUTextureView)
            binding = {"binding": slot, "resource": resource}

            binding_layout = {
                "binding": slot,
                "visibility": wgpu.ShaderStage.FRAGMENT,
                "texture": {
                    "sample_type": wgpu.TextureSampleType.depth,
                    "view_dimension": subtype,
                    "multisampled": False,
                },
            }
        elif self.type.startswith("shadow_sampler/"):
            # a shadow_sampler's resource is wgpu.GPUSampler
            assert isinstance(resource, wgpu.GPUSampler)
            binding = {"binding": slot, "resource": resource}

            binding_layout = {
                "binding": slot,
                "visibility": wgpu.ShaderStage.FRAGMENT,
                "sampler": {"type": wgpu.SamplerBindingType.comparison},
            }
        else:
            raise RuntimeError(f"Unexpected binding type: '{self.type}'")

        return binding, binding_layout


class PipelineContainerGroup:
    """This is a thin wrapper for a list of compute pipeline containers,
    and render pipeline containers. The purpose of this object is to
    obtain the appropriate shader objects and store them.
    """

    def __init__(self):
        self.compute_containers = None
        self.render_containers = None

    def update(self, wobject, environment, changed):
        """Update the pipeline containers that are wrapped. Creates (and re-creates)
        the containers if necessary.
        """

        if "create" in changed:
            self.compute_containers = []
            self.render_containers = []

            # Get render function for this world object,
            # and use it to get a high-level description of pipelines.
            renderfunc = registry.get_render_function(wobject)
            if renderfunc is None:
                raise ValueError(
                    f"Could not get a render function for {wobject.__class__.__name__} "
                    f"with {wobject.material.__class__.__name__}"
                )

            # Call render function
            with wobject.tracker.track_usage("create"):
                shaders = renderfunc(wobject)
                if isinstance(shaders, BaseShader):
                    shaders = [shaders]

            # Divide result over two bins, one for compute, and one for render
            for shader in shaders:
                assert isinstance(shader, BaseShader)
                if shader.type == "compute":
                    self.compute_containers.append(ComputePipelineContainer(shader))
                elif shader.type == "render":
                    self.render_containers.append(RenderPipelineContainer(shader))
                else:
                    raise ValueError(f"Shader type {shader.type} is unknown.")

        for container in self.compute_containers:
            container.update(wobject, environment, changed)
        for container in self.render_containers:
            container.update(wobject, environment, changed)


class PipelineContainer:
    """The pipeline container stores the wgpu pipeline object as well as intermediate
    steps needed to create it. When an dependency of a certain step changes (which we track)
    then only the steps below it need to be re-run.
    """

    def __init__(self, shader):
        self.shader = shader
        self.shared = get_shared()  # the globally Shared object
        self.device = self.shared.device

        # The info that the shader generates
        self.shader_hash = b""
        self.bindings_dicts = None  # dict of dict of bindings
        self.pipeline_info = None
        self.render_info = None

        # The wgpu objects that we generate
        # These map blend_mode to a list of objects (one for each pass).
        # For compute shaders the blend_mode is always "" and there is
        # one object in each.
        self.wgpu_shaders = {}
        self.wgpu_pipelines = {}

        # The bindings map group_index -> group
        self.bind_group_layout_descriptors = []
        self.wgpu_bind_group_layouts = []
        self.wgpu_bind_groups = []

        # A flag to indicate that an error occurred and we cannot dispatch
        self.broken = False

    def _check_bindings(self):
        assert isinstance(self.bindings_dicts, dict), "bindings_dicts must be a dict"
        for key1, bindings_dict in self.bindings_dicts.items():
            assert isinstance(key1, int), f"bindings slot must be int, not {key1}"
            assert isinstance(bindings_dict, dict), "bindings_dict must be a dict"
            for key2, b in bindings_dict.items():
                assert isinstance(key2, int), f"bind group slot must be int, not {key2}"
                assert isinstance(b, Binding), f"binding must be Binding, not {b}"

    def remove_env_hash(self, env_hash):
        """Called from the environment when it becomes inactive.
        This allows this object to remove all references to wgpu objects
        that won't be used, reclaiming memory on both CPU and GPU.
        """
        self.wgpu_shaders.pop(env_hash, None)
        self.wgpu_pipelines.pop(env_hash, None)

    def update(self, wobject, environment, changed):
        """Make sure that the pipeline is up-to-date."""

        if isinstance(self, RenderPipelineContainer):
            env_hash = environment.hash
        else:
            env_hash = ""

        # Ensure that the information provided by the shader is up-to-date
        if changed:
            try:
                self.update_shader_data(wobject, changed)
            except Exception as err:
                self.broken = 1
                raise err
            else:
                self.broken = 0

        # Ensure that the (environment specific) wgpu objects are up-to-date
        if not self.broken:
            try:
                self.update_wgpu_data(wobject, environment, env_hash, changed)
            except Exception as err:
                self.broken = 2
                raise err
            else:
                self.broken = False

        if changed:
            logger.info(f"{wobject} shader update: {', '.join(sorted(changed))}.")

    def update_shader_data(self, wobject, changed):
        """Update the info that applies to all passes and environments."""

        if "create" in changed:
            changed.update(("bindings", "pipeline_info", "render_info"))

        if "bindings" in changed:
            self.shader.unlock_hash()
            with wobject.tracker.track_usage("!bindings"):
                self.bindings_dicts = self.shader.get_bindings(wobject, self.shared)
            self.shader.lock_hash()
            self._check_bindings()
            self.update_shader_hash()
            self.update_bind_groups()

        if "pipeline_info" in changed:
            with wobject.tracker.track_usage("pipeline_info"):
                self.pipeline_info = self.shader.get_pipeline_info(wobject, self.shared)
            self._check_pipeline_info()
            changed.add("render_info")
            self.wgpu_pipelines = {}

        if "render_info" in changed:
            with wobject.tracker.track_usage("render_info"):
                self.render_info = self.shader.get_render_info(wobject, self.shared)
            self._check_render_info()

    def update_wgpu_data(self, wobject, environment, env_hash, changed):
        """Update the actual wgpu objects."""

        if self.wgpu_shaders.get(env_hash, None) is None:
            environment.register_pipeline_container(self)  # allows us to clean up
            changed.add("compile_shader")
            self.wgpu_shaders[env_hash] = self._compile_shaders(environment)
            self.wgpu_pipelines = {}  # Invalidate pipelines so new shaders get used

        if self.wgpu_pipelines.get(env_hash, None) is None:
            changed.add("compose_pipeline")
            shader_modules = self.wgpu_shaders[env_hash]
            self.wgpu_pipelines[env_hash] = self._compose_pipelines(
                environment, shader_modules
            )

    def update_shader_hash(self):
        """Update the shader hash, invalidating the wgpu shaders if it changed."""
        sh = self.shader.hash
        if sh != self.shader_hash:
            self.shader_hash = sh
            self.wgpu_shaders = {}

    def update_bind_groups(self):
        """
        - Calculate new bind_group_layout_descriptors (simple dicts).
        - When this has changed from our last version, we also update
          wgpu_bind_group_layouts and reset self.wgpu_pipelines
        - Calculate new wgpu_bind_groups.
        """

        # Check the bindings structure
        for i, bindings_dict in self.bindings_dicts.items():
            assert isinstance(i, int)
            assert isinstance(bindings_dict, dict)
            for j, b in bindings_dict.items():
                assert isinstance(j, int)
                assert isinstance(b, Binding)

        # Create two new dicts that correspond closely to the bindings.
        # Except this turns the dicts into lists.
        # These are the descriptors to create the wgpu bind groups and
        # bind group layouts.
        bg_descriptors = []
        bg_layout_descriptors = []
        for group_id in sorted(self.bindings_dicts.keys()):
            bindings_dict = self.bindings_dicts[group_id]
            while len(bg_descriptors) <= group_id:
                bg_descriptors.append([])
                bg_layout_descriptors.append([])
            bg_descriptor = bg_descriptors[group_id]
            bg_layout_descriptor = bg_layout_descriptors[group_id]
            for slot in sorted(bindings_dict.keys()):
                binding = bindings_dict[slot]
                binding_des, binding_layout_des = binding.get_bind_group_descriptors(
                    slot
                )
                bg_descriptor.append(binding_des)
                bg_layout_descriptor.append(binding_layout_des)

        # Clean up any trailing empty descriptors
        while bg_descriptors and not bg_descriptors[-1]:
            bg_descriptors.pop(-1)
            bg_layout_descriptors.pop(-1)

        # If the layout has changed, we need a new pipeline
        if self.bind_group_layout_descriptors != bg_layout_descriptors:
            self.bind_group_layout_descriptors = bg_layout_descriptors
            # Invalidate the pipeline
            self.wgpu_pipelines = {}
            # Create wgpu objects for the bind group layouts
            self.wgpu_bind_group_layouts = []
            for bg_layout_descriptor in bg_layout_descriptors:
                bind_group_layout = get_cached_bind_group_layout(
                    self.device, bg_layout_descriptor
                )
                self.wgpu_bind_group_layouts.append(bind_group_layout)

        # Always create wgpu objects for the bind groups. Note that this dict
        # includes the buffer texture objects.
        self.wgpu_bind_groups = []
        for bg_descriptor, layout in zip(bg_descriptors, self.wgpu_bind_group_layouts):
            bind_group = get_cached_bind_group(self.device, layout, bg_descriptor)
            self.wgpu_bind_groups.append(bind_group)


class ComputePipelineContainer(PipelineContainer):
    """Container for compute pipelines."""

    # These checks are here to validate the output of the shader
    # methods, not for user code. So its ok to use assertions here.

    def _check_pipeline_info(self):
        pipeline_info = self.pipeline_info
        assert isinstance(pipeline_info, dict)

        assert set(pipeline_info.keys()) == set(), f"{pipeline_info.keys()}"

    def _check_render_info(self):
        render_info = self.render_info
        assert isinstance(render_info, dict)

        assert set(render_info.keys()) == {"indices"}, f"{render_info.keys()}"

        indices = render_info["indices"]
        assert isinstance(indices, (tuple, list))
        assert all(isinstance(i, int) for i in indices)
        assert len(indices) == 3

    def _compile_shaders(self, env):
        """Compile the templateds wgsl shader to a wgpu shader module."""
        shader_module = get_cached_shader_module(self.device, self.shader, {})
        return {0: shader_module}

    def _compose_pipelines(self, env, shader_modules):
        """Create the wgpu pipeline object from the shader and bind group layouts."""

        # Create pipeline layout object from list of layouts
        pipeline_layout = get_cached_pipeline_layout(
            self.device, self.wgpu_bind_group_layouts
        )

        # Create pipeline object
        pipeline = get_cached_compute_pipeline(
            self.device, pipeline_layout, shader_modules[0]
        )
        return {0: pipeline}

    def dispatch(self, compute_pass):
        """Dispatch the pipeline, doing the actual compute job."""
        if self.broken:
            return

        # Collect what's needed
        pipeline = self.wgpu_pipeline
        indices = self.render_info["indices"]
        bind_groups = self.wgpu_bind_groups

        # Set pipeline and bindings
        compute_pass.set_pipeline(pipeline)
        for bind_group_id, bind_group in enumerate(bind_groups):
            compute_pass.set_bind_group(bind_group_id, bind_group, [], 0, 999999)

        # Compute!
        compute_pass.dispatch_workgroups(*indices)


class RenderPipelineContainer(PipelineContainer):
    """Container for render pipelines."""

    def __init__(self, shader):
        super().__init__(shader)
        self.strip_index_format = 0

    def _check_pipeline_info(self):
        pipeline_info = self.pipeline_info
        assert isinstance(pipeline_info, dict)

        expected = {"cull_mode", "primitive_topology"}
        assert set(pipeline_info.keys()) == expected, f"{pipeline_info.keys()}"
        self.update_strip_index_format()

    def _check_render_info(self):
        render_info = self.render_info
        assert isinstance(render_info, dict)

        expected = {"indices", "render_mask"}
        assert set(render_info.keys()) == expected, f"{render_info.keys()}"

        indices = render_info["indices"]
        assert isinstance(indices, (tuple, list))
        assert all(isinstance(i, int) for i in indices)
        assert len(indices) in (2, 4, 5)
        if len(indices) == 2:
            render_info["indices"] = indices[0], indices[1], 0, 0

        render_mask = render_info["render_mask"]
        assert isinstance(render_mask, int) and render_mask in (1, 2, 3)

    def update_strip_index_format(self):
        if not self.bindings_dicts or not self.pipeline_info:
            return
        # Set strip_index_format
        index_format = wgpu.IndexFormat.uint32
        strip_index_format = 0
        if "strip" in self.pipeline_info["primitive_topology"]:
            strip_index_format = index_format
        # Trigger a pipeline rebuild?
        if self.strip_index_format != strip_index_format:
            self.strip_index_format = strip_index_format
            self.wgpu_pipelines = {}

    def _compile_shaders(self, env):
        """Compile the templated shader to a list of wgpu shader modules
        (one for each pass of the blender).
        """
        blender = env.blender
        render_mask = self.render_info["render_mask"]

        shader_modules = {}
        for pass_index in range(blender.get_pass_count()):
            # No need to compile pass for transparent fragments if the object has none
            if not render_mask & blender.passes[pass_index].render_mask:
                continue

            color_descriptors = blender.get_color_descriptors(pass_index)
            if not color_descriptors:
                continue

            env_bind_group_index = len(self.wgpu_bind_groups)

            blender_kwargs = blender.get_shader_kwargs(pass_index)
            env_kwargs = env.get_shader_kwargs(env_bind_group_index)
            shader_kwargs = blender_kwargs.copy()
            shader_kwargs.update(env_kwargs)

            shader_module = get_cached_shader_module(
                self.device, self.shader, shader_kwargs
            )
            shader_modules[pass_index] = shader_module

        return shader_modules

    def _compose_pipelines(self, env, shader_modules):
        """Create a list of wgpu pipeline objects from the shader, bind group
        layouts and other pipeline info (one for each pass of the blender).
        """

        strip_index_format = self.strip_index_format
        primitive_topology = self.pipeline_info["primitive_topology"]
        cull_mode = self.pipeline_info["cull_mode"]
        render_mask = self.render_info["render_mask"]

        # Create pipeline layout object from list of layouts
        env_bind_group_layout, _ = env.wgpu_bind_group
        bind_group_layouts = [*self.wgpu_bind_group_layouts, env_bind_group_layout]

        pipeline_layout = get_cached_pipeline_layout(self.device, bind_group_layouts)

        # Instantiate the pipeline objects.
        # Note: The pipeline relies on the color and depth descriptors, which
        # include the texture format and a few other static things.
        # This step should *not* rerun when e.g. the canvas resizes.
        pipelines = {}
        blender = env.blender
        for pass_index in range(blender.get_pass_count()):
            # No need to compose pass for transparent fragments if the object has none
            if not render_mask & blender.passes[pass_index].render_mask:
                continue

            color_descriptors = blender.get_color_descriptors(pass_index)
            depth_descriptor = blender.get_depth_descriptor(pass_index)
            if not color_descriptors:
                continue

            shader_module = shader_modules[pass_index]

            pipeline = get_cached_render_pipeline(
                self.device,
                pipeline_layout,
                shader_module,
                primitive_topology,
                strip_index_format,
                cull_mode,
                depth_descriptor,
                color_descriptors,
            )
            pipelines[pass_index] = pipeline

        return pipelines

    def draw(self, render_pass, environment, pass_index, render_mask):
        """Draw the pipeline, doing the actual rendering job."""
        if self.broken:
            return

        if not (render_mask & self.render_info["render_mask"]):
            return
        env_hash = environment.hash

        # Collect what's needed
        pipeline = self.wgpu_pipelines[env_hash][pass_index]
        indices = self.render_info["indices"]
        bind_groups = self.wgpu_bind_groups

        # Set pipeline and bindings
        render_pass.set_pipeline(pipeline)

        for bind_group_id, bind_group in enumerate(bind_groups):
            render_pass.set_bind_group(bind_group_id, bind_group, [], 0, 99)

        env_bind_group_id = len(bind_groups)
        _, env_bind_group = environment.wgpu_bind_group
        render_pass.set_bind_group(env_bind_group_id, env_bind_group, [], 0, 99)

        # Draw!
        # draw(count_vertex, count_instance, first_vertex, first_instance)
        render_pass.draw(*indices)
