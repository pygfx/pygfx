"""
"""

import wgpu

from ...resources import Buffer, TextureView

from ._utils import to_vertex_format, to_texture_format
from ._update import update_resource, ALTTEXFORMAT
from ._shadercomposer import Binding
from . import registry


def ensure_pipeline(renderer, wobject):
    """Update the GPU objects associated with the given wobject. Returns
    quickly if no changes are needed. Only this function is used by the
    renderer.
    """

    levels = wobject.pop_changed()

    # todo: update resources

    if not levels:
        return

    try:
        pipeline_container = wobject._wgpu_pipeline_container
    except AttributeError:
        pipeline_container = PipelineContainer()
        wobject._wgpu_pipeline_container = pipeline_container

    pipeline_container.update(levels)

    shared = renderer._shared
    blender = renderer._blender
    pipelines = renderer._wobject_pipelines
    device = shared.device

    changed = pipeline_container.update(wobject)

    #

    # Get wobject_pipeline dict associated with this renderer and wobject
    wobject_pipeline = pipelines.get(wobject, {})

    # This becomes not-None if we need to update the pipeline dict
    new_pipeline_infos = None

    # Do we need to recreate the pipeline_objects?
    if wobject.rev != wobject_pipeline.get("ref", ()):
        # Create fresh wobject_pipeline
        wobject_pipeline = {"ref": wobject.rev, "renderable": False, "resources": []}
        pipelines[wobject] = wobject_pipeline
        # Create pipeline_info and collect resources
        new_pipeline_infos = create_pipeline_infos(shared, blender, wobject)
        if new_pipeline_infos:
            wobject_pipeline["renderable"] = True
            wobject_pipeline["resources"] = collect_pipeline_resources(
                shared, wobject, new_pipeline_infos
            )

    # Early exit?
    if not wobject_pipeline["renderable"]:
        return None, False

    # Check if we need to update any resources. The number of resources
    # should typically be small. We could implement a hook in the
    # resource's rev setter so we only have to check one flag ... or
    # collect all resources on a rendered .. but let's not optimize
    # prematurely.
    for kind, resource in wobject_pipeline["resources"]:
        our_version = getattr(resource, "_wgpu_" + kind, (-1, None))[0]
        if resource.rev > our_version:
            update_resource(device, resource, kind)

    # Create gpu objects?
    has_changed = bool(new_pipeline_infos)
    if has_changed:
        new_wobject_pipeline = create_pipeline_objects(
            shared, blender, wobject, new_pipeline_infos
        )
        wobject_pipeline.update(new_wobject_pipeline)

    # Set in what passes this object must render
    if has_changed:
        m = {"auto": 0, "opaque": 1, "transparent": 2, "all": 3}
        render_mask = m[wobject.render_mask]
        if not render_mask:
            render_mask = new_pipeline_infos[0].get("suggested_render_mask", 3)
        wobject_pipeline["render_mask"] = render_mask

    return wobject_pipeline, has_changed


class RenderInfo:
    """The type of object passed to each wgpu render function. Contains
    all the info that it might need:
    * wobject
    * stdinfo buffer
    * blender
    """

    def __init__(self, *, wobject, stdinfo_uniform, blender):
        self.wobject = wobject
        self.stdinfo_uniform = stdinfo_uniform
        # todo: blender not needed self.blender = blender


class Binding:
    """Simple object to hold together some information about a binding, for internal use.

    * name: the name in wgsl
    * type: "buffer/subtype", "sampler/subtype", "texture/subtype", "storage_texture/subtype".
      The subtype depends on the type:
      BufferBindingType, SamplerBindingType, TextureSampleType, or StorageTextureAccess.
    * resource: Buffer, Texture or TextureView.
    * visibility: wgpu.ShaderStage flag
    * kwargs: could add more specifics in the future.
    """

    def __init__(self, name, type, resource, visibility=visibility_render):
        if isinstance(visibility, str):
            visibility = getattr(wgpu.ShaderStage, visibility)
        self.name = name
        self.type = type
        self.resource = resource
        self.visibility = visibility

    def get_bind_group_descriptors(self, slot):
        binding = self
        resource = binding.resource
        subtype = binding.type.partition("/")[2]

        if binding.type.startswith("buffer/"):
            assert isinstance(resource, Buffer)
            binding = {
                "binding": slot,
                "resource": {
                    "buffer": resource._wgpu_buffer[1],
                    "offset": 0,
                    "size": resource.nbytes,
                },
            }
            binding_layout = {
                "binding": slot,
                "visibility": binding.visibility,
                "buffer": {
                    "type": getattr(wgpu.BufferBindingType, subtype),
                    "has_dynamic_offset": False,
                    "min_binding_size": 0,
                },
            }
        elif binding.type.startswith("sampler/"):
            assert isinstance(resource, TextureView)
            binding = {"binding": slot, "resource": resource._wgpu_sampler[1]}
            binding_layout = {
                "binding": slot,
                "visibility": binding.visibility,
                "sampler": {
                    "type": getattr(wgpu.SamplerBindingType, subtype),
                },
            }
        elif binding.type.startswith("texture/"):
            assert isinstance(resource, TextureView)
            binding = {"binding": slot, "resource": resource._wgpu_texture_view[1]}
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
                "visibility": binding.visibility,
                "texture": {
                    "sample_type": sample_type,
                    "view_dimension": dim,
                    "multisampled": False,
                },
            }
        elif binding.type.startswith("storage_texture/"):
            assert isinstance(resource, TextureView)
            binding = {"binding": slot, "resource": resource._wgpu_texture_view[1]}
            dim = resource.view_dim
            dim = getattr(wgpu.TextureViewDimension, dim, dim)
            fmt = to_texture_format(resource.format)
            fmt = ALTTEXFORMAT.get(fmt, [fmt])[0]
            binding_layout = {
                "binding": slot,
                "visibility": binding.visibility,
                "storage_texture": {
                    "access": getattr(wgpu.StorageTextureAccess, subtype),
                    "format": fmt,
                    "view_dimension": dim,
                },
            }

        return binding, binding_layout


class PipelineContainerGroup:

    # Different levels of updates:
    #
    # 1. Compose pipeline info (the render function).
    # 2. Compile shader (includes final templating).
    # 3. Build pipeline (the wgpu object).
    # 4. Record render bundles (we don't use this optimization yet).
    #
    # How changes affect these steps:
    #
    # * Updating uniforms and parts of buffers and textures:
    #   no updates needed.
    # * Resizing buffers and textures (new wgpu object, same bindgroup layout):
    #   rerecord the render bundle.
    # * Replacing a buffer or texture with a different layout:
    #   rebuild pipeline.
    # * Some material properties affect the rendering:
    #   recompose.
    # * Attaching a different material:
    #   recompose.
    # * Blend mode changes:
    #   recompile shader.
    # * New composition of a scene: probably a new render bundle.

    def __init__(self):
        self.compute_pipelines = None  # List of container objects, one for each info
        self.render_pipelines = None

    # todo: keep track of buffer/texture attributes changes
    # todo: when a buffer is replaced, check whether the layout still matches, and then decide what to do
    # todo: atlas should be *on* something
    # todo: lights could be a special uniform, on the scene. with dynamic size.
    # todo: clean up per-blend_mode stuff, by keeping track of what blend modes exist
    # todo: the number of indices is defined in the pipeline info

    def update(self, renderer, wobject, levels):

        if self.render_pipelines is None or "reset" in levels:
            self.compute_pipelines = []
            self.render_pipelines = []

            # Get render function for this world object,
            # and use it to get a high-level description of pipelines.
            renderfunc = registry.get_render_function(wobject)
            if renderfunc is None:
                raise ValueError(
                    f"Could not get a render function for {wobject.__class__.__name__} "
                    f"with {wobject.material.__class__.__name__}"
                )

            # Prepare info for the render function
            render_info = RenderInfo(
                wobject=wobject, stdinfo_uniform=shared.stdinfo_buffer, blender=blender
            )

            # Call render function
            builders = renderfunc(render_info)
            for builder in builders:
                if isinstance(builder, WobjectRenderBuilder):
                    self.render_pipelines.append(RenderPipelineContainer(builder))
                elif isinstance(builder, WobjectComputeBuilder):
                    self.compute_pipelines.append(ComputePipelineContainer(builder))
                else:
                    raise ValueError(
                        "Render function should return an iterable of WobjectComputeBuilder and WobjectRenderBuilder objects."
                    )

        for container in self.compute_pipelines:
            container.update()
        for container in self.render_pipelines:
            container.update()


class WobjectBuilder:
    def __setattr__(self, name, value):
        raise AttributeError("Its not allowed to store stuff on the builder object.")


class WobjectComputeBuilder(WobjectBuilder):
    pass


class WobjectRenderBuilder(WobjectBuilder):
    def get_shader(self):
        # material props here are tracked
        # resources are tracked to, but in this case the access to their metadata
        pass

    def get_resources(self):
        # indexbuffer
        # vertex_buffers
        # list of list of dicts
        pass

    def get_pipeline_info(self):
        # cull_mode
        # primitive_topology
        pass

    def get_render_info(self):
        # render_mask
        # indices
        pass


class PipelineContainer:
    """The pipeline container stores the wgpu pipeline object as well as intermediate
    steps needed to create it. When an dependency of a certain step changes (which we track)
    then only the steps below it need to be re-run.
    """

    def __init__(self, builder):
        self.builder = builder

        # The info that the builder generates
        self.shader = None
        self.resources = None
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

        # A flat list of all buffers, textures, samplers etc. in use
        self.flat_resources = []

    def update(self, shared, blender, levels):
        if isinstance(self, RenderPipelineContainer):
            blend_mode = blender.__class__.__name__
        else:
            blend_mode = ""
        bind_group_layouts = self.bind_group_layouts

        self.update_builder_data(shader, blender, levels)
        self.update_wgpu_data(shared, blender, blend_mde)

    def update_builder_data(self, shader, blender, levels):
        # Update the info that applies to all passes

        if "shader" in levels:
            with wobject.track_usage(self, "shader", False):
                self.shader = self.builder.get_shader()
            self._check_shader()
            levels.update(("resources", "pipeline"))
            self.wgpu_shaders = {}

        if "pipeline" in levels:
            with wobject.track_usage(self, "pipeline", False):
                self.pipeline_info = self.builder.get_pipeline_info()
            self._check_pipeline_info()
            levels.add("render")
            self.wgpu_pipelines = {}

        if "render" in levels:
            with wobject.track_usage(self, "pipeline", False):
                self.render_info = self.builder.get_rendder_info()
            self._check_render_info()

        if "resources" in levels:
            with wobject.track_usage(self, "resources", True):
                self.resources = self.builder.get_resources()
            self._check_resources()
            self.flat_resources = self.collect_flat_resources()

    def update_wgpu_data(self, shared, blender, blend_mode):
        # Update the actual wgpu objects

        if self.wgpu_shaders.get(blend_mode, None) is None:
            self.wgpu_shaders[blend_mode] = self._compile_shaders(shared, blender)

        if self.wgpu_pipelines.get(blend_mode, None) is None:
            self.wgpu_pipelines[blend_mode] = self._compose_pipelines(shared, blender)

    def update_bind_groups(self):

        # Check the bindings structure
        for i, group in resources["bindings"].items():
            assert isinstance(i, int)
            assert isinstance(group, dict)
            for j, b in group.items():
                assert isinstance(j, int)
                assert isinstance(b, Binding)

        # Create two new dicts that correspond closely to the bindings
        # in the resources. Except this turns the dicts into lists.
        # These are the descriptors to create the wgpu bind groups and
        # bind group layouts.
        bg_descriptors = []
        bg_layout_descriptors = []
        for group_id in sorted(self.resources["bindings"].keys()):
            bindings_dict = self.resources["bindings"][group_id]
            while len(bg_descriptors) <= group_id:
                bg_descriptors.append([])
                bg_layout_descriptors.append([])
            bg_descriptor = bg_descriptors[-1]
            bg_layout_descriptor = bg_layout_descriptors[-1]
            for slot in sorted(bindings_dict.keys()):
                binding = bindings_dict[slot]
                binding_des, binding_layout_des = binding.get_bind_group_descriptors(
                    slot
                )
                bg_descriptor.append(binding_des)
                bg_layout_descriptor.append(binding_layout_des)

        # If the layout has changed, we need a new pipeline
        if self.bind_group_layout_descriptors != bg_layout_descriptors:
            self.bind_group_layout_descriptors = bg_layout_descriptors
            self.wgpu_pipelines = {}
            # Create wgpu objects for the bind group layouts
            self.wgpu_bind_group_layouts = []
            for bg_layout_descriptor in bg_layout_descriptors:
                bind_group_layout = device.create_bind_group_layout(
                    entries=binding_layouts
                )
                self.wgpu_bind_group_layouts.append(bind_group_layout)

        # Always create wgpu objects for the bind groups. Note that this dict
        # includes the buffer texture objects.
        self.wgpu_bind_groups = []
        for group_id, bg_descriptor in bg_descriptors.items():
            bind_group = device.create_bind_group(
                layout=bind_group_layout, entries=bg_descriptor
            )
            self.wgpu_bind_groups.append(bind_group)

    def collect_flat_resources(self, shared):
        """Collect a list of all used resources, and also set their usage."""
        resources = self.resources
        pipeline_resources = []  # List, because order matters

        buffer = resources.get("index_buffer", None)
        if buffer is not None:
            buffer._wgpu_usage |= wgpu.BufferUsage.INDEX | wgpu.BufferUsage.STORAGE
            pipeline_resources.append(("buffer", buffer))

        for buffer in resources.get("vertex_buffers", {}).values():
            buffer._wgpu_usage |= wgpu.BufferUsage.VERTEX | wgpu.BufferUsage.STORAGE
            pipeline_resources.append(("buffer", buffer))

        for group_dict in resources.values():
            for binding in group_dict.values():
                resource = binding.resource
                if binding.type.startswith("buffer/"):
                    assert isinstance(resource, Buffer)
                    pipeline_resources.append(("buffer", resource))
                    if "uniform" in binding.type:
                        resource._wgpu_usage |= wgpu.BufferUsage.UNIFORM
                    elif "storage" in binding.type:
                        resource._wgpu_usage |= wgpu.BufferUsage.STORAGE
                        if "indices" in binding.name:
                            resource._wgpu_usage |= wgpu.BufferUsage.INDEX
                        else:
                            resource._wgpu_usage |= wgpu.BufferUsage.VERTEX
                elif binding.type.startswith("sampler/"):
                    assert isinstance(resource, TextureView)
                    pipeline_resources.append(("sampler", resource))
                elif binding.type.startswith("texture/"):
                    assert isinstance(resource, TextureView)
                    resource.texture._wgpu_usage |= wgpu.TextureUsage.TEXTURE_BINDING
                    pipeline_resources.append(("texture", resource.texture))
                    pipeline_resources.append(("texture_view", resource))
                elif binding.type.startswith("storage_texture/"):
                    assert isinstance(resource, TextureView)
                    resource.texture._wgpu_usage |= wgpu.TextureUsage.STORAGE_BINDING
                    pipeline_resources.append(("texture", resource.texture))
                    pipeline_resources.append(("texture_view", resource))
                else:
                    raise RuntimeError(
                        f"Unknown resource binding {binding.name} of type {binding.type}"
                    )

        return pipeline_resources


class ComputePipelineContainer(PipelineContainer):
    """Container for compute pipelines."""

    # These checks are here to validate the output of the builder
    # methods, not for user code. So its ok to use assertions here.

    def _check_shader(self):
        assert hasattr(self.shader, "generate_wgsl")

    def _check_pipeline_info(self):
        pipeline_info = self.pipeline_info
        assert isinstance(pipeline_info, dict)

        assert set(render_info.keys()) == set()

    def _check_render_info(self):
        render_info = self.render_info
        assert isinstance(render_info, dict)

        assert set(render_info.keys()) == {"indices"}

        indices = render_info["indices"]
        assert isinstance(indices, (tuple, list))
        assert all(isinstance(i, int) for i in indices)
        assert len(indices) == 3

    def _check_resources(self):
        resources = self.resources
        assert isinstance(resources, dict)
        assert set(resources.keys()) == {"bindings"}

        # Process bind groups - may invalidate the wgpu pipelines
        self.update_bind_groups()

    def _compile_shaders(self, shared, blender):
        """Compile the templateds wgsl shader to a wgpu shader module."""
        wgsl = self.shader.generate_wgsl()
        shader_module = get_shader_module(shared, wgsl)
        return {0: shader_module}

    def _compose_pipelines(self, shared, blender):
        """Create the wgpu pipeline object from the shader and bind group layouts."""

        # todo: cache the pipeline with the shader (and entrypoint) as a hash

        # Create pipeline layout object from list of layouts
        pipeline_layout = device.create_pipeline_layout(
            bind_group_layouts=self.wgpu_bind_group_layouts
        )

        # Create pipeline object
        cs_module = self.wgpu_shaders[blend_mode][0]
        pipeline = device.create_compute_pipeline(
            layout=pipeline_layout,
            compute={"module": cs_module, "entry_point": "main"},
        )

        return {0: pipeline}

    def dispatch(self, compute_pass):
        """Dispatch the pipeline, doing the actual compute job."""
        # Collect what's needed
        pipeline = self.wgpu_pipeline
        indices = self.render_info["indices"]
        bind_groups = self.wgpu_bind_groups

        # Set pipeline and resources
        compute_pass.set_pipeline(pipeline)
        for bind_group_id, bind_group in enumerate(bind_groups):
            compute_pass.set_bind_group(bind_group_id, bind_group, [], 0, 999999)

        # Compute!
        compute_pass.dispatch_workgroups(*indices)


class RenderPipelineContainer(PipelineContainer):
    """Container for render pipelines."""

    def __init__(self):
        super().__init__()
        self.strip_index_format = 0

    def _check_shader(self):
        assert hasattr(self.shader, "generate_wgsl")

    def _check_pipeline_info(self):
        pipeline_info = self.pipeline_info
        assert isinstance(pipeline_info, dict)

        assert set(render_info.keys()) == {"cull_mode", "primitive_topology"}

    def _check_render_info(self):
        assert set(self.render_info.keys()) == {"indices", "render_mask"}

        indices = self.render_info["indices"]
        assert isinstance(indices, (tuple, list))
        assert all(isinstance(i, int) for i in indices)
        assert len(indices) in (2, 4, 5)
        if len(indices) == 2:
            self.render_info["indices"] = indices[0], indices[1], 0, 0

        render_mask = self.render_info["render_mask"]
        assert isinstance(render_mask, int) and render_mask in (1, 2, 3)

    def _check_resources(self):
        resources = self.resources
        assert isinstance(resources, dict)
        assert set(resources.keys()) == {"index_buffer", "vertex_buffers", "bindings"}

        assert isinstance(resources["index_buffer"], (None.__class__, Buffer))
        assert isinstance(resources["vertex_buffers"], dict)
        assert all(isinstance(slot, int) for slot in resources["vertex_buffers"].keys())
        assert all(isinstance(b, Buffer) for b in resources["vertex_buffers"].values())

        # Process resources - may invalidate the wgpu pipelines
        self.update_index_buffer_format()
        self.update_vertex_buffer_descriptors()
        self.update_bind_groups()

    def update_index_buffer_format(self):

        # Set strip_index_format
        index_format = wgpu.IndexFormat.uint32
        index_buffer = self.resources["index_buffer"]
        if index_buffer is not None:
            index_format = to_vertex_format(index_buffer.format)
            index_format = index_format.split("x")[0].replace("s", "u")
        strip_index_format = 0
        if "strip" in self.pipeline_info["primitive_topology"]:
            strip_index_format = index_format
        # Trigger a pipeline rebuild?
        if self.strip_index_format != strip_index_format:
            self.strip_index_format = strip_index_format
            self.wgpu_pipelines = {}

    def update_vertex_buffer_descriptors(self):
        # todo: we can probably expose multiple attributes per buffer using a BufferView
        vertex_buffer_descriptors = []
        for slot, buffer in self.resources["vertex_buffers"].items():
            vbo_des = {
                "array_stride": buffer.nbytes // buffer.nitems,
                "step_mode": wgpu.VertexStepMode.vertex,  # vertex or instance
                "attributes": [
                    {
                        "format": to_vertex_format(buffer.format),
                        "offset": 0,
                        "shader_location": slot,
                    }
                ],
            }
            vertex_buffer_descriptors.append(vbo_des)
        # Trigger a pipeline rebuild?
        if vertex_buffer_descriptors != self.vertex_buffer_descriptors:
            self.vertex_buffer_descriptors = vertex_buffer_descriptors
            self.wgpu_pipelines = {}

    def _compile_shaders(self, shared, blender):
        """Compile the templated shader to a list of wgpu shader modules
        (one for each pass of the blender).
        """
        shader_modules = {}
        for pass_index in range(blender.get_pass_count()):
            color_descriptors = blender.get_color_descriptors(pass_index)
            if not color_descriptors:
                continue

            wgsl = self.shader.generate_wgsl(**blender.get_shader_kwargs(pass_index))
            shader_modules[pass_index] = get_shader_module(shared, wgsl)

        return shader_modules

    def _compose_pipeline(self, shared, blender, blend_mode):
        """Create a list of wgpu pipeline objects from the shader, bind group
        layouts and other pipeline info (one for each pass of the blender).
        """

        # todo: cache the pipeline with a lot of things as the hash
        # todo: cache vertex descriptors

        device = shared.device
        strip_index_format = self.strip_index_format
        vertex_buffer_descriptors = self.vertex_buffer_descriptors
        primitive_topology = self.pipeline_info["primitive_topology"]
        cull_mode = self.pipeline_info["cull_mode"]

        # Create pipeline layout object from list of layouts
        pipeline_layout = device.create_pipeline_layout(
            bind_group_layouts=self.wgpu_bind_group_layouts
        )

        # Instantiate the pipeline objects
        pipelines = {}
        for pass_index in range(blender.get_pass_count()):
            color_descriptors = blender.get_color_descriptors(pass_index)
            depth_descriptor = blender.get_depth_descriptor(pass_index)
            if not color_descriptors:
                continue

            shader_module = self.wgpu_shaders[blend_mode][pass_index]

            pipelines[pass_index] = device.create_render_pipeline(
                layout=pipeline_layout,
                vertex={
                    "module": shader_module,
                    "entry_point": "vs_main",
                    "buffers": vertex_buffer_descriptors,
                },
                primitive={
                    "topology": primitive_topology,
                    "strip_index_format": strip_index_format,
                    "front_face": wgpu.FrontFace.ccw,
                    "cull_mode": cull_mode,
                },
                depth_stencil={
                    **depth_descriptor,
                    "stencil_front": {},  # use defaults
                    "stencil_back": {},  # use defaults
                },
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

        return pipelines

    def dispatch(self, render_pass, blender, render_mask):
        """Dispatch the pipeline, doing the actual rendering job."""
        if not (render_mask & self.render_info["render_mask"]):
            return
        blend_mode = blender.__class__.__name__

        # Collect what's needed
        pipeline = self.wgpu_pipelines[blend_mode][pass_index]
        indices = self.render_info["indices"]
        index_buffer = self.resources["index_buffer"]
        vertex_buffers = self.resources["vertex_buffers"]
        bind_groups = self.wgpu_bind_groups

        # Set pipeline and resources
        render_pass.set_pipeline(pipeline)
        for slot, vbuffer in vertex_buffers.items():
            render_pass.set_vertex_buffer(
                slot,
                vbuffer._wgpu_buffer[1],
                vbuffer.vertex_byte_range[0],
                vbuffer.vertex_byte_range[1],
            )
        for bind_group_id, bind_group in enumerate(bind_groups):
            render_pass.set_bind_group(bind_group_id, bind_group, [], 0, 99)

        # Draw!
        # draw_indexed(count_v, count_i, first_vertex, base_vertex, first_instance)
        # draw(count_vertex, count_instance, first_vertex, first_instance)
        if index_buffer is not None:
            render_pass.set_index_buffer(index_buffer, 0, index_buffer.size)
            if len(indices) == 4:
                base_vertex = 0  # A value added to each index before reading [...]
                indices = list(indices)
                indices.insert(-1, base_vertex)
            render_pass.draw_indexed(*indices)
        else:
            render_pass.draw(*indices)


def get_shader_module(shared, source):
    """Compile a shader module object, or re-use it from the cache."""
    # todo: also release shader modules that are no longer used

    assert isinstance(source, str)
    key = source  # or hash(code)

    if key not in shared.shader_cache:
        m = shared.device.create_shader_module(code=source)
        shared.shader_cache[key] = m
    return shared.shader_cache[key]
