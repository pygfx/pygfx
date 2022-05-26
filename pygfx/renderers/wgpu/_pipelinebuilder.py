"""
"""

import wgpu

from ...resources import Buffer, TextureView

from ._utils import to_vertex_format, to_texture_format
from ._update import update_resource, ALTTEXFORMAT
from . import registry


visibility_render = wgpu.ShaderStage.VERTEX | wgpu.ShaderStage.FRAGMENT
visibility_all = (
    wgpu.ShaderStage.VERTEX | wgpu.ShaderStage.FRAGMENT | wgpu.ShaderStage.COMPUTE
)


def ensure_pipeline(renderer, wobject):
    """Update the GPU objects associated with the given wobject. Returns
    quickly if no changes are needed. Only this function is used by the
    renderer.
    """

    shared = renderer._shared
    blender = renderer._blender

    # Get pipeline_container
    try:
        pipeline_container_group = wobject._wgpu_pipeline_container_group
    except AttributeError:
        pipeline_container_group = PipelineContainerGroup()
        wobject._wgpu_pipeline_container_group = pipeline_container_group
        levels = {"reset"}
    else:
        # Get whether the object has changes
        levels = wobject.pop_changed()

    # Update if necessary
    if levels:
        pipeline_container_group.update(wobject, shared, blender, levels)

    # Check if we need to update any resources. The number of resources
    # should typically be small. We could optimize though, e.g. to raise
    # a flag at the wobject if its resources need an update. Or collect
    # all resources in a scene, because some may be used by multiple wobjects.
    flat_resources = pipeline_container_group.get_flat_resources()
    for kind, resource in flat_resources:
        our_version = getattr(resource, "_wgpu_" + kind, (-1, None))[0]
        if resource.rev > our_version:
            update_resource(shared.device, resource, kind)

    # Return the pipeline container objects
    return (
        pipeline_container_group.compute_containers,
        pipeline_container_group.render_containers,
    )


class BuilderArgs:
    """The type of object passed to each pipeline builder method. Contains
    all the base objects that it might need:
    * wobject
    * shared
    * to come: something like scene or lights?
    """

    def __init__(self, *, wobject, shared):
        self.wobject = wobject
        self.shared = shared


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
        resource = self.resource
        subtype = self.type.partition("/")[2]

        if self.type.startswith("buffer/"):
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
                "visibility": self.visibility,
                "buffer": {
                    "type": getattr(wgpu.BufferBindingType, subtype),
                    "has_dynamic_offset": False,
                    "min_binding_size": 0,
                },
            }
        elif self.type.startswith("sampler/"):
            assert isinstance(resource, TextureView)
            binding = {"binding": slot, "resource": resource._wgpu_sampler[1]}
            binding_layout = {
                "binding": slot,
                "visibility": self.visibility,
                "sampler": {
                    "type": getattr(wgpu.SamplerBindingType, subtype),
                },
            }
        elif self.type.startswith("texture/"):
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
                "visibility": self.visibility,
                "texture": {
                    "sample_type": sample_type,
                    "view_dimension": dim,
                    "multisampled": False,
                },
            }
        elif self.type.startswith("storage_texture/"):
            assert isinstance(resource, TextureView)
            binding = {"binding": slot, "resource": resource._wgpu_texture_view[1]}
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
        self.compute_containers = None  # List of container objects, one for each info
        self.render_containers = None

    # todo: keep track of buffer/texture attributes changes
    # todo: when a buffer is replaced, check whether the layout still matches, and then decide what to do
    # todo: atlas should be *on* something
    # todo: lights could be a special uniform, on the scene. with dynamic size.
    # todo: clean up per-blend_mode stuff, by keeping track of what blend modes exist
    # todo: the number of indices is defined in the pipeline info

    def update(self, wobject, shared, blender, levels):

        print("update", levels)

        if self.render_containers is None or "reset" in levels:
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
            args = BuilderArgs(wobject=wobject, shared=shared)
            with wobject.track_usage("reset", False):
                builders = renderfunc(args)

            # Divide result over two bins, one for compute, and one for render
            for builder in builders:
                assert isinstance(builder, PipelineBuilder)
                if builder.type == "compute":
                    self.compute_containers.append(ComputePipelineContainer(builder))
                elif builder.type == "render":
                    self.render_containers.append(RenderPipelineContainer(builder))
                else:
                    raise ValueError(f"PipelineBuilder type {builder.type} is unknown.")

            # Trigger builder.get_shader()
            levels.add("shader")

        for container in self.compute_containers:
            container.update(wobject, shared, blender, levels)
        for container in self.render_containers:
            container.update(wobject, shared, blender, levels)

    def get_flat_resources(self):
        flat_resources = set()
        for container in self.compute_containers:
            flat_resources.update(container.flat_resources)
        for container in self.render_containers:
            flat_resources.update(container.flat_resources)
        return flat_resources


class PipelineBuilder:
    """Class that render functions return (in a list).
    Each such object represents the high level representation
    of a pipeline. It has multiple functions that must be implemented in order
    to generate information. When something in the world-object (or its sub objects)
    changes, then the appropriate steps are repeated.

    For type == "compute":

    * get_shader(): should return a (templated) Shader object.
    * get_pipeline_info(): an empty dict.
    * get_render_info(): a dict with fields "indices" (3 ints)
    * get_resources(): a dict with fields:
      * "bindings": a dict of dicts with binding objects (group_slot -> binding_slot -> binding)

    For type == "render":

    * get_shader(): should return a (templated) Shader object.
    * get_pipeline_info(): a dict with fields "cull_mode" and "primitive_topology"
    * get_render_info(): a dict with fields "render_mask" and "indices" (list of 2 or 4 ints).
    * get_resources(): a dict with fields:
      * "index_buffer": None or a Buffer object.
      * "vertex_buffer": a dict of buffer objects.
      * "bindings": a dict of dicts with binding objects (group_slot -> binding_slot -> binding)

    """

    type = "unspecified"  # must be "compute" or "render"

    def __setattr__(self, name, value):
        raise AttributeError("Its not allowed to store stuff on the builder object.")


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

        self.broken = False

    def update(self, wobject, shared, blender, levels):

        device = shared.device
        if isinstance(self, RenderPipelineContainer):
            blend_mode = blender.__class__.__name__
        else:
            blend_mode = ""

        try:
            self.update_builder_data(wobject, shared, levels)
            self.update_wgpu_data(device, blender, blend_mode)
        except Exception as err:
            self.broken = True
            raise err
        else:
            self.broken = False

    def update_builder_data(self, wobject, shared, levels):
        # Update the info that applies to all passes

        builder_args = BuilderArgs(wobject=wobject, shared=shared)

        if "shader" in levels:
            with wobject.track_usage("shader", False):
                self.shader = self.builder.get_shader(builder_args)
            self._check_shader()
            levels.update(("resources", "pipeline"))
            self.wgpu_shaders = {}

        if "pipeline" in levels:
            with wobject.track_usage("pipeline", False):
                self.pipeline_info = self.builder.get_pipeline_info(
                    builder_args, self.shader
                )
            self._check_pipeline_info()
            levels.add("render")
            self.wgpu_pipelines = {}

        if "render" in levels:
            with wobject.track_usage("render", False):
                self.render_info = self.builder.get_render_info(
                    builder_args, self.shader
                )
            self._check_render_info()

        if "resources" in levels:
            with wobject.track_usage("resources", True):
                self.resources = self.builder.get_resources(builder_args, self.shader)
            self.flat_resources = self.collect_flat_resources()
            for kind, resource in self.flat_resources:
                update_resource(shared.device, resource, kind)
            self._check_resources()
            self.update_bind_groups(shared.device)

    def update_wgpu_data(self, device, blender, blend_mode):
        # Update the actual wgpu objects

        if self.wgpu_shaders.get(blend_mode, None) is None:
            self.wgpu_shaders[blend_mode] = self._compile_shaders(device, blender)

        if self.wgpu_pipelines.get(blend_mode, None) is None:
            shader_modules = self.wgpu_shaders[blend_mode]
            self.wgpu_pipelines[blend_mode] = self._compose_pipelines(
                device, blender, shader_modules
            )

    def update_bind_groups(self, device):
        binding_groups = self.resources["bindings"]

        # Check the bindings structure
        for i, group in binding_groups.items():
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
        for group_id in sorted(binding_groups.keys()):
            bindings_dict = binding_groups[group_id]
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
                bind_group_layout = device.create_bind_group_layout(
                    entries=bg_layout_descriptor
                )
                self.wgpu_bind_group_layouts.append(bind_group_layout)

        # Always create wgpu objects for the bind groups. Note that this dict
        # includes the buffer texture objects.
        self.wgpu_bind_groups = []
        for bg_descriptor, layout in zip(bg_descriptors, self.wgpu_bind_group_layouts):
            bind_group = device.create_bind_group(layout=layout, entries=bg_descriptor)
            self.wgpu_bind_groups.append(bind_group)

    def collect_flat_resources(self):
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

        for group_dict in resources.get("bindings", {}).values():
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

        assert set(pipeline_info.keys()) == set(), f"{pipeline_info.keys()}"

    def _check_render_info(self):
        render_info = self.render_info
        assert isinstance(render_info, dict)

        assert set(render_info.keys()) == {"indices"}, f"{render_info.keys()}"

        indices = render_info["indices"]
        assert isinstance(indices, (tuple, list))
        assert all(isinstance(i, int) for i in indices)
        assert len(indices) == 3

    def _check_resources(self):
        resources = self.resources
        assert isinstance(resources, dict)
        assert set(resources.keys()) == {"bindings"}, f"{resources.keys()}"

    def _compile_shaders(self, device, blender):
        """Compile the templateds wgsl shader to a wgpu shader module."""
        wgsl = self.shader.generate_wgsl()
        shader_module = cache.get_shader_module(device, wgsl)
        return {0: shader_module}

    def _compose_pipelines(self, device, blender, shader_modules):
        """Create the wgpu pipeline object from the shader and bind group layouts."""

        # Create pipeline layout object from list of layouts
        pipeline_layout = device.create_pipeline_layout(
            bind_group_layouts=self.wgpu_bind_group_layouts
        )

        # Create pipeline object
        pipeline = device.create_compute_pipeline(
            layout=pipeline_layout,
            compute={"module": shader_modules[0], "entry_point": "main"},
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

        # Set pipeline and resources
        compute_pass.set_pipeline(pipeline)
        for bind_group_id, bind_group in enumerate(bind_groups):
            compute_pass.set_bind_group(bind_group_id, bind_group, [], 0, 999999)

        # Compute!
        compute_pass.dispatch_workgroups(*indices)


class RenderPipelineContainer(PipelineContainer):
    """Container for render pipelines."""

    def __init__(self, builder):
        super().__init__(builder)
        self.strip_index_format = 0
        self.vertex_buffer_descriptors = []

    def _check_shader(self):
        assert hasattr(self.shader, "generate_wgsl")

    def _check_pipeline_info(self):
        pipeline_info = self.pipeline_info
        assert isinstance(pipeline_info, dict)

        expected = {"cull_mode", "primitive_topology"}
        assert set(pipeline_info.keys()) == expected, f"{pipeline_info.keys()}"

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

    def _check_resources(self):
        resources = self.resources
        assert isinstance(resources, dict)

        expected = {"index_buffer", "vertex_buffers", "bindings"}
        assert set(resources.keys()) == expected, f"{resources.keys()}"

        assert isinstance(resources["index_buffer"], (None.__class__, Buffer))
        assert isinstance(resources["vertex_buffers"], dict)
        assert all(isinstance(slot, int) for slot in resources["vertex_buffers"].keys())
        assert all(isinstance(b, Buffer) for b in resources["vertex_buffers"].values())

        # Process resources - may invalidate the wgpu pipelines
        self.update_index_buffer_format()
        self.update_vertex_buffer_descriptors()

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

    def _compile_shaders(self, device, blender):
        """Compile the templated shader to a list of wgpu shader modules
        (one for each pass of the blender).
        """
        shader_modules = {}
        for pass_index in range(blender.get_pass_count()):
            color_descriptors = blender.get_color_descriptors(pass_index)
            if not color_descriptors:
                continue

            wgsl = self.shader.generate_wgsl(**blender.get_shader_kwargs(pass_index))
            shader_modules[pass_index] = cache.get_shader_module(device, wgsl)

        return shader_modules

    def _compose_pipelines(self, device, blender, shader_modules):
        """Create a list of wgpu pipeline objects from the shader, bind group
        layouts and other pipeline info (one for each pass of the blender).
        """

        # todo: cache the pipeline with a lot of things as the hash
        # todo: cache vertex descriptors

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

            shader_module = shader_modules[pass_index]

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

    def dispatch(self, render_pass, blender, pass_index, render_mask):
        """Dispatch the pipeline, doing the actual rendering job."""
        if self.broken:
            return

        # todo: take wobject.render_mask into account
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


class Cache:
    def __init__(self):
        self._d = {}

    def get(self, key):
        return self._d.get(key, None)

    def set(self, key, value):
        self._d[key] = value

    def get_shader_module(self, device, source):
        """Compile a shader module object, or re-use it from the cache."""
        # todo: also release shader modules that are no longer used
        # todo: cache more objects, like pipelines once we figure out how to clean things up

        assert isinstance(source, str)
        key = source  # or hash(code)

        m = self.get(key)
        if m is None:
            m = device.create_shader_module(code=source)
            self.set(key, m)

        return m


cache = Cache()
