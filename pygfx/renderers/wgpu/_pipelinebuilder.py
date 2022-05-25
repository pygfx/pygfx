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


class Resource:
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

    def update(self, shared, blender, levels):
        if isinstance(self, RenderPipelineContainer):
            blend_mode = blender.__name__
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

        if "resources" in levels:
            with wobject.track_usage(self, "resources", True):
                self.resources = self.builder.get_resources()
            self.update_bind_groups()  # may invalidate the wgpu pipelines

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
                resource = bindings_dict[slot]
                binding_des, binding_layout_des = resource.get_bind_group_descriptors(
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


class ComputePipelineContainer(PipelineContainer):
    """Container for compute pipelines."""

    # These checks are here to validate the output of the builder methods,
    # not for user code. So its ok to use assertions here.

    def _check_shader(self):
        self.shader  # todo: isinstance

    def _check_pipeline_info(self):
        assert self.pipeline_info == {}

    def _check_render_info(self):
        render_info = self.render_info
        assert isinstance(render_info, dict)

        assert set(render_info.keys()) == {"indices"}

        indices = render_info["indices"]
        assert isinstance(indices, tuple)
        assert len(indices) == 3
        assert all(isinstance(i, int) for i in indices)

    def check_resources(self):
        resources = self.resources
        assert isinstance(resources, dict)
        assert set(resources.keys()) == {"bindings"}

    def _compile_shaders(self, shared, blender):
        wgsl = self.shader.generate_wgsl()
        shader_module = get_shader_module(shared, wgsl)
        return [shader_module]

    def _compose_pipelines(self, shared, blender):
        """Given a high-level compute pipeline description, creates a
        lower-level representation that can be consumed by wgpu.
        """

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

        return [pipeline]

    def dispatch(self, compute_pass):
        # Collect what's needed
        indices = self.render_info["indices"]
        pipeline = self.wgpu_pipeline
        bind_groups = self.wgpu_bind_groups

        # Set pipeline and dispatch
        compute_pass.set_pipeline(pipeline)
        for bind_group_id, bind_group in enumerate(bind_groups):
            compute_pass.set_bind_group(bind_group_id, bind_group, [], 0, 999999)
        compute_pass.dispatch_workgroups(*indices)


class RenderPipelineContainer(PipelineContainer):
    """Container for render pipelines."""

    def _check_shader(self):
        self.shader  # todo: isinstance

    def _check_pipeline_info(self):
        assert self.pipeline_info == {}

    def _check_render_info(self):
        assert set(self.render_info.keys()) == {"indices"}

        indices = self.render_info["indices"]
        if not (
            isinstance(indices, tuple)
            and len(indices) == 3
            and all(isinstance(i, int) for i in indices)
        ):
            raise RuntimeError(
                f"Compute indices must be 3-tuple of ints, not {indices}."
            )

    def check_resources(self):
        resources = self.resources
        assert isinstance(resources, dict)
        keys = set(resources.keys())
        keys.difference_update(range(10))
        assert not keys, f"Unexpected resources: {keys}"

    def _compile_shaders(self):
        for pass_index in range(blender.get_pass_count()):
            color_descriptors = blender.get_color_descriptors(pass_index)
            if not color_descriptors:
                continue

            wgsl = shader.generate_wgsl(**blender.get_shader_kwargs(pass_index))
            shader_module = get_shader_module(shared, wgsl)

    def _compose_pipeline(
        self, shared, blender, pipeline_info, bind_group_layouts, shader_modules
    ):
        """Given a high-level render pipeline description, creates a
        lower-level representation that can be consumed by wgpu.
        """

        # todo: cache the pipeline with a lot of things as the hash
        # todo: cache vertex descriptors

        device = shared.device

        # If an index buffer is present, update it, and get index_format.
        wgpu_index_buffer = None
        index_format = wgpu.IndexFormat.uint32
        index_buffer = pipeline_info.get("index_buffer", None)
        if index_buffer is not None:
            wgpu_index_buffer = index_buffer._wgpu_buffer[1]
            index_format = to_vertex_format(index_buffer.format)
            index_format = index_format.split("x")[0].replace("s", "u")

        # Convert and check high-level indices. Indices represent a range
        # of index id's, or define what indices in the index buffer are used.
        indices = pipeline_info.get("indices", None)
        if indices is None:
            if index_buffer is None:
                raise RuntimeError("Need indices or index_buffer ")
            indices = range(index_buffer.data.size)
        # Convert to 2-element tuple (vertex, instance)
        if not isinstance(indices, tuple):
            indices = (indices,)
        if len(indices) == 1:
            indices = indices + (1,)  # add instancing index
        if len(indices) != 2:
            raise RuntimeError("Render pipeline indices must be a 2-element tuple.")

        # Convert indices to args for the render_pass.draw() or draw_indexed()
        # draw(count_vertex, count_instance, first_vertex, first_instance)
        # draw_indexed(count_v, count_i, first_vertex, base_vertex, first_instance)
        index_args = [0, 0, 0, 0]
        for i, index in enumerate(indices):
            if isinstance(index, int):
                index_args[i] = index
            elif isinstance(index, range):
                assert index.step == 1
                index_args[i] = index.stop - index.start
                index_args[i + 2] = index.start
            else:
                raise RuntimeError(
                    "Render pipeline indices must be a 2-element tuple with ints or ranges."
                )
        if wgpu_index_buffer is not None:
            base_vertex = 0  # A value added to each index before reading [...]
            index_args.insert(-1, base_vertex)

        # Process vertex buffers. Update the buffer, and produces a descriptor.
        vertex_buffers = {}
        vertex_buffer_descriptors = []
        # todo: we can probably expose multiple attributes per buffer using a BufferView
        # -> can we also leverage numpy here?
        for slot, buffer in pipeline_info.get("vertex_buffers", {}).items():
            slot = int(slot)
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
            vertex_buffers[slot] = buffer
            vertex_buffer_descriptors.append(vbo_des)

        # todo: is this how strip_index_format is supposed to work?
        strip_index_format = 0
        if "strip" in pipeline_info["primitive_topology"]:
            strip_index_format = index_format

        # Create pipeline layout object from list of layouts
        pipeline_layout = device.create_pipeline_layout(
            bind_group_layouts=bind_group_layouts
        )

        # Instantiate the pipeline objects

        pipelines = {}
        for pass_index in range(blender.get_pass_count()):
            color_descriptors = blender.get_color_descriptors(pass_index)
            depth_descriptor = blender.get_depth_descriptor(pass_index)
            if not color_descriptors:
                continue

            # Compile shader
            shader_module = shader_modules[pass_index]

            pipelines[pass_index] = device.create_render_pipeline(
                layout=pipeline_layout,
                vertex={
                    "module": shader_module,
                    "entry_point": "vs_main",
                    "buffers": vertex_buffer_descriptors,
                },
                primitive={
                    "topology": pipeline_info["primitive_topology"],
                    "strip_index_format": strip_index_format,
                    "front_face": wgpu.FrontFace.ccw,
                    "cull_mode": pipeline_info.get("cull_mode", wgpu.CullMode.none),
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

        return {
            "pipelines": pipelines,  # wgpu objects
            "index_args": index_args,  # tuple
            "index_buffer": wgpu_index_buffer,  # Buffer
            "vertex_buffers": vertex_buffers,  # dict of slot -> Buffer
        }

    def render(self):
        pass


def collect_pipeline_resources(shared, pipeline_info):

    pipeline_resources = []  # List, because order matters

    # Collect list of resources. This way we can easily iterate
    # over dependent resource on each render call. We also set the
    # usage flag of buffers and textures specified in the pipeline.
    assert isinstance(pipeline_info, dict)
    buffer = pipeline_info.get("index_buffer", None)
    if buffer is not None:
        buffer._wgpu_usage |= wgpu.BufferUsage.INDEX | wgpu.BufferUsage.STORAGE
        pipeline_resources.append(("buffer", buffer))
    for buffer in pipeline_info.get("vertex_buffers", {}).values():
        buffer._wgpu_usage |= wgpu.BufferUsage.VERTEX | wgpu.BufferUsage.STORAGE
        pipeline_resources.append(("buffer", buffer))
    for key in pipeline_info.keys():
        if key.startswith("bindings"):
            resources = pipeline_info[key]
            if isinstance(resources, dict):
                resources = resources.values()
            for binding in resources:
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


# def create_pipeline_objects(shared, blender, wobject, pipeline_infos):
#     """Generate wgpu pipeline objects from the list of pipeline info dicts."""
#
#     # Prepare the three kinds of pipelines that we can get
#     compute_pipelines = []
#     render_pipelines = []
#     alt_render_pipelines = []
#
#     # Process each pipeline info object, converting each to a more concrete dict
#     for pipeline_info in pipeline_infos:
#         if "render_shader" in pipeline_info:
#             pipeline = compose_render_pipeline(shared, blender, wobject, pipeline_info)
#             if pipeline_info.get("target", None) is None:
#                 render_pipelines.append(pipeline)
#             else:
#                 raise NotImplementedError("Alternative render pipelines")
#                 alt_render_pipelines.append(pipeline)
#         elif "compute_shader" in pipeline_info:
#             compute_pipelines.append(
#                 compose_compute_pipeline(shared, wobject, pipeline_info)
#             )
#         else:
#             raise ValueError(
#                 "Did not find compute_shader nor render_shader in pipeline info."
#             )
#
#     return {
#         "compute_pipelines": compute_pipelines,
#         "render_pipelines": render_pipelines,
#         "alt_render_pipelines": alt_render_pipelines,
#     }


def collect_bindings(pipeline_info):
    """Collect the bindings from the pipeline info. Putting them in a consistent format.
    The result is a list of dicts. And each dict maps slot index to a Binding object.
    """

    # Collect resource groups (keys e.g. "bindings1", "bindings132")
    binding_structure = []
    for key in pipeline_info.keys():
        if key.startswith("bindings"):
            i = int(key[len("bindings") :])
            assert i >= 0
            if not pipeline_info[key]:
                continue
            while len(binding_structure) <= i:
                binding_structure.append({})
            binding_structure[i] = pipeline_info[key]

    # Create bind groups and bind group layouts
    for i in range(len(binding_structure)):
        bindings = binding_structure[i]  # dict or list
        if not isinstance(bindings, dict):
            binding_structure[i] = {
                slot: resource for slot, resource in enumerate(bindings)
            }
        for binding in bindings.values():
            assert isinstance(binding, Binding)

    return binding_structure


def get_bind_groups(shared, binding_structure):
    """Given high-level information on bindings, create the corresponding
    wgpu objects. This assumes that all buffers and textures are up-to-date.
    Returns (bind_groups, pipeline_layout).
    """
    # todo: cache bind_group_layout objects
    # todo: cache pipeline_layout objects

    device = shared.device

    # Create bind groups and bind group layouts
    bind_groups = []
    bind_group_layouts = []
    for bindings_dict in binding_structure:
        # Collect list of dicts
        bindings = []
        binding_layouts = []
        for slot, binding in bindings_dict.items():
            binding, binding_layout = get_bind_group(binding)
            bindings.append(binding)
            binding_layouts.append(binding_layout)

        # Create wgpu objects
        bind_group_layout = device.create_bind_group_layout(entries=binding_layouts)
        bind_group = device.create_bind_group(
            layout=bind_group_layout, entries=bindings
        )
        bind_groups.append(bind_group)
        bind_group_layouts.append(bind_group_layout)

    return bind_groups, bind_group_layouts


def get_shader_module(shared, source):
    """Compile a shader module object, or re-use it from the cache."""
    # todo: also release shader modules that are no longer used

    assert isinstance(source, str)
    key = source  # or hash(code)

    if key not in shared.shader_cache:
        m = shared.device.create_shader_module(code=source)
        shared.shader_cache[key] = m
    return shared.shader_cache[key]
