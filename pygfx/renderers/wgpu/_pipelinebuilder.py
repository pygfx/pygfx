"""
"""

import wgpu

from ...resources import Buffer, TextureView

from ._utils import to_vertex_format, to_texture_format
from ._update import update_resource, ALTTEXFORMAT
from . import registry


def ensure_pipeline(renderer, wobject):
    """Update the GPU objects associated with the given wobject. Returns
    quickly if no changes are needed. Only this function is used by the
    renderer.
    """

    shared = renderer._shared
    blender = renderer._blender
    pipelines = renderer._wobject_pipelines
    device = shared.device

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
        self.blender = blender


def create_pipeline_infos(shared, blender, wobject):
    """Use the render function for this wobject and material,
    and return a list of dicts representing pipelines in an abstract way.
    These dicts can then be turned into actual pipeline objects.
    """

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
    pipeline_infos = renderfunc(render_info)
    if not pipeline_infos:
        pipeline_infos = None
    else:
        assert isinstance(pipeline_infos, list)

    return pipeline_infos


def collect_pipeline_resources(shared, wobject, pipeline_infos):

    pipeline_resources = []  # List, because order matters

    # Collect list of resources. This way we can easily iterate
    # over dependent resource on each render call. We also set the
    # usage flag of buffers and textures specified in the pipeline.
    for pipeline_info in pipeline_infos:
        assert isinstance(pipeline_info, dict)
        buffer = pipeline_info.get("index_buffer", None)
        if buffer is not None:
            buffer._wgpu_usage |= wgpu.BufferUsage.INDEX
            pipeline_resources.append(("buffer", buffer))
        for buffer in pipeline_info.get("vertex_buffers", {}).values():
            buffer._wgpu_usage |= wgpu.BufferUsage.VERTEX
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
                    elif binding.type.startswith("sampler/"):
                        assert isinstance(resource, TextureView)
                        pipeline_resources.append(("sampler", resource))
                    elif binding.type.startswith("texture/"):
                        assert isinstance(resource, TextureView)
                        resource.texture._wgpu_usage |= (
                            wgpu.TextureUsage.TEXTURE_BINDING
                        )
                        pipeline_resources.append(("texture", resource.texture))
                        pipeline_resources.append(("texture_view", resource))
                    elif binding.type.startswith("storage_texture/"):
                        assert isinstance(resource, TextureView)
                        resource.texture._wgpu_usage |= (
                            wgpu.TextureUsage.STORAGE_BINDING
                        )
                        pipeline_resources.append(("texture", resource.texture))
                        pipeline_resources.append(("texture_view", resource))
                    else:
                        raise RuntimeError(
                            f"Unknown resource binding {binding.name} of type {binding.type}"
                        )

    return pipeline_resources


def create_pipeline_objects(shared, blender, wobject, pipeline_infos):
    """Generate wgpu pipeline objects from the list of pipeline info dicts."""

    # Prepare the three kinds of pipelines that we can get
    compute_pipelines = []
    render_pipelines = []
    alt_render_pipelines = []

    # Process each pipeline info object, converting each to a more concrete dict
    for pipeline_info in pipeline_infos:
        if "render_shader" in pipeline_info:
            pipeline = compose_render_pipeline(shared, blender, wobject, pipeline_info)
            if pipeline_info.get("target", None) is None:
                render_pipelines.append(pipeline)
            else:
                raise NotImplementedError("Alternative render pipelines")
                alt_render_pipelines.append(pipeline)
        elif "compute_shader" in pipeline_info:
            compute_pipelines.append(
                compose_compute_pipeline(shared, wobject, pipeline_info)
            )
        else:
            raise ValueError(
                "Did not find compute_shader nor render_shader in pipeline info."
            )

    return {
        "compute_pipelines": compute_pipelines,
        "render_pipelines": render_pipelines,
        "alt_render_pipelines": alt_render_pipelines,
    }


def compose_compute_pipeline(shared, wobject, pipeline_info):
    """Given a high-level compute pipeline description, creates a
    lower-level representation that can be consumed by wgpu.
    """

    # todo: cache the pipeline with the shader (and entrypoint) as a hash

    device = shared.device

    # Convert indices to args for the compute_pass.dispatch() call
    indices = pipeline_info["indices"]
    if not (
        isinstance(indices, tuple)
        and len(indices) == 3
        and all(isinstance(i, int) for i in indices)
    ):
        raise RuntimeError(f"Compute indices must be 3-tuple of ints, not {indices}.")
    index_args = indices

    # Get bind groups and pipeline layout from the buffers in pipeline_info.
    # This also makes sure the buffers and textures are up-to-date.
    bind_groups, pipeline_layout = get_bind_groups(shared, pipeline_info)

    # Compile shader and create pipeline object
    cshader = pipeline_info["compute_shader"]
    cs_module = device.create_shader_module(code=cshader)
    compute_pipeline = device.create_compute_pipeline(
        layout=pipeline_layout,
        compute={"module": cs_module, "entry_point": "main"},
    )

    return {
        "pipeline": compute_pipeline,  # wgpu object
        "index_args": index_args,  # tuple
        "bind_groups": bind_groups,  # list of wgpu bind_group objects
    }


def compose_render_pipeline(shared, blender, wobject, pipeline_info):
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

    # Get bind groups and pipeline layout from the buffers in pipeline_info.
    # This also makes sure the buffers and textures are up-to-date.
    bind_groups, pipeline_layout = get_bind_groups(shared, pipeline_info)

    # todo: is this how strip_index_format is supposed to work?
    strip_index_format = 0
    if "strip" in pipeline_info["primitive_topology"]:
        strip_index_format = index_format

    # Instantiate the pipeline objects

    pipelines = {}
    for pass_index in range(blender.get_pass_count()):
        color_descriptors = blender.get_color_descriptors(pass_index)
        depth_descriptor = blender.get_depth_descriptor(pass_index)
        if not color_descriptors:
            continue

        # Compile shader
        shader = pipeline_info["render_shader"]
        wgsl = shader.generate_wgsl(**blender.get_shader_kwargs(pass_index))
        shader_module = get_shader_module(shared, wgsl)

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
        "bind_groups": bind_groups,  # list of wgpu bind_group objects
    }


def get_bind_groups(shared, pipeline_info):
    """Given high-level information on bindings, create the corresponding
    wgpu objects. This assumes that all buffers and textures are up-to-date.
    Returns (bind_groups, pipeline_layout).
    """
    # todo: cache bind_group_layout objects
    # todo: cache pipeline_layout objects

    device = shared.device

    # Collect resource groups (keys e.g. "bindings1", "bindings132")
    resource_groups = []
    for key in pipeline_info.keys():
        if key.startswith("bindings"):
            i = int(key[len("bindings") :])
            assert i >= 0
            if not pipeline_info[key]:
                continue
            while len(resource_groups) <= i:
                resource_groups.append({})
            resource_groups[i] = pipeline_info[key]

    # Create bind groups and bind group layouts
    bind_groups = []
    bind_group_layouts = []
    for resources in resource_groups:
        if not isinstance(resources, dict):
            resources = {slot: resource for slot, resource in enumerate(resources)}
        # Collect list of dicts
        bindings = []
        binding_layouts = []
        for slot, binding in resources.items():
            resource = binding.resource
            subtype = binding.type.partition("/")[2]

            if binding.type.startswith("buffer/"):
                assert isinstance(resource, Buffer)
                bindings.append(
                    {
                        "binding": slot,
                        "resource": {
                            "buffer": resource._wgpu_buffer[1],
                            "offset": 0,
                            "size": resource.nbytes,
                        },
                    }
                )
                binding_layouts.append(
                    {
                        "binding": slot,
                        "visibility": binding.visibility,
                        "buffer": {
                            "type": getattr(wgpu.BufferBindingType, subtype),
                            "has_dynamic_offset": False,
                            "min_binding_size": 0,
                        },
                    }
                )
            elif binding.type.startswith("sampler/"):
                assert isinstance(resource, TextureView)
                bindings.append(
                    {"binding": slot, "resource": resource._wgpu_sampler[1]}
                )
                binding_layouts.append(
                    {
                        "binding": slot,
                        "visibility": binding.visibility,
                        "sampler": {
                            "type": getattr(wgpu.SamplerBindingType, subtype),
                        },
                    }
                )
            elif binding.type.startswith("texture/"):
                assert isinstance(resource, TextureView)
                bindings.append(
                    {"binding": slot, "resource": resource._wgpu_texture_view[1]}
                )
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
                binding_layouts.append(
                    {
                        "binding": slot,
                        "visibility": binding.visibility,
                        "texture": {
                            "sample_type": sample_type,
                            "view_dimension": dim,
                            "multisampled": False,
                        },
                    }
                )
            elif binding.type.startswith("storage_texture/"):
                assert isinstance(resource, TextureView)
                bindings.append(
                    {"binding": slot, "resource": resource._wgpu_texture_view[1]}
                )
                dim = resource.view_dim
                dim = getattr(wgpu.TextureViewDimension, dim, dim)
                fmt = to_texture_format(resource.format)
                fmt = ALTTEXFORMAT.get(fmt, [fmt])[0]
                binding_layouts.append(
                    {
                        "binding": slot,
                        "visibility": binding.visibility,
                        "storage_texture": {
                            "access": getattr(wgpu.StorageTextureAccess, subtype),
                            "format": fmt,
                            "view_dimension": dim,
                        },
                    }
                )

        # Create wgpu objects
        bind_group_layout = device.create_bind_group_layout(entries=binding_layouts)
        bind_group = device.create_bind_group(
            layout=bind_group_layout, entries=bindings
        )
        bind_groups.append(bind_group)
        bind_group_layouts.append(bind_group_layout)

    # Create pipeline layout object from list of layouts
    pipeline_layout = device.create_pipeline_layout(
        bind_group_layouts=bind_group_layouts
    )

    return bind_groups, pipeline_layout


def get_shader_module(shared, source):
    """Compile a shader module object, or re-use it from the cache."""
    # todo: also release shader modules that are no longer used

    assert isinstance(source, str)
    key = source  # or hash(code)

    # AK: this dev snippet is so useful that I leave it as a comment.
    # After a crash you can run naga on the generated file to get better feedback.
    # with open(__import__("os").path.expanduser("~/tmp.wgsl"), "wb") as f:
    #     f.write(source.encode())

    if key not in shared.shader_cache:
        m = shared.device.create_shader_module(code=source)
        shared.shader_cache[key] = m
    return shared.shader_cache[key]
