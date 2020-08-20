import time
import weakref

import pyshader  # noqa
from pyshader import Struct, vec2, mat4
import wgpu.backends.rs

from .. import Renderer, RenderFunctionRegistry
from ...objects import WorldObject
from ...cameras import Camera
from ...datawrappers import Buffer, TextureView
from ...utils import array_from_shadertype


# Definition uniform struct with standard info related to transforms,
# provided to each shader as uniform at slot 0.
# todo: a combined transform would be nice too, for performance
stdinfo_uniform_type = Struct(
    cam_transform=mat4,
    projection_transform=mat4,
    physical_size=vec2,
    logical_size=vec2,
)

wobject_uniform_type = Struct(world_transform=mat4,)


registry = RenderFunctionRegistry()

visibility_all = (
    wgpu.ShaderStage.VERTEX | wgpu.ShaderStage.FRAGMENT | wgpu.ShaderStage.COMPUTE
)


def register_wgpu_render_function(wobject_cls, material_cls):
    """ Decorator to register a WGPU render function.
    """

    def _register_wgpu_renderer(f):
        registry.register(wobject_cls, material_cls, f)
        return f

    return _register_wgpu_renderer


class RenderInfo:
    """ The type of object passed to each wgpu render function together
    with the world object. Contains stdinfo buffer for now. In time
    will probably also include lights etc.
    """

    def __init__(self, *, stdinfo_uniform, wobject_uniform):
        self.stdinfo_uniform = stdinfo_uniform
        self.wobject_uniform = wobject_uniform


class WgpuRenderer(Renderer):
    """ A renderer that renders to a surface.
    """

    def __init__(self, canvas):
        self._canvas = canvas

        self._pipelines = []

        self._adapter = wgpu.request_adapter(
            canvas=canvas, power_preference="high-performance"
        )
        self._device = self._adapter.request_device(extensions=[], limits={})

        self._swap_chain = self._device.configure_swap_chain(
            canvas, wgpu.TextureFormat.bgra8unorm_srgb,
        )

    def render(self, scene: WorldObject, camera: Camera):
        """ Main render method, called from the canvas.
        """

        now = time.perf_counter()  # noqa
        # Uncomment to show FPS each second
        # if not hasattr(self, "_fps"):
        #     self._fps = now, now, 1
        # elif now > self._fps[0] + 1:
        #     print(f"FPS mean: {self._fps[2]/(now - self._fps[0]):0.1f} last: {1/(now-self._fps[1]):0.1f}")
        #     self._fps = now, now, 1
        # else:
        #     self._fps = self._fps[0], now, self._fps[2] + 1

        # todo: support for alt render pipelines (object that renders to texture then renders that)
        # todo: also note that the fragment shader is (should be) optional
        #      (e.g. depth only passes like shadow mapping or z prepass)

        device = self._device
        physical_size = self._canvas.get_physical_size()  # 2 ints
        logical_size = self._canvas.get_logical_size()  # 2 floats
        # pixelratio = self._canvas.get_pixel_ratio()

        # Ensure that matrices are up-to-date
        scene.update_matrix_world()
        camera.set_viewport_size(*logical_size)
        camera.update_matrix_world()  # camera may not be a member of the scene
        camera.update_projection_matrix()

        # Get the list of objects to render (visible and having a material)
        q = self.get_render_list(scene)

        # Update stdinfo uniform buffer object that we'll use during this render call
        self._update_stdinfo_buffer(camera, physical_size, logical_size)

        # Ensure each wobject has pipeline info
        for wobject in q:
            self._ensure_up_to_date(wobject)

        # Filter out objects that we cannot render
        q = [wobject for wobject in q if wobject._wgpu_pipeline_objects is not None]

        with self._swap_chain as texture_view_target:

            # Prepare depth texture
            if texture_view_target.size != getattr(self, "_swap_chain_size", None):
                self._swap_chain_size = texture_view_target.size
                self._depth_texture = device.create_texture(
                    size=texture_view_target.size,
                    usage=wgpu.TextureUsage.OUTPUT_ATTACHMENT,
                    dimension="2d",
                    format=wgpu.TextureFormat.depth32float,
                )
                self._depth_texture_view = self._depth_texture.create_view()
                #
                # self._render_texture = device.create_texture(
                #     size=texture_view_target.size,
                #     usage=wgpu.TextureUsage.OUTPUT_ATTACHMENT,
                #     dimension="2d",
                #     format=wgpu.TextureFormat.bgra8unorm_srgb,
                # )
                # self._render_texture_view = self._render_texture.create_view()

            self._render_texture_view = texture_view_target

            command_encoder = device.create_command_encoder()
            self._render_recording(command_encoder, q)
            self._command_buffers = [command_encoder.finish()]

            device.default_queue.submit(self._command_buffers)

    def _render_recording(self, command_encoder, q):

        # You might think that this is slow for large number of world
        # object. But it is actually pretty good. It does iterate over
        # all world objects, and over stuff in each object. But that's
        # it, really.
        # todo: we may be able to speed this up with render bundles though

        # ----- compute pipelines

        compute_pass = command_encoder.begin_compute_pass()

        for wobject in q:
            wgpu_data = wobject._wgpu_pipeline_objects
            for pinfo in wgpu_data["compute_pipelines"]:
                compute_pass.set_pipeline(pinfo["pipeline"])
                for bind_group_id, bind_group in enumerate(pinfo["bind_groups"]):
                    compute_pass.set_bind_group(
                        bind_group_id, bind_group, [], 0, 999999
                    )
                compute_pass.dispatch(*pinfo["index_args"])

        compute_pass.end_pass()

        # ----- render pipelines rendering to the default target

        render_pass = command_encoder.begin_render_pass(
            color_attachments=[
                {
                    "attachment": self._render_texture_view,  # texture_view_target,
                    "resolve_target": None,
                    "load_value": (0, 0, 0, 0),  # LoadOp.load or color
                    "store_op": wgpu.StoreOp.store,
                }
            ],
            depth_stencil_attachment={
                "attachment": self._depth_texture_view,
                "depth_load_value": 10 ** 38,
                "depth_store_op": wgpu.StoreOp.store,
                "stencil_load_value": wgpu.LoadOp.load,
                "stencil_store_op": wgpu.StoreOp.store,
            },
            occlusion_query_set=None,
        )

        for wobject in q:
            wgpu_data = wobject._wgpu_pipeline_objects
            for pinfo in wgpu_data["render_pipelines"]:
                render_pass.set_pipeline(pinfo["pipeline"])
                for slot, vbuffer in pinfo["vertex_buffers"].items():
                    render_pass.set_vertex_buffer(
                        slot,
                        vbuffer._wgpu_buffer[1],
                        vbuffer.vertex_byte_range[0],
                        vbuffer.vertex_byte_range[1],
                    )
                for bind_group_id, bind_group in enumerate(pinfo["bind_groups"]):
                    render_pass.set_bind_group(bind_group_id, bind_group, [], 0, 99)
                # Draw with or without index buffer
                if pinfo["index_buffer"] is not None:
                    ibuffer = pinfo["index_buffer"]
                    render_pass.set_index_buffer(ibuffer, 0, ibuffer.size)
                    render_pass.draw_indexed(*pinfo["index_args"])
                else:
                    render_pass.draw(*pinfo["index_args"])

        render_pass.end_pass()

    def _update_stdinfo_buffer(self, camera, physical_size, logical_size):
        # Make sure we have a buffer object
        if not hasattr(self, "_wgpu_stdinfo_buffer"):
            self._wgpu_stdinfo_buffer = Buffer(
                array_from_shadertype(stdinfo_uniform_type), usage="uniform"
            )
        # Update its data
        stdinfo_data = self._wgpu_stdinfo_buffer.data
        stdinfo_data["cam_transform"] = tuple(camera.matrix_world_inverse.elements)
        stdinfo_data["projection_transform"] = tuple(camera.projection_matrix.elements)
        stdinfo_data["physical_size"] = physical_size
        stdinfo_data["logical_size"] = logical_size
        # Upload to GPU
        self._wgpu_stdinfo_buffer.update_range(0, 1)
        self._update_buffer(self._wgpu_stdinfo_buffer)

    def get_render_list(self, scene: WorldObject):
        """ Given a scene object, get a flat list of objects to render.
        """

        def visit(wobject):
            nonlocal q
            if wobject.visible and hasattr(wobject, "material"):
                q.append(wobject)

        q = []
        scene.traverse(visit)
        return q

    def _ensure_up_to_date(self, wobject):
        """ Update the GPU objects associated with the given wobject. Returns
        quickly if no changes are needed.
        """

        # Update the wobject's uniform
        if True:  # wobject.matrix_world_dirty:
            if not hasattr(wobject, "_wgpu_uniform_buffer"):
                wobject._wgpu_uniform_buffer = Buffer(
                    array_from_shadertype(wobject_uniform_type), usage="uniform"
                )
            wobject._wgpu_uniform_buffer.data["world_transform"] = tuple(
                wobject.matrix_world.elements
            )
            wobject._wgpu_uniform_buffer.update_range(0, 1)
            self._update_buffer(wobject._wgpu_uniform_buffer)
            # todo: wobject.matrix_world_dirty is not quite the right flag :P -> versioning too?

        # Do we need to create the pipeline infos (from the renderfunc for this wobject)?
        if wobject.versionflag > getattr(wobject, "_wgpu_versionflag", 0):
            wobject._wgpu_versionflag = wobject.versionflag
            wobject._wgpu_pipeline_infos = self._create_pipeline_infos(wobject)
            wobject._wgpu_pipeline_res = self._collect_pipeline_resources(wobject)
            wobject._wgpu_pipeline_objects = None  # Invalidate

        # Early exit?
        if not wobject._wgpu_pipeline_infos:
            return

        # Check if we need to update any resources. The number of
        # resources should typically be small. We could implement a
        # hook in the resource's versionflag setter so we only have to check
        # one flag ... but let's not optimize prematurely.
        for kind, resource in wobject._wgpu_pipeline_res:
            our_version = getattr(resource, "_wgpu_" + kind, (-1, None))[0]
            if resource.versionflag > our_version:
                update_func = getattr(self, "_update_" + kind)
                update_func(resource)
                # one of self._update_buffer self._update_texture, self._update_texture_view, self._update_sampler

        # Create gpu objects?
        if wobject._wgpu_pipeline_objects is None:
            wobject._wgpu_pipeline_objects = self._create_pipeline_objects(wobject)

    def _create_pipeline_infos(self, wobject):
        """ Use the render function for this wobject and material,
        and return a list of dicts representing pipelines in an abstract way.
        These dicts can then be turned into actual pipeline objects.
        """
        print("create pipeline for", wobject)

        # Set/update function to mark the pipeline dirty. Renderfuncs
        # can make this function be called when certain props on
        # wobject/material/geometry are set
        # todo: can remove this?
        wref = weakref.ref(wobject)
        dirtymaker = lambda *_:None # lambda *_: setattr(wref(), "_wgpu_pipeline_dirty", True)  # noqa
        wobject._wgpu_set_pipeline_dirty = dirtymaker

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
            stdinfo_uniform=self._wgpu_stdinfo_buffer,
            wobject_uniform=wobject._wgpu_uniform_buffer,
        )

        # Call render function
        pipeline_infos = renderfunc(wobject, render_info)
        if not pipeline_infos:
            pipeline_infos = None
        else:
            assert isinstance(pipeline_infos, list)

        return pipeline_infos

    def _collect_pipeline_resources(self, wobject):

        pipeline_infos = wobject._wgpu_pipeline_infos or []

        pipeline_resources = []  # List, because order matters

        # Collect list of resources. That we can we can easily iterate over
        # dependent resource on each render call.
        for pipeline_info in pipeline_infos:
            assert isinstance(pipeline_info, dict)
            buffer = pipeline_info.get("index_buffer", None)
            if buffer is not None:
                pipeline_resources.append(("buffer", buffer))
            for buffer in pipeline_info.get("vertex_buffers", {}).values():
                pipeline_resources.append(("buffer", buffer))
            for key in pipeline_info.keys():
                if key.startswith("bindings"):
                    resources = pipeline_info[key]
                    if isinstance(resources, dict):
                        resources = resources.values()
                    for binding_type, resource in resources:
                        if binding_type in (
                            wgpu.BindingType.uniform_buffer,
                            wgpu.BindingType.storage_buffer,
                            wgpu.BindingType.readonly_storage_buffer,
                        ):
                            assert isinstance(resource, Buffer)
                            pipeline_resources.append(("buffer", resource))
                        elif binding_type in (
                            wgpu.BindingType.sampled_texture,
                            wgpu.BindingType.readonly_storage_texture,
                            wgpu.BindingType.writeonly_storage_texture,
                        ):
                            assert isinstance(resource, TextureView)
                            pipeline_resources.append(("texture", resource.texture))
                            pipeline_resources.append(("texture_view", resource))
                        elif binding_type in (
                            wgpu.BindingType.sampler,
                            wgpu.BindingType.comparison_sampler,
                        ):
                            assert isinstance(resource, TextureView)
                            pipeline_resources.append(("sampler", resource))
                        else:
                            assert (
                                False
                            ), f"Unknown resource binding type {binding_type}"

        return pipeline_resources

    def _create_pipeline_objects(self, wobject):
        """ Generate wgpu pipeline objects from the list of pipeline info dicts.
        """

        # Prepare the three kinds of pipelines that we can get
        compute_pipelines = []
        render_pipelines = []
        alt_render_pipelines = []

        # Process each pipeline info object, converting each to a more concrete dict
        for pipeline_info in wobject._wgpu_pipeline_infos:
            if "vertex_shader" in pipeline_info and "fragment_shader" in pipeline_info:
                pipeline = self._compose_render_pipeline(wobject, pipeline_info)
                if pipeline_info.get("target", None) is None:
                    render_pipelines.append(pipeline)
                else:
                    raise NotImplementedError("Alternative render pipelines")
                    alt_render_pipelines.append(pipeline)
            elif "compute_shader" in pipeline_info:
                compute_pipelines.append(
                    self._compose_compute_pipeline(wobject, pipeline_info)
                )
            else:
                raise ValueError(
                    "Did not find compute_shader nor vertex_shader+fragment_shader in pipeline info."
                )

        return {
            "compute_pipelines": compute_pipelines,
            "render_pipelines": render_pipelines,
            "alt_render_pipelines": alt_render_pipelines,
        }

    def _compose_compute_pipeline(self, wobject, pipeline_info):
        """ Given a high-level compute pipeline description, creates a
        lower-level representation that can be consumed by wgpu.
        """

        # todo: cache the pipeline with the shader (and entrypoint) as a hash

        device = self._device

        # Convert indices to args for the compute_pass.dispatch() call
        indices = pipeline_info["indices"]
        if not (
            isinstance(indices, tuple)
            and len(indices) == 3
            and all(isinstance(i, int) for i in indices)
        ):
            raise RuntimeError(
                f"Compute indices must be 3-tuple of ints, not {indices}."
            )
        index_args = indices

        # Get bind groups and pipeline layout from the buffers in pipeline_info.
        # This also makes sure the buffers and textures are up-to-date.
        bind_groups, pipeline_layout = self._get_bind_groups(pipeline_info)

        # Compile shader and create pipeline object
        cshader = pipeline_info["compute_shader"]
        cs_module = device.create_shader_module(code=cshader)
        compute_pipeline = device.create_compute_pipeline(
            layout=pipeline_layout,
            compute_stage={"module": cs_module, "entry_point": "main"},
        )

        return {
            "pipeline": compute_pipeline,  # wgpu object
            "index_args": index_args,  # tuple
            "bind_groups": bind_groups,  # list of wgpu bind_group objects
        }

    def _compose_render_pipeline(self, wobject, pipeline_info):
        """ Given a high-level render pipeline description, creates a
        lower-level representation that can be consumed by wgpu.
        """

        # todo: cache the pipeline with a lot of things as the hash
        # todo: cache vertex descriptors

        device = self._device

        # If an index buffer is present, update it, and get index_format.
        wgpu_index_buffer = None
        index_format = wgpu.IndexFormat.uint32
        index_buffer = pipeline_info.get("index_buffer", None)
        if index_buffer is not None:
            wgpu_index_buffer = index_buffer._wgpu_buffer[1]
            index_format = index_buffer.format

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
                "step_mode": wgpu.InputStepMode.vertex,  # vertex or instance
                "attributes": [
                    {"format": buffer.format, "offset": 0, "shader_location": slot,}
                ],
            }
            vertex_buffers[slot] = buffer
            vertex_buffer_descriptors.append(vbo_des)

        # Get bind groups and pipeline layout from the buffers in pipeline_info.
        # This also makes sure the buffers and textures are up-to-date.
        bind_groups, pipeline_layout = self._get_bind_groups(pipeline_info)

        # Compile shaders
        vshader = pipeline_info["vertex_shader"]
        fshader = pipeline_info["fragment_shader"]
        vs_module = device.create_shader_module(code=vshader)
        fs_module = device.create_shader_module(code=fshader)

        # Instantiate the pipeline object
        pipeline = device.create_render_pipeline(
            layout=pipeline_layout,
            vertex_stage={"module": vs_module, "entry_point": "main"},
            fragment_stage={"module": fs_module, "entry_point": "main"},
            primitive_topology=pipeline_info["primitive_topology"],
            rasterization_state={
                "front_face": wgpu.FrontFace.ccw,
                "cull_mode": wgpu.CullMode.none,
                "depth_bias": 0,
                "depth_bias_slope_scale": 0.0,
                "depth_bias_clamp": 0.0,
            },
            color_states=[
                {
                    "format": wgpu.TextureFormat.bgra8unorm_srgb,
                    "alpha_blend": (
                        wgpu.BlendFactor.one,
                        wgpu.BlendFactor.zero,
                        wgpu.BlendOperation.add,
                    ),
                    "color_blend": (
                        wgpu.BlendFactor.src_alpha,
                        wgpu.BlendFactor.one_minus_src_alpha,
                        wgpu.BlendOperation.add,
                    ),
                    "write_mask": wgpu.ColorWrite.ALL,
                }
            ],
            depth_stencil_state={
                "format": wgpu.TextureFormat.depth32float,
                "depth_write_enabled": True,  # optional
                "depth_compare": wgpu.CompareFunction.less,  # optional
            },
            vertex_state={
                "index_format": index_format,
                "vertex_buffers": vertex_buffer_descriptors,
            },
            sample_count=1,
            sample_mask=0xFFFFFFFF,
            alpha_to_coverage_enabled=False,
        )

        return {
            "pipeline": pipeline,  # wgpu object
            "index_args": index_args,  # tuple
            "index_buffer": wgpu_index_buffer,  # Buffer
            "vertex_buffers": vertex_buffers,  # dict of slot -> Buffer
            "bind_groups": bind_groups,  # list of wgpu bind_group objects
        }

    def _get_bind_groups(self, pipeline_info):
        """ Given high-level information on bindings, create the corresponding
        wgpu objects. This assumes that all buffers and textures are up-to-date.
        Returns (bind_groups, pipeline_layout).
        """
        # todo: cache bind_group_layout objects
        # todo: cache pipeline_layout objects
        # todo: can perhaps be more specific about visibility

        device = self._device

        # Collect resource groups (keys e.g. "bindings1", "bindings132")
        resource_groups = []
        for key in pipeline_info.keys():
            if key.startswith("bindings"):
                i = int(key[len("bindings") :])
                assert i >= 0
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
            for slot, type_resource in resources.items():
                assert isinstance(type_resource, tuple) and len(type_resource) == 2
                binding_type, resource = type_resource

                if binding_type in (
                    wgpu.BindingType.uniform_buffer,
                    wgpu.BindingType.storage_buffer,
                    wgpu.BindingType.readonly_storage_buffer,
                ):
                    # A buffer resource
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
                            "visibility": visibility_all,
                            "type": binding_type,
                            "has_dynamic_offset": False,
                        }
                    )

                elif binding_type in (
                    wgpu.BindingType.sampled_texture,
                    wgpu.BindingType.readonly_storage_texture,
                    wgpu.BindingType.writeonly_storage_texture,
                ):
                    # A texture view resource
                    assert isinstance(resource, TextureView)
                    bindings.append(
                        {"binding": slot, "resource": resource._wgpu_texture_view[1]}
                    )
                    visibility = visibility_all
                    if binding_type == wgpu.BindingType.sampled_texture:
                        visibility = wgpu.ShaderStage.FRAGMENT
                    fmt = resource.format
                    dim = resource.view_dim
                    component_type = wgpu.TextureComponentType.sint
                    if "uint" in fmt:
                        component_type = wgpu.TextureComponentType.uint
                    if "float" in fmt or "norm" in fmt:
                        component_type = wgpu.TextureComponentType.float
                    binding_layout = {
                        "binding": slot,
                        "visibility": visibility,
                        "type": binding_type,
                        "view_dimension": getattr(wgpu.TextureViewDimension, dim, dim),
                        "texture_component_type": component_type,
                        # "multisampled": False,
                    }
                    if "storage" in binding_type:
                        binding_layout["storage_texture_format"] = fmt
                    binding_layouts.append(binding_layout)

                elif binding_type in (
                    wgpu.BindingType.sampler,
                    wgpu.BindingType.comparison_sampler,
                ):
                    # A sampler resource
                    assert isinstance(resource, TextureView)
                    bindings.append(
                        {"binding": slot, "resource": resource._wgpu_sampler[1]}
                    )
                    binding_layouts.append(
                        {
                            "binding": slot,
                            "visibility": wgpu.ShaderStage.FRAGMENT,
                            "type": binding_type,
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

    def _update_buffer(self, resource):
        buffer = getattr(resource, "_wgpu_buffer", (-1, None))[1]

        # todo: dispose an old buffer? / reuse an old buffer?

        pending_uploads = resource._pending_uploads
        resource._pending_uploads = []
        bytes_per_item = resource.nbytes // resource.nitems

        # Create buffer if needed
        if buffer is None or buffer.size != resource.nbytes:
            usage = wgpu.BufferUsage.COPY_DST
            for u in resource.usage.split("|"):
                usage |= getattr(wgpu.BufferUsage, u)
            buffer = self._device.create_buffer(size=resource.nbytes, usage=usage)

        queue = self._device.default_queue
        encoder = self._device.create_command_encoder()

        # Upload any pending data
        for offset, size in pending_uploads:
            subdata = resource._get_subdata(offset, size)
            # A: map the buffer, writes to it, then unmaps.
            # Seems nice, but requires BufferUsage.MAP_WRITE. Not recommended.
            # buffer.write_data(subdata, bytes_per_item * offset)
            # B: roll data in new buffer, copy from there to existing buffer
            tmp_buffer = self._device.create_buffer_with_data(
                data=subdata, usage=wgpu.BufferUsage.COPY_SRC,
            )
            boffset, bsize = bytes_per_item * offset, bytes_per_item * size
            encoder.copy_buffer_to_buffer(tmp_buffer, 0, buffer, boffset, bsize)
            # C: using queue. This may be sugar for B, but it may also be optimized
            # Unfortunately, this seems to crash the device :/
            # queue.write_buffer(buffer, bytes_per_item * offset, subdata, 0, subdata.nbytes)
            # D: A staging buffer/belt https://github.com/gfx-rs/wgpu-rs/blob/master/src/util/belt.rs
            # todo: look into staging buffers?

        queue.submit([encoder.finish()])
        resource._wgpu_buffer = resource.versionflag, buffer

    def _update_texture_view(self, resource):
        if resource._is_default_view:
            texture_view = resource.texture._wgpu_texture[1].create_view()
        else:
            dim = resource._view_dim
            assert resource._mip_range.step == 1
            assert resource._layer_range.step == 1
            texture_view = resource.texture._wgpu_texture[1].create_view(
                format=resource.format,
                dimension=f"{dim}d" if isinstance(dim, int) else dim,
                aspect=resource._aspect,
                base_mip_level=resource._mip_range.start,
                mip_level_count=len(resource._mip_range),
                base_array_layer=resource._layer_range.start,
                array_layer_count=len(resource._layer_range),
            )
        resource._wgpu_texture_view = resource.versionflag, texture_view

    def _update_texture(self, resource):

        texture = getattr(resource, "_wgpu_texture", (-1, None))[1]
        pending_uploads = resource._pending_uploads
        resource._pending_uploads = []

        # Create texture if needed
        if texture is None:  # todo: or needs to be replaced (e.g. resized)
            usage = wgpu.TextureUsage.COPY_DST
            for u in resource.usage.split("|"):
                usage |= getattr(wgpu.TextureUsage, u)
            texture = self._device.create_texture(
                size=resource.size,
                usage=usage,
                dimension=f"{resource.dim}d",
                format=getattr(wgpu.TextureFormat, resource.format),
                mip_level_count=1,
                sample_count=1,
            )  # todo: let resource specify mip_level_count and sample_count

        bytes_per_pixel = resource.nbytes // (
            resource.size[0] * resource.size[1] * resource.size[2]
        )

        queue = self._device.default_queue
        encoder = self._device.create_command_encoder()

        # Upload any pending data
        for offset, size in pending_uploads:
            subdata = resource._get_subdata(offset, size)
            # B: using a temp buffer
            # tmp_buffer = self._device.create_buffer_with_data(data=subdata,
            #     usage=wgpu.BufferUsage.COPY_SRC,
            # )
            # encoder.copy_buffer_to_texture(
            #     {
            #         "buffer": tmp_buffer,
            #         "offset": 0,
            #         "bytes_per_row": size[0] * bytes_per_pixel,
            #         "rows_per_image": size[1],
            #     },
            #     {
            #         "texture": texture,
            #         "mip_level": 0,
            #         "origin": offset,
            #     },
            #     copy_size=size,
            # )
            # C: using the queue, which may be doing B, but may also be optimized
            queue.write_texture(
                {"texture": texture, "origin": offset, "mip_level": 0},
                subdata,
                {"bytes_per_row": size[0] * bytes_per_pixel, "rows_per_image": size[1]},
                size,
            )

        queue.submit([encoder.finish()])
        resource._wgpu_texture = resource.versionflag, texture

    def _update_sampler(self, resource):
        # A sampler's info (and raw object) are stored on a TextureView
        amodes = resource._address_mode.replace(",", " ").split() or ["clamp"]
        while len(amodes) < 3:
            amodes.append(amodes[-1])
        filters = resource._filter.replace(",", " ").split() or ["nearest"]
        while len(filters) < 3:
            filters.append(filters[-1])
        ammap = {"clamp": "clamp-to-edge", "mirror": "mirror-repeat"}
        sampler = self._device.create_sampler(
            address_mode_u=ammap.get(amodes[0], amodes[0]),
            address_mode_v=ammap.get(amodes[1], amodes[1]),
            address_mode_w=ammap.get(amodes[2], amodes[2]),
            mag_filter=filters[0],
            min_filter=filters[1],
            mipmap_filter=filters[2],
            # lod_min_clamp -> use default 0
            # lod_max_clamp -> use default inf
            # compare -> only not-None for comparison samplers!
        )
        resource._wgpu_sampler = resource.versionflag, sampler
