import pyshader  # noqa
from pyshader import Struct, vec2, mat4
import wgpu.backends.rs

from .. import Renderer, RenderFunctionRegistry
from ...objects import WorldObject
from ...cameras import Camera
from ...linalg import Matrix4, Vector3
from ...datawrappers import BaseBuffer, Buffer, TextureView
from ...utils import array_from_shadertype


# Definition uniform struct with standard info related to transforms,
# provided to each shader as uniform at slot 0.
stdinfo_uniform_type = Struct(
    world_transform=mat4,
    cam_transform=mat4,
    projection_transform=mat4,
    physical_size=vec2,
    logical_size=vec2,
)


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

    def __init__(self, *, stdinfo):
        self.stdinfo = stdinfo


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
        self._depth_texture_size = (0, 0)

    def render(self, scene: WorldObject, camera: Camera):
        """ Main render method, called from the canvas.
        """

        # todo: support for alt render pipelines (object that renders to texture then renders that)
        # todo: also note that the fragment shader is (should be) optional
        #      (e.g. depth only passes like shadow mapping or z prepass)

        device = self._device
        physical_size = self._canvas.get_physical_size()  # 2 ints
        logical_size = self._canvas.get_logical_size()  # 2 floats
        # pixelratio = self._canvas.get_pixel_ratio()

        # Ensure that matrices are up-to-date
        scene.update_matrix_world()
        camera.update_matrix_world()  # camera may not be a member of the scene
        camera.update_projection_matrix()

        # Get the sorted list of objects to render (guaranteed to be visible and having a material)
        proj_screen_matrix = Matrix4().multiply_matrices(
            camera.projection_matrix, camera.matrix_world_inverse
        )
        q = self.get_render_list(scene, proj_screen_matrix)

        # Ensure each wobject has pipeline info
        for wobject in q:
            self._make_up_to_date(wobject)

        # Filter out objects that we cannot render
        q = [wobject for wobject in q if wobject._wgpu_data is not None]

        # Prepate depth texture
        if self._depth_texture_size != physical_size:
            self._depth_texture_size = physical_size
            self._depth_texture = device.create_texture(
                size=(physical_size[0], physical_size[1], 1),
                usage=wgpu.TextureUsage.OUTPUT_ATTACHMENT,
                dimension="2d",
                format=wgpu.TextureFormat.depth32float,
            )
            self._depth_texture_view = self._depth_texture.create_view()

        # Prepare for rendering
        command_encoder = device.create_command_encoder()
        command_buffers = []

        # Update stdinfo buffer for all objects
        # todo: a lot of duplicate data here. Let's revisit when we implement point / line collections.
        for wobject in q:
            wgpu_data = wobject._wgpu_data
            stdinfo = wgpu_data["stdinfo"]
            stdinfo.data["world_transform"] = tuple(wobject.matrix_world.elements)
            stdinfo.data["cam_transform"] = tuple(camera.matrix_world_inverse.elements)
            stdinfo.data["projection_transform"] = tuple(
                camera.projection_matrix.elements
            )
            stdinfo.data["physical_size"] = physical_size
            stdinfo.data["logical_size"] = logical_size
            stdinfo.update_range(0, 1)
            self._update_buffer(stdinfo)

        # ----- compute pipelines

        compute_pass = command_encoder.begin_compute_pass()

        for wobject in q:
            wgpu_data = wobject._wgpu_data
            for pinfo in wgpu_data["compute_pipelines"]:
                compute_pass.set_pipeline(pinfo["pipeline"])
                for bind_group_id, bind_group in enumerate(pinfo["bind_groups"]):
                    compute_pass.set_bind_group(
                        bind_group_id, bind_group, [], 0, 999999
                    )
                compute_pass.dispatch(*pinfo["index_args"])

        compute_pass.end_pass()

        # ----- render pipelines rendering to the default target

        with self._swap_chain as texture_view_target:

            render_pass = command_encoder.begin_render_pass(
                color_attachments=[
                    {
                        "attachment": texture_view_target,
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
                wgpu_data = wobject._wgpu_data
                for pinfo in wgpu_data["render_pipelines"]:
                    render_pass.set_pipeline(pinfo["pipeline"])
                    for slot, vbuffer in enumerate(pinfo["vertex_buffers"]):
                        render_pass.set_vertex_buffer(
                            slot,
                            vbuffer._wgpu_buffer,
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

            command_buffers.append(command_encoder.finish())
            device.default_queue.submit(command_buffers)

        # -----

    def get_render_list(self, scene: WorldObject, proj_screen_matrix: Matrix4):
        """ Given a scene object, get a list of objects to render.
        """

        # start by gathering everything that is visible and has a material
        q = []

        def visit(wobject):
            nonlocal q
            if wobject.visible and hasattr(wobject, "material"):
                q.append(wobject)

        scene.traverse(visit)

        # next, sort them from back-to-front
        def sort_func(wobject: WorldObject):
            z = (
                Vector3()
                .set_from_matrix_position(wobject.matrix_world)
                .apply_matrix4(proj_screen_matrix)
                .z
            )
            return wobject.render_order, z

        return list(sorted(q, key=sort_func))

    def _make_up_to_date(self, wobject):
        """ Update the GPU objects associated with the given wobject. Returns
        quickly if no changes are needed.
        """

        # Can return fast?
        if not wobject.material.dirty and hasattr(wobject, "_wgpu_data"):
            return

        wobject.material.dirty = False

        # Need a pipeline reset?
        if getattr(wobject.material, "_wgpu_pipeline_dirty", False):
            wobject._wgpu_pipeline_infos = None

        # Do we need to create the pipeline infos (from the renderfunc for this wobject)?
        if getattr(wobject, "_wgpu_pipeline_infos", None) is None:
            wobject._wgpu_data = None
            wobject._wgpu_pipeline_infos = self._get_pipeline_infos(wobject)

        # This could be enough
        if wobject._wgpu_pipeline_infos is None:
            wobject._wgpu_data = None
            return

        # Check if we need to update any resources
        # todo: this seems like a lot of work, can we keep track of what objects
        # need an update with higher precision?
        for pipeline_info in wobject._wgpu_pipeline_infos:
            buffer = pipeline_info.get("index_buffer", None)
            if buffer is not None:
                self._update_buffer(buffer)
            for buffer in pipeline_info.get("vertex_buffers", []):
                self._update_buffer(buffer)
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
                            assert isinstance(resource, BaseBuffer)
                            self._update_buffer(resource)
                        elif binding_type in (
                            wgpu.BindingType.sampled_texture,
                            wgpu.BindingType.readonly_storage_texture,
                            wgpu.BindingType.writeonly_storage_texture,
                        ):
                            assert isinstance(resource, TextureView)
                            self._update_texture(resource.texture)
                            self._update_texture_view(resource)
                        elif binding_type in (
                            wgpu.BindingType.sampler,
                            wgpu.BindingType.comparison_sampler,
                        ):
                            assert isinstance(resource, TextureView)
                            self._update_sampler(resource)

        # Create gpu data?
        if wobject._wgpu_data is None:
            wobject._wgpu_data = self._get_pipeline_objects(wobject)

    def _get_pipeline_infos(self, wobject):

        # Make sure that the wobject has an stdinfo object
        if not hasattr(wobject, "_wgpu_stdinfo_buffer"):
            wobject._wgpu_stdinfo_buffer = Buffer(
                array_from_shadertype(stdinfo_uniform_type), usage="uniform"
            )

        # Get render function for this world object,
        # and use it to get a high-level description of pipelines.
        renderfunc = registry.get_render_function(wobject)
        if renderfunc is None:
            raise ValueError(
                f"Could not get a render function for {wobject.__class__.__name__} "
                f"with {wobject.material.__class__.__name__}"
            )

        # Call render function
        render_info = RenderInfo(stdinfo=wobject._wgpu_stdinfo_buffer)
        pipeline_infos = renderfunc(wobject, render_info)
        if pipeline_infos is not None:
            assert isinstance(pipeline_infos, list)
            assert all(
                isinstance(pipeline_info, dict) for pipeline_info in pipeline_infos
            )
            return pipeline_infos
        else:
            return None

    def _get_pipeline_objects(self, wobject):

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
            "stdinfo": wobject._wgpu_stdinfo_buffer,
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
            wgpu_index_buffer = index_buffer._wgpu_buffer
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
        vertex_buffers = []
        vertex_buffer_descriptors = []
        # todo: we can probably expose multiple attributes per buffer using a BufferView
        # todo: also, must vertex_buffers be a dict?
        # -> can we also leverage numpy here?
        for slot, buffer in enumerate(pipeline_info.get("vertex_buffers", [])):
            vbo_des = {
                "array_stride": buffer.nbytes // buffer.nitems,
                "step_mode": wgpu.InputStepMode.vertex,  # vertex or instance
                "attributes": [
                    {"format": buffer.format, "offset": 0, "shader_location": slot,}
                ],
            }
            vertex_buffers.append(buffer)
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
            "index_buffer": wgpu_index_buffer,  # BaseBuffer
            "vertex_buffers": vertex_buffers,  # list of BaseBuffer
            "bind_groups": bind_groups,  # list of wgpu bind_group objects
        }

    def _get_bind_groups(self, pipeline_info):
        """ Given high-level information on bindings, create the corresponding
        wgpu objects and make sure that all buffers and textures are up-to-date.
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
                    assert isinstance(resource, BaseBuffer)
                    bindings.append(
                        {
                            "binding": slot,
                            "resource": {
                                "buffer": resource._wgpu_buffer,
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
                        {"binding": slot, "resource": resource._wgpu_texture_view,}
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
                        {"binding": slot, "resource": resource._wgpu_sampler,}
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
        buffer = getattr(resource, "_wgpu_buffer", None)
        if not (buffer is None or resource.dirty):
            return

        # todo: dispose an old buffer? / reuse an old buffer?

        pending_uploads = resource._pending_uploads
        resource._pending_uploads = []

        # Create buffer if needed
        if buffer is None or buffer.size != resource.nbytes:
            usage = wgpu.BufferUsage.COPY_DST
            for u in resource.usage.split("|"):
                usage |= getattr(wgpu.BufferUsage, u)
            buffer = self._device.create_buffer(size=resource.nbytes, usage=usage)

        # Upload any pending data
        for offset, size in pending_uploads:
            bytes_per_item = resource.nbytes // resource.nitems
            boffset, bsize = bytes_per_item * offset, bytes_per_item * size
            sub_buffer = self._device.create_buffer_mapped(
                size=bsize, usage=wgpu.BufferUsage.COPY_SRC,
            )
            resource._renderer_copy_data_to_ctypes_object(
                sub_buffer.mapping, offset, size
            )
            sub_buffer.unmap()
            command_encoder = self._device.create_command_encoder()
            command_encoder.copy_buffer_to_buffer(sub_buffer, 0, buffer, boffset, bsize)
            self._device.default_queue.submit([command_encoder.finish()])
        resource._wgpu_buffer = buffer

    def _update_texture_view(self, resource):
        if getattr(resource, "_wgpu_texture_view", None) is None:
            if resource._is_default_view:
                texture_view = resource.texture._wgpu_texture.create_view()
            else:
                dim = resource._view_dim
                assert resource._mip_range.step == 1
                assert resource._layer_range.step == 1
                texture_view = resource.texture._wgpu_texture.create_view(
                    format=resource._format,
                    dimension=f"{dim}d" if isinstance(dim, int) else dim,
                    aspect=resource._aspect,
                    base_mip_level=resource._mip_range.start,
                    mip_level_count=len(resource._mip_range),
                    base_array_layer=resource._layer_range.start,
                    array_layer_count=len(resource._layer_range),
                )
            resource._wgpu_texture_view = texture_view

    def _update_texture(self, resource):
        if not resource.dirty:
            return

        texture = getattr(resource, "_wgpu_texture", None)
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

        # Upload any pending data
        for offset, size in pending_uploads:
            bytes_per_pixel = resource.nbytes // (
                resource.size[0] * resource.size[1] * resource.size[2]
            )
            nbytes = bytes_per_pixel * size[0] * size[1] * size[2]
            sub_buffer = self._device.create_buffer_mapped(
                size=nbytes, usage=wgpu.BufferUsage.COPY_SRC,
            )
            resource._renderer_copy_data_to_ctypes_object(
                sub_buffer.mapping, offset, size
            )
            sub_buffer.unmap()
            command_encoder = self._device.create_command_encoder()
            command_encoder.copy_buffer_to_texture(
                {
                    "buffer": sub_buffer,
                    "offset": 0,
                    "bytes_per_row": size[0] * bytes_per_pixel,
                    "rows_per_image": size[1],
                },
                {
                    "texture": texture,
                    "mip_level": 0,
                    "array_layer": 0,
                    "origin": offset,
                },
                copy_size=size,
            )
            self._device.default_queue.submit([command_encoder.finish()])
        resource._wgpu_texture = texture

    def _update_sampler(self, resource):
        # A sampler's info (and raw object) are stored on a TextureView
        if getattr(resource, "_wgpu_sampler", None) is None:
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
            resource._wgpu_sampler = sampler
