import python_shader  # noqa
from python_shader import Struct, vec2, mat4
import wgpu.backend.rs

from .. import Renderer, RenderFunctionRegistry
from ...objects import WorldObject
from ...cameras import Camera
from ...linalg import Matrix4, Vector3
from ..._wrappers import BufferWrapper
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

        adapter = wgpu.request_adapter(power_preference="high-performance")
        self._device = adapter.request_device(extensions=[], limits={})

        self._swap_chain = self._canvas.configure_swap_chain(
            self._device,
            wgpu.TextureFormat.bgra8unorm_srgb,
            wgpu.TextureUsage.OUTPUT_ATTACHMENT,
        )

    def render(self, scene: WorldObject, camera: Camera):
        """ Main render method, called from the canvas.
        """

        # todo: support for alt render pipelines (object that renders to texture than renders that)

        device = self._device
        width, height, pixelratio = self._canvas.get_size_and_pixel_ratio()

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
            self._update_pipelines(wobject)

        # Filter out objects that we cannot render
        q = [wobject for wobject in q if wobject._wgpu_data is not None]

        # Prepare for rendering
        current_texture_view = self._swap_chain.get_current_texture_view()
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
            stdinfo.data["physical_size"] = width, height  # or the other way around? :P
            stdinfo.data["logical_size"] = width * pixelratio, height * pixelratio

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

        render_pass = command_encoder.begin_render_pass(
            color_attachments=[
                {
                    "attachment": current_texture_view,
                    "resolve_target": None,
                    "load_value": (0, 0, 0, 1),  # LoadOp.load or color
                    "store_op": wgpu.StoreOp.store,
                }
            ],
            depth_stencil_attachment=None,
        )

        for wobject in q:
            wgpu_data = wobject._wgpu_data
            for pinfo in wgpu_data["render_pipelines"]:
                render_pass.set_pipeline(pinfo["pipeline"])
                for bind_group_id, bind_group in enumerate(pinfo["bind_groups"]):
                    render_pass.set_bind_group(bind_group_id, bind_group, [], 0, 999999)
                for slot, vertex_buffer in enumerate(pinfo["vertex_buffers"]):
                    render_pass.set_vertex_buffer(slot, vertex_buffer, 0)
                # Draw with or without index buffer
                if pinfo["index_buffer"] is not None:
                    render_pass.set_index_buffer(pinfo["index_buffer"], 0)
                    render_pass.draw_indexed(*pinfo["index_args"])
                else:
                    render_pass.draw(*pinfo["index_args"])

        render_pass.end_pass()

        # -----

        command_buffers.append(command_encoder.finish())
        device.default_queue.submit(command_buffers)

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

    def _update_pipelines(self, wobject):
        """ Update the pipelines associated with the given wobject. Returns
        quickly if no changes are needed.
        """

        # Can return fast?
        if not wobject.material.dirty and hasattr(wobject, "_wgpu_data"):
            return

        wobject.material.dirty = False
        wobject._wgpu_data = None

        # Get render function for this world object,
        # and use it to get a high-level description of pipelines.
        renderfunc = registry.get_render_function(wobject)
        if renderfunc is None:
            wobject._wgpu_data = None

        # Make sure that the wobject has an stdinfo object
        if not hasattr(wobject, "_wgpu_stdinfo_buffer"):
            wobject._wgpu_stdinfo_buffer = BufferWrapper(
                array_from_shadertype(stdinfo_uniform_type), mapped=1, usage="uniform"
            )

        # Call render function
        render_info = RenderInfo(stdinfo=wobject._wgpu_stdinfo_buffer)
        pipeline_infos = renderfunc(wobject, render_info)
        if pipeline_infos is not None:
            assert isinstance(pipeline_infos, list)
            assert all(
                isinstance(pipeline_info, dict) for pipeline_info in pipeline_infos
            )
        else:
            return

        # Prepare the three kinds of pipelines that we can get
        compute_pipelines = []
        render_pipelines = []
        alt_render_pipelines = []

        # Process each pipeline info object, converting each to a more concrete dict
        for pipeline_info in pipeline_infos:
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

        # Store on the wobject
        wobject._wgpu_data = {
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
        index_buffer = None
        index_format = wgpu.IndexFormat.uint32
        index_buffer_wrapper = pipeline_info.get("index_buffer", None)
        if index_buffer_wrapper is not None:
            self._update_buffer(index_buffer_wrapper)
            index_buffer = index_buffer_wrapper.gpu_buffer
            index_format_map = {
                "int16": wgpu.IndexFormat.uint16,
                "uint16": wgpu.IndexFormat.uint16,
                "int32": wgpu.IndexFormat.uint32,
                "uint32": wgpu.IndexFormat.uint32,
            }
            dtype = index_buffer_wrapper._renderer_get_data_dtype_str()
            try:
                index_format = index_format_map[dtype]
            except KeyError:
                raise TypeError(
                    "Need dtype (u)int16 or (u)int32 for index data, not '{dtype}'."
                )

        # Convert and check high-level indices. Indices represent a range
        # of index id's, or define what indices in the index buffer are used.
        indices = pipeline_info.get("indices", None)
        if indices is None:
            if index_buffer_wrapper is None:
                raise RuntimeError("Need indices or index_buffer ")
            indices = range(index_buffer_wrapper.data.size)
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
        if index_buffer is not None:
            base_vertex = 0  # A value added to each index before reading [...]
            index_args.insert(-1, base_vertex)

        # Process vertex buffers. Update the buffer, and produces a descriptor.
        vertex_buffers = []
        vertex_buffer_descriptors = []
        for slot, buffer in enumerate(pipeline_info.get("vertex_buffers", [])):
            self._update_buffer(buffer)
            vbo_des = {
                "array_stride": buffer.strides[0],
                "stepmode": wgpu.InputStepMode.vertex,  # vertex or instance
                "attributes": [
                    {
                        "format": buffer._renderer_get_vertex_format(),
                        "offset": 0,
                        "shader_location": slot,
                    }
                ],
            }
            vertex_buffers.append(buffer.gpu_buffer)
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
            depth_stencil_state=None,
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
            "index_buffer": index_buffer,  # BufferWrapper
            "vertex_buffers": vertex_buffers,  # list of BufferWrapper
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
        for buffers in resource_groups:
            if not isinstance(buffers, dict):
                buffers = {slot: buffer for slot, buffer in enumerate(buffers)}
            # Collect list of dicts
            bindings = []
            binding_layouts = []
            for slot, buffer in buffers.items():
                self._update_buffer(buffer)
                bindings.append(
                    {
                        "binding": slot,
                        "resource": {
                            "buffer": buffer.gpu_buffer,
                            "offset": 0,
                            "size": buffer.nbytes,
                        },
                    }
                )
                if buffer.usage & wgpu.BufferUsage.UNIFORM:
                    buffer_type = wgpu.BindingType.uniform_buffer
                else:
                    buffer_type = wgpu.BindingType.storage_buffer
                binding_layouts.append(
                    {
                        "binding": slot,
                        "visibility": wgpu.ShaderStage.VERTEX
                        | wgpu.ShaderStage.FRAGMENT
                        | wgpu.ShaderStage.COMPUTE,
                        "type": buffer_type,
                    }
                )
            # Create wgpu objects
            bind_group_layout = device.create_bind_group_layout(
                bindings=binding_layouts
            )
            bind_group = device.create_bind_group(
                layout=bind_group_layout, bindings=bindings
            )
            bind_groups.append(bind_group)
            bind_group_layouts.append(bind_group_layout)

        # Create pipeline layout object from list of layouts
        pipeline_layout = device.create_pipeline_layout(
            bind_group_layouts=bind_group_layouts
        )

        return bind_groups, pipeline_layout

    def _update_buffer(self, resource):
        """ Ensure that a buffer is up-to-date. If the buffer is not dirty,
        this is a no-op.
        """

        # todo: dispose an old buffer? / reuse an old buffer?

        assert isinstance(resource, BufferWrapper)
        if not resource.dirty:
            return

        if not resource.mapped and resource.data is None:
            buffer = self._device.create_buffer(
                size=resource.nbytes, usage=resource.usage
            )
        else:
            buffer = self._device.create_buffer_mapped(
                size=resource.nbytes, usage=resource.usage
            )
            if resource.data is not None:
                # Copy data from array to new buffer
                resource._renderer_copy_data_to_ctypes_object(buffer.mapping)
            if resource.mapped:
                # Replace data in Python BufferWrapper object
                resource._renderer_set_data_from_ctypes_object(buffer.mapping)
            else:
                # Simply unmap
                buffer.unmap()

        # Store buffer object on the resource object
        resource._renderer_set_gpu_buffer(buffer)  # sets dirty to False
