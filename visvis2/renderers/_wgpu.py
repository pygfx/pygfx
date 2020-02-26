import python_shader  # noqa
import wgpu.backend.rs
import numpy as np

from ._base import Renderer
from ..objects import WorldObject
from ..cameras import Camera
from ..linalg import Matrix4, Vector3
from ..material._base import stdinfo_type
from .._wrappers import BufferWrapper


class WgpuBaseRenderer(Renderer):
    """ Render using WGPU.
    """


class WgpuOffscreenRenderer(WgpuBaseRenderer):
    """ Render using WGPU, but offscreen, not using a surface.
    """


class WgpuSurfaceRenderer(WgpuBaseRenderer):
    """ A renderer that renders to a surface.
    """

    def __init__(self, canvas):
        self._pipelines = []

        adapter = wgpu.request_adapter(power_preference="high-performance")
        self._device = adapter.request_device(extensions=[], limits={})

        self._canvas = canvas
        self._swap_chain = self._canvas.configure_swap_chain(
            self._device,
            wgpu.TextureFormat.bgra8unorm_srgb,
            wgpu.TextureUsage.OUTPUT_ATTACHMENT,
        )

        # Create uniform buffer that containse transform related data
        # is reused for *all* objects.
        # todo: or have one per scene, or per object?
        self._stdinfo_buffer = BufferWrapper(
            np.asarray(stdinfo_type()), mapped=1, usage="uniform"
        )
        # self._update_buffer(self._stdinfo_buffer)

    def get_render_list(self, scene: WorldObject, proj_screen_matrix: Matrix4):
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

        q = tuple(sorted(q, key=sort_func))

        # finally ensure they have pipeline info
        for wobject in q:
            wobject._wgpu_info = self.compose_pipeline(wobject)
        return q

    def compose_pipeline(self, wobject):

        if not wobject.material.dirty and hasattr(wobject, "_wgpu_info"):
            return wobject._wgpu_info

        pipeline_infos = wobject.material.get_wgpu_info(wobject)
        assert isinstance(pipeline_infos, list)

        wobject.material.dirty = False

        compute_pipelines = []
        render_pipelines = []
        alt_render_pipelines = []
        buffers_to_update = []

        for pipeline_info in pipeline_infos:
            if "vertex_shader" in pipeline_info and "fragment_shader" in pipeline_info:
                pipeline = self._compose_render_pipeline(
                    wobject, pipeline_info, buffers_to_update
                )
                if pipeline_info["target"] is None:
                    render_pipelines.append(pipeline)
                else:
                    raise NotImplementedError("Alternative render pipelines")
                    alt_render_pipelines.append(pipeline)
            elif "compute_shader" in pipeline_info:
                compute_pipelines.append(
                    self._compose_compute_pipeline(
                        wobject, pipeline_info, buffers_to_update
                    )
                )
            else:
                raise ValueError(
                    "Did not find compute_shader nor vertex_shader+fragment_shader."
                )

        # todo: remove buffers_to_update stuff
        # for buffer in buffers_to_update:
        # self._update_buffer(buffer)

        return {
            "compute_pipelines": compute_pipelines,
            "render_pipelines": render_pipelines,
            "alt_render_pipelines": alt_render_pipelines,
            "stdinfo": self._stdinfo_buffer.data,  # todo: or ... use a global object?
        }

    def _compose_compute_pipeline(self, wobject, pipeline_info, buffers_to_update):
        device = self._device

        bind_groups, pipeline_layout = self._compose_binding_layout(
            pipeline_info, buffers_to_update
        )

        cshader = pipeline_info["compute_shader"]
        cs_module = device.create_shader_module(code=cshader)

        compute_pipeline = device.create_compute_pipeline(
            layout=pipeline_layout,
            compute_stage={"module": cs_module, "entry_point": "main"},
        )

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

        return compute_pipeline, bind_groups, index_args

    def _compose_render_pipeline(self, wobject, pipeline_info, buffers_to_update):
        device = self._device

        bind_groups, pipeline_layout = self._compose_binding_layout(
            pipeline_info, buffers_to_update
        )

        # -- index buffer
        index_buffer = pipeline_info.get("index_buffer", None)
        index_format = wgpu.IndexFormat.uint32
        if index_buffer is not None:
            self._update_buffer(index_buffer)
            index_format_map = {
                "int16": wgpu.IndexFormat.uint16,
                "uint16": wgpu.IndexFormat.uint16,
                "int32": wgpu.IndexFormat.uint32,
                "uint32": wgpu.IndexFormat.uint32,
            }
            dtype = index_buffer._renderer_get_data_dtype_str()
            try:
                index_format = index_format_map[dtype]
            except KeyError:
                raise TypeError(
                    "Need dtype (u)int16 or (u)int32 for index data, not '{dtype}'."
                )

        # Get indices
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
        # Convert to args (count_vertex, count_instance, first_vertex, first_instance)
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
            index_args.insert(-1, base_vertex)  # insert at second-but-last place

        # # -- vertex buffers
        # Ref: https://github.com/gfx-rs/wgpu-rs/blob/master/examples/cube/main.rs
        vertex_buffers = []
        vertex_buffer_descriptors = []
        # We might just do it all without VBO's
        # for array in wobject.geometry.vertex_data:
        #     nbytes = array.nbytes
        #     usage = wgpu.BufferUsage.VERTEX
        #     buffer = device.create_buffer_mapped(size=nbytes, usage=usage)
        #     # Copy data from array to buffer
        #     ctypes.memmove(buffer.mapping, array.ctypes.data, nbytes)
        #     buffer.unmap()
        #     shader_location = len(buffers)
        #     # buffers[shader_location] = buffer
        #     vbo_des = {
        #         "array_stride": 3 * 4,
        #         "stepmode": wgpu.InputStepMode.vertex,
        #         "attributes": [
        #             {
        #                 "format": wgpu.VertexFormat.float3,
        #                 "offset": 0,
        #                 "shader_location": shader_location,
        #             }
        #         ],
        #     }
        #     vertex_buffers.append(buffer)
        #     vertex_buffer_descriptors.append(vbo_des)

        # ----- pipelines

        vshader = pipeline_info["vertex_shader"]
        fshader = pipeline_info["fragment_shader"]
        vs_module = device.create_shader_module(code=vshader)
        fs_module = device.create_shader_module(code=fshader)

        render_pipeline = device.create_render_pipeline(
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

        return render_pipeline, bind_groups, vertex_buffers, index_buffer, index_args

    def _compose_binding_layout(self, pipeline_info, buffers_to_update):
        device = self._device

        # Standard resources
        scene_buffers = [self._stdinfo_buffer]  # stdinfo is at slot 0 of bindgroup 0

        # Collect resource groups. The default bind group zero is the scene buffers
        resource_groups = [scene_buffers]
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
            binding_layouts = []
            bindings = []
            for slot, buffer in buffers.items():
                buffers_to_update.append(buffer)
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

            bind_group_layout = device.create_bind_group_layout(
                bindings=binding_layouts
            )
            bind_group = device.create_bind_group(
                layout=bind_group_layout, bindings=bindings
            )
            bind_groups.append(bind_group)
            bind_group_layouts.append(bind_group_layout)

        pipeline_layout = device.create_pipeline_layout(
            bind_group_layouts=bind_group_layouts
        )

        return bind_groups, pipeline_layout

    def _update_buffer(self, resource):
        assert isinstance(resource, BufferWrapper)
        if resource.dirty:
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
            # Store ob the resource object
            resource._renderer_set_gpu_buffer(buffer)
            # todo: dispose an old buffer? / reuse an old buffer?

    def render(self, scene: WorldObject, camera: Camera):
        # Called by figure/canvas

        device = self._device
        width, height, pixelratio = self._canvas.get_size_and_pixel_ratio()

        # ensure all world matrices are up to date
        scene.update_matrix_world()
        # ensure camera world matrix is up to date (it may not be member of the scene)
        camera.update_matrix_world()
        # ensure camera projection matrix is up to date
        camera.update_projection_matrix()
        # compute the screen projection matrix
        proj_screen_matrix = Matrix4().multiply_matrices(
            camera.projection_matrix, camera.matrix_world_inverse
        )
        # get the sorted list of objects to render (guaranteed to be visible and having a material)
        q = self.get_render_list(scene, proj_screen_matrix)

        current_texture_view = self._swap_chain.get_current_texture_view()
        command_encoder = device.create_command_encoder()
        # todo: what do I need to duplicate if I have two objects to draw???

        command_buffers = []

        # Update stdinfo struct for all objects
        # Set info(if we use a per-scene stdinfo object, we'd only need to update world_transform)
        for obj in q:
            info = obj._wgpu_info
            if not info:
                continue  # not drawn
            stdinfo = info["stdinfo"]
            stdinfo["world_transform"] = tuple(obj.matrix_world.elements)
            stdinfo["cam_transform"] = tuple(camera.matrix_world_inverse.elements)
            stdinfo["projection_transform"] = tuple(camera.projection_matrix.elements)
            stdinfo["physical_size"] = width, height  # or the other way around? :P
            stdinfo["logical_size"] = width * pixelratio, height * pixelratio

        # ----- compute pipelines

        compute_pass = command_encoder.begin_compute_pass()

        for obj in q:
            info = obj._wgpu_info
            if not info:
                continue  # not drawn

            for pipeline, bind_groups, index_args in info["compute_pipelines"]:
                compute_pass.set_pipeline(pipeline)
                for bind_group_id, bind_group in enumerate(bind_groups):
                    compute_pass.set_bind_group(
                        bind_group_id, bind_group, [], 0, 999999
                    )
                # print(args)
                compute_pass.dispatch(*index_args)

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

        for obj in q:
            info = obj._wgpu_info
            if not info:
                continue  # not drawn

            for xx in info["render_pipelines"]:
                pipeline, bind_groups, vertex_buffers, index_buffer, index_args = xx
                render_pass.set_pipeline(pipeline)
                for bind_group_id, bind_group in enumerate(bind_groups):
                    render_pass.set_bind_group(bind_group_id, bind_group, [], 0, 999999)
                for slot, vertex_buffer in enumerate(vertex_buffers):
                    render_pass.set_vertex_buffer(slot, vertex_buffer, 0)
                # Draw with or without index buffer
                if index_buffer is not None:
                    # todo: pr should index_buffer be a raw gpu buffer already?
                    render_pass.set_index_buffer(index_buffer.gpu_buffer, 0)
                    render_pass.draw_indexed(*index_args)
                else:
                    # print(count, first)
                    render_pass.draw(*index_args)

        render_pass.end_pass()

        # -----

        command_buffers.append(command_encoder.finish())
        device.default_queue.submit(command_buffers)
