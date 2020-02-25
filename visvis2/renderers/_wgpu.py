import ctypes

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
        self._stdinfo_buffer = BufferWrapper(np.asarray(stdinfo_type()), mapped=2)
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

    def _create_buffers_and_textures(self, obj):
        buffers = {}
        textures = {}
        for slot in list(obj.bindings.keys()):
            array, mapped = obj.bindings[slot]
            if isinstance(array, int):
                nbytes, array = array, None
            else:
                nbytes = array.nbytes
            if mapped == 2:
                usage = wgpu.BufferUsage.UNIFORM
            else:
                usage = wgpu.BufferUsage.STORAGE  # | wgpu.BufferUsage.
            buffer = self._device.create_buffer_mapped(size=nbytes, usage=usage)
            if array is not None:
                # Copy data from array to buffer
                ctypes.memmove(
                    ctypes.addressof(buffer.mapping), array.ctypes.data, nbytes,
                )
                if mapped:
                    # Replace geometry's binding array. DO NOT UNMAP!
                    new_array = np.frombuffer(buffer.mapping, np.uint8, nbytes)
                    new_array.dtype = array.dtype
                    new_array.shape = array.shape
                    obj.bindings[slot] = new_array, mapped
                else:
                    # Simply unmap
                    buffer.unmap()
            buffers[slot] = buffer
        return buffers, textures

    def compose_pipeline_xx(self, wobject):
        device = self._device

        if not wobject.material.dirty and hasattr(wobject, "_wgpu_info"):
            return wobject._wgpu_info

        # -- shaders
        # assert len(wobject.material.shaders) == 2, "compute shaders not yet supported"
        cshader = wobject.material.shaders.get("compute", None)
        vshader, fshader = (
            wobject.material.shaders["vertex"],
            wobject.material.shaders["fragment"],
        )
        # python_shader.dev.validate(vshader)
        # python_shader.dev.validate(fshader)

        # -- index buffer
        if wobject.geometry.index is None:
            index_buffer = None
            index_format = wgpu.IndexFormat.uint32
        else:
            # todo: also allow a range object
            # todo: also allow mapped indices (e.g. dynamic mesh)
            array = wobject.geometry.index
            nbytes = array.nbytes
            usage = wgpu.BufferUsage.INDEX
            index_buffer = device.create_buffer_mapped(size=nbytes, usage=usage)
            # Copy data from array to buffer
            ctypes.memmove(index_buffer.mapping, array.ctypes.data, nbytes)
            index_buffer.unmap()
            # Set format
            index_format_map = {
                "int16": wgpu.IndexFormat.uint16,
                "uint16": wgpu.IndexFormat.uint16,
                "int32": wgpu.IndexFormat.uint32,
                "uint32": wgpu.IndexFormat.uint32,
            }
            try:
                index_format = index_format_map[str(array.dtype)]
            except KeyError:
                raise TypeError(
                    "Need dtype (u)int16 or (u)int32 for index data, not '{array.dtype}'."
                )

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

        # -- standard buffer
        scene_buffers, scene_textures = {}, {}
        stub_stdinfo_obj = stdinfo_type()
        nbytes = ctypes.sizeof(stub_stdinfo_obj)
        usage = wgpu.BufferUsage.UNIFORM
        uniform_buffer = device.create_buffer_mapped(size=nbytes, usage=usage)
        stdinfo = stub_stdinfo_obj.__class__.from_buffer(uniform_buffer.mapping)
        scene_buffers[0] = uniform_buffer  # stdinfo is at slot zero

        # -- buffers from the geometry
        geometry_buffers, geometry_textures = self._create_buffers_and_textures(
            wobject.geometry
        )

        # -- uniform buffer
        # todo: actually store these on the respective object.material
        material_buffers, material_textures = self._create_buffers_and_textures(
            wobject.material
        )

        # Create buffer bindings and layouts.
        # These will go in bindgroup 0, 1, 2, respecitively.
        bind_groups = []
        bind_group_layouts = []
        for buffers in [scene_buffers, geometry_buffers, material_buffers]:
            binding_layouts = []
            bindings = []
            for slot, buffer in buffers.items():
                bindings.append(
                    {
                        "binding": slot,
                        "resource": {
                            "buffer": buffer,
                            "offset": 0,
                            "size": buffer.size,
                        },
                    }
                )
                if buffer.usage & wgpu.BufferUsage.UNIFORM:
                    buffer_type = wgpu.BindingType.uniform_buffer
                elif buffer.usage & wgpu.BufferUsage.STORAGE:
                    buffer_type = wgpu.BindingType.storage_buffer
                else:
                    assert False
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

        # Get draw_range (index range or vertex range)
        if index_buffer is not None:
            draw_range = 0, wobject.geometry.index.size
        elif vertex_buffers:
            draw_range = 0, len(wobject.geometry.vertex_data[0])
        else:
            draw_range = 0, 0  # null range

        # ----- pipelines

        compute_pipeline = render_pipeline = None

        if cshader is not None:
            cs_module = device.create_shader_module(code=cshader)
            compute_pipeline = device.create_compute_pipeline(
                layout=pipeline_layout,
                compute_stage={"module": cs_module, "entry_point": "main"},
            )

        if vshader:
            vs_module = device.create_shader_module(code=vshader)
            fs_module = device.create_shader_module(code=fshader)

            render_pipeline = device.create_render_pipeline(
                layout=pipeline_layout,
                vertex_stage={"module": vs_module, "entry_point": "main"},
                fragment_stage={"module": fs_module, "entry_point": "main"},
                primitive_topology=wobject.material.primitive_topology,
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

        wobject.material.dirty = False
        return {
            "compute_pipeline": compute_pipeline,
            "render_pipeline": render_pipeline,
            "bind_groups": bind_groups,
            "draw_range": draw_range,
            "index_buffer": index_buffer,
            "vertex_buffers": vertex_buffers,
            "stdinfo": self._stdinfo_buffer,  # todo: or ... use a global object?
        }

    def compose_pipeline(self, wobject):
        device = self._device

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
            # "bind_groups": bind_groups,
            # "draw_range": draw_range,
            # "index_buffer": index_buffer,
            # "vertex_buffers": vertex_buffers,
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
        return compute_pipeline, bind_groups, indices

    def _compose_render_pipeline(self, wobject, pipeline_info, buffers_to_update):
        device = self._device

        bind_groups, pipeline_layout = self._compose_binding_layout(
            pipeline_info, buffers_to_update
        )

        # -- index buffer
        if wobject.geometry.index is None:
            index_buffer = None
            index_format = wgpu.IndexFormat.uint32
        else:
            # todo: also allow a range object
            # todo: also allow mapped indices (e.g. dynamic mesh)
            array = wobject.geometry.index
            nbytes = array.nbytes
            usage = wgpu.BufferUsage.INDEX
            index_buffer = device.create_buffer_mapped(size=nbytes, usage=usage)
            # Copy data from array to buffer
            ctypes.memmove(index_buffer.mapping, array.ctypes.data, nbytes)
            index_buffer.unmap()
            # Set format
            index_format_map = {
                "int16": wgpu.IndexFormat.uint16,
                "uint16": wgpu.IndexFormat.uint16,
                "int32": wgpu.IndexFormat.uint32,
                "uint32": wgpu.IndexFormat.uint32,
            }
            try:
                index_format = index_format_map[str(array.dtype)]
            except KeyError:
                raise TypeError(
                    "Need dtype (u)int16 or (u)int32 for index data, not '{array.dtype}'."
                )

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

        # Get draw_range (index range or vertex range)
        if index_buffer is not None:
            draw_range = 0, wobject.geometry.index.size
        elif vertex_buffers:
            draw_range = 0, len(wobject.geometry.vertex_data[0])
        else:
            draw_range = 0, 0  # null range

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

        indices = pipeline_info["indices"]

        return render_pipeline, bind_groups, vertex_buffers, indices

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
                            "buffer": buffer._gpu_buffer,
                            "offset": 0,
                            "size": buffer.nbytes,
                        },
                    }
                )
                if buffer.mapped == 2:  # also see buffer._gpu_buffer.usage
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
        if resource._dirty:
            resource._dirty = False
            if resource.mapped == 2:
                usage = wgpu.BufferUsage.UNIFORM
            else:
                usage = wgpu.BufferUsage.STORAGE
            if not resource.mapped and resource.data is None:
                buffer = self._device.create_buffer(size=resource.nbytes, usage=usage)
            else:
                buffer = self._device.create_buffer_mapped(
                    size=resource.nbytes, usage=usage
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
            resource._gpu_buffer = buffer
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

            for pipeline, bind_groups, indices in info["compute_pipelines"]:
                compute_pass.set_pipeline(pipeline)
                for bind_group_id, bind_group in enumerate(bind_groups):
                    compute_pass.set_bind_group(
                        bind_group_id, bind_group, [], 0, 999999
                    )
                args = indices[0].stop, indices[1].stop, indices[2].stop
                # print(args)
                compute_pass.dispatch(*args)

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

            for pipeline, bind_groups, vertex_buffers, indices in info[
                "render_pipelines"
            ]:
                render_pass.set_pipeline(pipeline)
                for bind_group_id, bind_group in enumerate(bind_groups):
                    render_pass.set_bind_group(bind_group_id, bind_group, [], 0, 999999)
                for slot, vertex_buffer in enumerate(vertex_buffers):
                    render_pass.set_vertex_buffer(slot, vertex_buffer, 0)
                # Draw with or without index buffer
                first, count = indices.start, indices.stop - indices.start
                if False:  # info["index_buffer"] is not None:
                    render_pass.set_index_buffer(info["index_buffer"], 0)
                    base_vertex = 0  # or first?
                    render_pass.draw_indexed(count, 1, first, base_vertex, 0)
                else:
                    # print(count, first)
                    render_pass.draw(count, 1, first, 0)

        render_pass.end_pass()

        # -----

        command_buffers.append(command_encoder.finish())
        device.default_queue.submit(command_buffers)
