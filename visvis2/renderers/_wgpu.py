import ctypes

import wgpu.backend.rs
import numpy as np

from ._base import Renderer
from ..objects import WorldObject
from ..cameras import Camera
from ..linalg import Matrix4, Vector3
from ..material._base import stdinfo_type


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
        device = self._device

        if not wobject.material.dirty and hasattr(wobject, "_wgpu_info"):
            return wobject._wgpu_info

        # -- shaders
        assert len(wobject.material.shaders) == 2, "compute shaders not yet supported"
        vshader, fshader = (
            wobject.material.shaders["vertex"],
            wobject.material.shaders["fragment"],
        )
        # python_shader.dev.validate(vshader)
        # python_shader.dev.validate(fshader)
        vs_module = device.create_shader_module(code=vshader)
        fs_module = device.create_shader_module(code=fshader)

        buffers = {}
        # todo: is there one namespace (of indices) for all buffers, or is vertex and storage separate?

        # -- index buffer
        if wobject.geometry.index is None:
            index_buffer = None
            index_format = wgpu.IndexFormat.uint32
        else:
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

        # -- vertex buffers
        # Ref: https://github.com/gfx-rs/wgpu-rs/blob/master/examples/cube/main.rs
        vertex_buffers = []
        vertex_buffer_descriptors = []
        for array in wobject.geometry.vertex_data:
            nbytes = array.nbytes
            usage = wgpu.BufferUsage.VERTEX
            buffer = device.create_buffer_mapped(size=nbytes, usage=usage)
            # Copy data from array to buffer
            ctypes.memmove(buffer.mapping, array.ctypes.data, nbytes)
            buffer.unmap()
            shader_location = len(buffers)
            # buffers[shader_location] = buffer
            vbo_des = {
                "array_stride": 3 * 4,
                "stepmode": wgpu.InputStepMode.vertex,
                "attributes": [
                    {
                        "format": wgpu.VertexFormat.float3,
                        "offset": 0,
                        "shader_location": shader_location,
                    }
                ],
            }
            vertex_buffers.append(buffer)
            vertex_buffer_descriptors.append(vbo_des)

        # -- standard buffer
        stub_stdinfo_obj = stdinfo_type()
        nbytes = ctypes.sizeof(stub_stdinfo_obj)
        usage = wgpu.BufferUsage.UNIFORM
        uniform_buffer = device.create_buffer_mapped(size=nbytes, usage=usage)
        stdinfo = stub_stdinfo_obj.__class__.from_buffer(uniform_buffer.mapping)
        buffers[0] = uniform_buffer  # stdinfo is at slot zero

        # -- uniform buffer
        for slot in list(wobject.material.bindings.keys()):
            array, mapped = wobject.material.bindings[slot]
            nbytes = array.nbytes  # ctypes.sizeof(array.nbytes)
            usage = wgpu.BufferUsage.UNIFORM
            uniform_buffer = device.create_buffer_mapped(size=nbytes, usage=usage)
            # Copy data from array to buffer
            ctypes.memmove(
                ctypes.addressof(uniform_buffer.mapping), array.ctypes.data, nbytes,
            )
            if mapped:
                # Replace material's binding array. DO NOT UNMAP!
                new_array = np.frombuffer(uniform_buffer.mapping, np.uint8, nbytes)
                new_array.dtype = array.dtype
                new_array.shape = array.shape
                wobject.material.bindings[slot] = new_array, mapped
            else:
                # Simply unmap
                uniform_buffer.unmap()
            buffers[slot] = uniform_buffer

        # -- storage buffers
        binding_layouts = []
        bindings = []
        for binding_index, buffer in buffers.items():
            bindings.append(
                {
                    "binding": binding_index,
                    "resource": {"buffer": buffer, "offset": 0, "size": buffer.size},
                }
            )
            binding_layouts.append(
                {
                    "binding": binding_index,
                    "visibility": wgpu.ShaderStage.VERTEX
                    | wgpu.ShaderStage.FRAGMENT
                    | wgpu.ShaderStage.COMPUTE,
                    "type": wgpu.BindingType.uniform_buffer,  # <- it depends!
                }
            )

        bind_group_layout = device.create_bind_group_layout(bindings=binding_layouts)
        pipeline_layout = device.create_pipeline_layout(
            bind_group_layouts=[bind_group_layout]
        )
        bind_group = device.create_bind_group(
            layout=bind_group_layout, bindings=bindings
        )

        # Get draw_range (index range or vertex range)
        if index_buffer is not None:
            draw_range = 0, wobject.geometry.index.size
        elif vertex_buffers:
            draw_range = 0, len(wobject.geometry.vertex_data[0])
        else:
            draw_range = 0, 0  # null range

        pipeline = device.create_render_pipeline(
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
            "pipeline": pipeline,
            "bind_group": bind_group,
            "draw_range": draw_range,
            "index_buffer": index_buffer,
            "vertex_buffers": vertex_buffers,
            "stdinfo": stdinfo,  # todo: or ... use a global object?
        }

    def render(self, scene: WorldObject, camera: Camera):
        # Called by figure/canvas

        device = self._device
        width, height, pixelratio = self._canvas.get_size_and_pixel_ratio()

        # ensure all world matrices are up to date
        scene.update_matrix_world()
        # ensure camera projection matrix is up to date
        camera.update_matrix_world()
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

            # Set info(if we use a per-scene stdinfo object, we'd only need to update world_transform)
            stdinfo = info["stdinfo"]
            stdinfo.world_transform = tuple(obj.matrix_world.elements)
            stdinfo.cam_transform = tuple(camera.matrix_world_inverse.elements)
            stdinfo.projection_transform = tuple(camera.projection_matrix.elements)
            stdinfo.physical_size = width, height  # or the other way around? :P
            stdinfo.logical_size = width * pixelratio, height * pixelratio

            render_pass.set_pipeline(info["pipeline"])
            render_pass.set_bind_group(0, info["bind_group"], [], 0, 999999)
            for slot, vertex_buffer in enumerate(info["vertex_buffers"]):
                render_pass.set_vertex_buffer(slot, vertex_buffer, 0)
            # Draw with or without index buffer
            draw_range = info["draw_range"]
            first, count = draw_range[0], draw_range[1] - draw_range[0]
            if info["index_buffer"] is not None:
                render_pass.set_index_buffer(info["index_buffer"], 0)
                base_vertex = 0  # or first?
                render_pass.draw_indexed(count, 1, first, base_vertex, 0)
            else:
                render_pass.draw(count, 1, first, 0)

        render_pass.end_pass()
        command_buffers.append(command_encoder.finish())
        device.default_queue.submit(command_buffers)
