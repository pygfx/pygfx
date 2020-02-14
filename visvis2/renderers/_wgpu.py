import ctypes

import wgpu.backend.rs

from ._base import Renderer
from ..objects import Mesh, WorldObject
from ..cameras import Camera
from ..linalg import Matrix4, Vector3


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
            z = Vector3().set_from_matrix_position(wobject.matrix_world).apply_matrix4(proj_screen_matrix).z
            return wobject.render_order, z
        q = tuple(sorted(q, key=sort_func))
        
        # finally ensure they have pipeline info
        for wobject in q:
            wobject._pipeline_info = self.compose_pipeline(wobject)
        return q

    def compose_pipeline(self, wobject):
        device = self._device

        if not wobject.material.dirty and hasattr(wobject, "_pipeline_info"):
            return wobject._pipeline_info

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
            buffers[shader_location] = buffer
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

        # -- storage buffers
        binding_layouts = []
        bindings = []
        # for binding_index, buffer in buffers.items():
        #     bindings.append(
        #         {
        #             "binding": binding_index,
        #             "resource": {"buffer": buffer, "offset": 0, "size": buffer.size},
        #         }
        #     )
        #     binding_layouts.append(
        #         {
        #             "binding": binding_index,
        #             "visibility": wgpu.ShaderStage.VERTEX,  # <- it depends!
        #             "type": wgpu.BindingType.readonly_storage_buffer,
        #         }
        #     )

        bind_group_layout = device.create_bind_group_layout(bindings=binding_layouts)
        pipeline_layout = device.create_pipeline_layout(
            bind_group_layouts=[bind_group_layout]
        )
        bind_group = device.create_bind_group(
            layout=bind_group_layout, bindings=bindings
        )

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
                "index_format": wgpu.IndexFormat.uint32,
                "vertex_buffers": vertex_buffer_descriptors,
            },
            sample_count=1,
            sample_mask=0xFFFFFFFF,
            alpha_to_coverage_enabled=False,
        )
        wobject.material.dirty = False
        return pipeline, bind_group, vertex_buffers

    def render(self, scene: WorldObject, camera: Camera):
        # Called by figure/canvas

        device = self._device

        # ensure all world matrices are up to date
        scene.update_matrix_world()
        # ensure camera projection matrix is up to date
        camera.update_projection_matrix()
        # compute the screen projection matrix
        proj_screen_matrix = Matrix4().multiply_matrices(camera.projection_matrix, camera.matrix_world_inverse)
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
            pipeline, bind_group, vertex_buffers = obj._pipeline_info

            if pipeline is None:
                continue  # not drawn

            render_pass.set_pipeline(pipeline)
            render_pass.set_bind_group(0, bind_group, [], 0, 999999)
            for slot, vertex_buffer in enumerate(vertex_buffers):
                render_pass.set_vertex_buffer(slot, vertex_buffer, 0)
            render_pass.draw(12 * 3, 1, 0, 0)

        render_pass.end_pass()
        command_buffers.append(command_encoder.finish())
        device.default_queue.submit(command_buffers)
