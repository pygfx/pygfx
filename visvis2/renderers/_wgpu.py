import ctypes

import wgpu.backend.rs

from ._base import Renderer
from ..objects import Mesh


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

    def traverse(self, obj):
        yield obj
        for child in obj.children:
            yield from self.traverse(child)

    def compose_pipeline(self, wobject):
        device = self._device

        # object type determines pipeline composition
        if not isinstance(wobject, Mesh):
            return None, None, None

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

    def render(self, scene, camera):
        # Called by figure/canvas

        device = self._device

        # First make sure that all objects in the scene have a pipeline
        for obj in self.traverse(scene):
            obj._pipeline_info = self.compose_pipeline(obj)

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

        for obj in self.traverse(scene):
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
