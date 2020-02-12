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

        adapter = wgpu.requestAdapter(powerPreference="high-performance")
        self._device = adapter.requestDevice(extensions=[], limits=wgpu.GPULimits())

        self._canvas = canvas
        self._swap_chain = self._canvas.configureSwapChain(
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
        vs_module = device.createShaderModule(code=vshader)
        fs_module = device.createShaderModule(code=fshader)

        buffers = {}
        # todo: is there one namespace (of indices) for all buffers, or is vertex and storage separate?

        # -- vertex buffers
        # Ref: https://github.com/gfx-rs/wgpu-rs/blob/master/examples/cube/main.rs
        vertex_buffers = []
        vertex_buffer_descriptors = []
        for array in wobject.geometry.vertex_data:
            nbytes = array.nbytes
            usage = wgpu.BufferUsage.VERTEX
            buffer = device.createBufferMapped(size=nbytes, usage=usage)
            # Copy data from array to buffer
            ctypes.memmove(buffer.mapping, array.ctypes.data, nbytes)
            buffer.unmap()
            shader_location = len(buffers)
            buffers[shader_location] = buffer
            vbo_des = {
                "arrayStride": 3 * 4,
                "stepmode": wgpu.InputStepMode.vertex,
                "attributes": [
                    {
                        "format": wgpu.VertexFormat.float3,
                        "offset": 0,
                        "shaderLocation": shader_location,
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

        bind_group_layout = device.createBindGroupLayout(bindings=binding_layouts)
        pipeline_layout = device.createPipelineLayout(
            bindGroupLayouts=[bind_group_layout]
        )
        bind_group = device.createBindGroup(layout=bind_group_layout, bindings=bindings)

        pipeline = device.createRenderPipeline(
            layout=pipeline_layout,
            vertexStage={"module": vs_module, "entryPoint": "main"},
            fragmentStage={"module": fs_module, "entryPoint": "main"},
            primitiveTopology=wobject.material.primitiveTopology,
            rasterizationState={
                "frontFace": wgpu.FrontFace.ccw,
                "cullMode": wgpu.CullMode.none,
                "depthBias": 0,
                "depthBiasSlopeScale": 0.0,
                "depthBiasClamp": 0.0,
            },
            colorStates=[
                {
                    "format": wgpu.TextureFormat.bgra8unorm_srgb,
                    "alphaBlend": (
                        wgpu.BlendFactor.one,
                        wgpu.BlendFactor.zero,
                        wgpu.BlendOperation.add,
                    ),
                    "colorBlend": (
                        wgpu.BlendFactor.src_alpha,
                        wgpu.BlendFactor.one_minus_src_alpha,
                        wgpu.BlendOperation.add,
                    ),
                    "writeMask": wgpu.ColorWrite.ALL,
                }
            ],
            depthStencilState=None,
            vertexState={
                "indexFormat": wgpu.IndexFormat.uint32,
                "vertexBuffers": vertex_buffer_descriptors,
            },
            sampleCount=1,
            sampleMask=0xFFFFFFFF,
            alphaToCoverageEnabled=False,
        )
        return pipeline, bind_group, vertex_buffers

    def render(self, scene, camera):
        # Called by figure/canvas

        device = self._device

        # First make sure that all objects in the scene have a pipeline
        for obj in self.traverse(scene):
            obj._pipeline_info = self.compose_pipeline(obj)

        current_texture_view = self._swap_chain.getCurrentTextureView()
        command_encoder = device.createCommandEncoder()
        # todo: what do I need to duplicate if I have two objects to draw???

        command_buffers = []

        render_pass = command_encoder.beginRenderPass(
            colorAttachments=[
                {
                    "attachment": current_texture_view,
                    "resolveTarget": None,
                    "loadValue": (0, 0, 0, 1),  # LoadOp.load or color
                    "storeOp": wgpu.StoreOp.store,
                }
            ],
            depthStencilAttachment=None,
        )

        for obj in self.traverse(scene):
            pipeline, bind_group, vertex_buffers = obj._pipeline_info

            if pipeline is None:
                continue  # not drawn

            render_pass.setPipeline(pipeline)
            render_pass.setBindGroup(0, bind_group, [], 0, 999999)
            for slot, vertex_buffer in enumerate(vertex_buffers):
                render_pass.setVertexBuffer(slot, vertex_buffer, 0)
            render_pass.draw(12 * 3, 1, 0, 0)

        render_pass.endPass()
        command_buffers.append(command_encoder.finish())
        device.defaultQueue.submit(command_buffers)
