import wgpu.backend.rs
import python_shader
import numpy as np


from .. import Mesh, MeshBasicMaterial
from ._base import Renderer


class BaseWgpuRenderer(Renderer):
    """ Render using WGPU.
    """


class WgpuOffscreenRenderer(BaseWgpuRenderer):
    """ Render using WGPU, but offscreen, not using a surface.
    """


class WgpuSurfaceRenderer(BaseWgpuRenderer):
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

        info = wobject.get_renderer_info_wgpu()
        if not info:
            return None, None

        assert len(info["shaders"]) == 2, "compute shaders not yet supported"
        vshader, fshader = info["shaders"]
        # python_shader.dev.validate(vshader)
        # python_shader.dev.validate(fshader)
        vs_module = device.createShaderModule(code=vshader)
        fs_module = device.createShaderModule(code=fshader)

        bindings_layout = []
        bindings = []

        bind_group_layout = device.createBindGroupLayout(bindings=bindings)
        bind_group = device.createBindGroup(layout=bind_group_layout, bindings=[])
        pipeline_layout = device.createPipelineLayout(
            bindGroupLayouts=[bind_group_layout]
        )

        pipeline = device.createRenderPipeline(
            layout=pipeline_layout,
            vertexStage={"module": vs_module, "entryPoint": "main"},
            fragmentStage={"module": fs_module, "entryPoint": "main"},
            primitiveTopology=info["primitiveTopology"],
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
                        wgpu.BlendFactor.one,
                        wgpu.BlendFactor.zero,
                        wgpu.BlendOperation.add,
                    ),
                    "writeMask": wgpu.ColorWrite.ALL,
                }
            ],
            depthStencilState=None,
            vertexState={"indexFormat": wgpu.IndexFormat.uint32, "vertexBuffers": []},
            sampleCount=1,
            sampleMask=0xFFFFFFFF,
            alphaToCoverageEnabled=False,
        )
        return pipeline, bind_group

    def render(self, scene, camera):
        # Called by figure/canvas

        device = self._device

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
            if not hasattr(obj, "_pipeline_info"):
                obj._pipeline_info = self.compose_pipeline(obj)
            pipeline, bind_group = obj._pipeline_info

            if pipeline is None:
                continue  # not drawn

            render_pass.setPipeline(pipeline)
            render_pass.setBindGroup(0, bind_group, [], 0, 999999)
            render_pass.draw(12, 1, 0, 0)

        render_pass.endPass()
        command_buffers.append(command_encoder.finish())
        device.defaultQueue.submit(command_buffers)
