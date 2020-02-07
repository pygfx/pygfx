import asyncio

import wgpu.backend.rs
import python_shader
import numpy as np


class BaseRenderer:
    """ Base class for other renderers. A renderer takes a figure,
    collect data that describe how it should be drawn, and then draws it.
    """

    pass


class SvgRenderer(BaseRenderer):
    """ Render to SVG. Because why not.
    """

    pass


class GlRenderer(BaseRenderer):
    """ Render with OpenGL. This is mostly there to illustrate that it *could* be done.
    WGPU can (in time) also be used to render using OpenGL.
    """

    pass


class BaseWgpuRenderer(BaseRenderer):
    """ Render using WGPU.
    """


class OffscreenWgpuRenderer(BaseWgpuRenderer):
    """ Render using WGPU, but offscreen, not using a surface.
    """


class SurfaceWgpuRenderer(BaseWgpuRenderer):
    """ A renderer that renders to a surface.
    """

    def __init__(self):
        self._pipelines = []

        adapter = wgpu.requestAdapter(powerPreference="high-performance")
        self._device = adapter.requestDevice(extensions=[], limits=wgpu.GPULimits())
        self._swap_chain = None

    def collect_from_figure(self, figure):

        wobjects = []
        for view in figure.views:
            for ob in view.scene.children:  # todo: and their children, and ...
                wobjects.append(ob)

        return wobjects

    def compose_pipeline(self, wobject):

        device = self._device

        # Get description from world object
        pipelinedescription = wobject.describe_pipeline()

        vshader = pipelinedescription["vertex_shader"]
        python_shader.dev.validate(vshader)
        vs_module = device.createShaderModule(code=vshader)

        fshader = pipelinedescription["fragment_shader"]
        python_shader.dev.validate(fshader)
        fs_module = device.createShaderModule(code=fshader)

        bindings_layout = []
        bindings = []

        bind_group_layout = device.createBindGroupLayout(bindings=bindings)
        bind_group = device.createBindGroup(layout=bind_group_layout, bindings=[])
        pipeline_layout = device.createPipelineLayout(bindGroupLayouts=[bind_group_layout])

        pipeline = device.createRenderPipeline(
            layout=pipeline_layout,
            vertexStage={"module": vs_module, "entryPoint": "main"},
            fragmentStage={"module": fs_module, "entryPoint": "main"},
            primitiveTopology=wgpu.PrimitiveTopology.triangle_list,
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

    def draw_frame(self, figure):
        # Called by figure/canvas

        device = self._device

        wobjects = self.collect_from_figure(figure)
        if len(wobjects) != len(self._pipelines):
            self._pipelines = [self.compose_pipeline(wo) for wo in wobjects]

        if not self._pipelines:
            return

        if self._swap_chain is None:
            self._swap_chain = figure.widget.configureSwapChain(
                device, wgpu.TextureFormat.bgra8unorm_srgb, wgpu.TextureUsage.OUTPUT_ATTACHMENT
            )

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

        for pipeline, bind_group in self._pipelines:
            render_pass.setPipeline(pipeline)
            render_pass.setBindGroup(0, bind_group, [], 0, 999999)
            render_pass.draw(3, 1, 0, 0)

        render_pass.endPass()
        command_buffers.append(command_encoder.finish())
        device.defaultQueue.submit(command_buffers)
