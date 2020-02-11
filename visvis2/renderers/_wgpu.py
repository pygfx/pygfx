import wgpu.backend.rs
import python_shader
import numpy as np


from .. import Mesh, MeshBasicMaterial
from ._base import Renderer


# probably want to import these rather than define them inline here
SHADER_INFO = {MeshBasicMaterial: {"vert": "", "frag": "", "uniforms": []}}


class BaseWgpuRenderer(Renderer):
    """ Render using WGPU.
    """


class OffscreenWgpuRenderer(BaseWgpuRenderer):
    """ Render using WGPU, but offscreen, not using a surface.
    """


class SurfaceWgpuRenderer(BaseWgpuRenderer):
    """ A renderer that renders to a surface.
    """

    def __init__(self, canvas):
        self._pipelines = []

        adapter = wgpu.requestAdapter(powerPreference="high-performance")
        self._device = adapter.requestDevice(extensions=[], limits=wgpu.GPULimits())

        self._canvas = canvas
        self._swap_chain = self._canvas.configureSwapChain(
            device,
            wgpu.TextureFormat.bgra8unorm_srgb,
            wgpu.TextureUsage.OUTPUT_ATTACHMENT,
        )

        self._compose = {
            Mesh: self.pipeline_mesh,
        }

    def traverse(self, obj):
        yield obj
        for child in obj.children:
            yield from self.traverse(child)

    def pipeline_mesh(self, mesh):
        shader_info = SHADER_INFO[type(mesh.material)]

        vshader = shader_info["vertex_shader"]
        # python_shader.dev.validate(vshader)
        vs_module = device.createShaderModule(code=vshader)

        fshader = shader_info["fragment_shader"]
        # python_shader.dev.validate(fshader)
        fs_module = device.createShaderModule(code=fshader)

        bindings_layout = []
        bindings = []

        bind_group_layout = device.createBindGroupLayout(bindings=bindings)
        bind_group = device.createBindGroup(layout=bind_group_layout, bindings=[])
        pipeline_layout = device.createPipelineLayout(
            bindGroupLayouts=[bind_group_layout]
        )

    def compose_pipeline(self, wobject):

        device = self._device

        info = self._compose[type(wobject)](wobject)

        if isinstance(wobject, Mesh):
            # Get description from world object
            shaders = wobject.material.get_wgpu_shaders()

            vshader = pipelinedescription["vertex_shader"]
            # python_shader.dev.validate(vshader)
            vs_module = device.createShaderModule(code=vshader)

            fshader = pipelinedescription["fragment_shader"]
            # python_shader.dev.validate(fshader)
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
            if not hasatrr(ob, "_pipeline_info"):
                obj._pipeline_info = self.compose_pipeline(obj)
            pipeline, bind_group = obj._pipeline_info

            render_pass.setPipeline(pipeline)
            render_pass.setBindGroup(0, bind_group, [], 0, 999999)
            render_pass.draw(12, 1, 0, 0)

        render_pass.endPass()
        command_buffers.append(command_encoder.finish())
        device.defaultQueue.submit(command_buffers)
