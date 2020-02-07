import asyncio
import ctypes

import cffi
import wgpu.wgpu_ffi
import numpy as np


ffi = cffi.FFI()


def ffi_code_from_spirv_module(m):
    data = m.to_bytes()
    x = ffi.new("uint8_t[]", data)
    y = ffi.cast("uint32_t *", x)
    return dict(bytes=y, length=len(data) // 4)


# # todo: stop when window is closed ...
# async def drawer():
#     while True:
#         await asyncio.sleep(0.1)
#         # print("draw")
#         drawFrame()
#
# asyncio.get_event_loop().create_task(drawer())


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

        ffi = wgpu.wgpu_ffi.ffi

        # todo: this context is not really a context, its just an API, maybe keep it that way, or make it a context by including device_id
        ctx = wgpu.wgpu_ffi.RsWGPU()
        self._ctx = ctx

        self._surface_size = 0, 0
        # backends = 1 | 2 | 4 | 8  # modern, but Dx12 is still buggy
        backends = 2 | 4  # Vulkan or Metal
        # on HP laptop: 2 and 8 are available, but 2 does not work. 8 works, but it wants zero bind groups :/
        # 1 => Backend::Empty,
        # 2 => Backend::Vulkan,
        # 4 => Backend::Metal,
        # 8 => Backend::Dx12,
        # 16 => Backend::Dx11,
        # 32 => Backend::Gl,

        # Initialize adapter id. It will be set from the callback that we pass
        # to request_adapter_async. At the moment, the callback will be called
        # directly (in-line), but this will change in the future when wgpu also
        # introduces the concept of an event loop, to support JS Futures.
        adapter_id = None

        @ffi.callback("void(uint64_t, void *)")
        def request_adapter_callback(received, userdata):
            nonlocal adapter_id
            adapter_id = received

        ctx.request_adapter_async(
            ctx.create_RequestAdapterOptions(
                power_preference=ctx.PowerPreference_Default
            ),
            backends,
            request_adapter_callback,
            ffi.NULL,  # userdata, stub
        )

        device_des = ctx.create_DeviceDescriptor(
            extensions=ctx.create_Extensions(anisotropic_filtering=False),
            # extensions={"anisotropic_filtering": False},
            limits=ctx.create_Limits(max_bind_groups=8),
        )

        self._device_id = ctx.adapter_request_device(adapter_id, device_des)

    def collect_from_figure(self, figure):

        wobjects = []
        for view in figure.views:
            for ob in view.scene.children:  # todo: and their children, and ...
                wobjects.append(ob)

        return wobjects

    def compose_pipeline(self, wobject):
        ffi = wgpu.wgpu_ffi.ffi
        ctx = self._ctx
        device_id = self._device_id

        # Get description from world object
        pipelinedescription = wobject.describe_pipeline()

        vshader = pipelinedescription["vertex_shader"]  # an SpirVModule
        vshader.validate()
        vs_module = ctx.device_create_shader_module(
            device_id,
            ctx.create_ShaderModuleDescriptor(code=ffi_code_from_spirv_module(vshader)),
        )

        fshader = pipelinedescription["fragment_shader"]
        fshader.validate()
        fs_module = ctx.device_create_shader_module(
            device_id,
            ctx.create_ShaderModuleDescriptor(code=ffi_code_from_spirv_module(fshader)),
        )

        ##

        if True:
            # todo: wrap this up in buffer object
            # todo: wondering if we must expose the C-API so directly, or use oo api instead, like wgpu-rs.

            # Create buffer objects (UBO), each mapped to a piece of memory.
            # Create binding resources, mapping to these buffers.
            # Create bindings layout
            # Create bind group
            # Wrap bindings layout in pipeline layout object
            # Use the bind group during draw to specify UBOs

            # Pointer that device_create_buffer_mapped sets, so that we can write stuff there
            buffer_memory_pointer = ffi.new("uint8_t * *")

            # Creates a new buffer, maps it into host-visible memory, copies data from the given slice,  and finally unmaps it.
            # As in create_buffer_with_data in
            # https://github.com/gfx-rs/wgpu-rs/blob/ab787e2b2c3ff1411ee505d40d449dd4e7949e52/src/lib.rs#L829
            buffer = ctx.device_create_buffer_mapped(
                device_id,
                ctx.create_BufferDescriptor(
                    size=4*3,
                    usage=ctx.BufferUsage_UNIFORM,
                ),
                buffer_memory_pointer
            )
            if True:
                # Map a numpy array onto the data. We can now update the uniforms simply
                # by editing the numpy array, how cool is that!
                pointer_as_int = int(ffi.cast('intptr_t', buffer_memory_pointer[0]))
                pointer_as_ctypes = ctypes.cast(pointer_as_int, ctypes.POINTER(ctypes.c_uint8))
                uniforms = np.ctypeslib.as_array(pointer_as_ctypes, shape=(4*3,))
                uniforms.dtype = np.float32
                uniforms[:] = 1.0, 0.0, 0.0
                self.dev_uniforms = uniforms
            else:
                # For reference, use ffi cdata object
                import struct
                nums = 1.0, 0.0, 0.0 #+0.0, -0.5, +0.5, +0.5, -0.9, +0.7
                bb = b"".join(struct.pack("<f", i) for i in nums)
                for i in range(len(bb)):
                    buffer_memory_pointer[0][i] = bb[i]

            # Unmap the buffer, not sure what the implications are if we do not do this.
            # Seems kinda nice to map this onto a numpy array for instance ...
            # When updating a buffer, one can create a new buffer in the same way, unmap
            # it and then copy memory from one buffer to another.
            # ctx.buffer_unmap(buffer)

            # ctx.buffer_map_read_async(buffer, 0, 6*4, map_read_callback, ffi.NULL)
            # ctx.buffer_map_write_async(buffer, 0, 6*4, xxx, ffi.NULL)

            resource = ctx.create_BindingResource(
                tag = ctx.BindingResource_Buffer,  # Buffer / Sampler / TextureView
                buffer = {"_0": ctx.create_BufferBinding(buffer=buffer, size=4*3, offset=0)},
                # one of:
                # buffer = create_BufferBinding
                # sampler = WGPUSamplerId
                # texture_view = WGPUTextureViewId
            )

            bindings_layout = [
                ctx.create_BindGroupLayoutBinding(
                    binding = 0,
                    visibility = ctx.ShaderStage_FRAGMENT,
                    ty = ctx.BindingType_UniformBuffer,
                    texture_dimension = ctx.TextureDimension_D1,
                    multisampled = False,
                    dynamic = True,
                )
            ]

            bindings = [
                ctx.create_BindGroupBinding(binding=0, resource=resource)
            ]

        else:
            bindings_layout = []
            bindings = []


        bind_group_layout = ctx.device_create_bind_group_layout(
            device_id,
            ctx.create_BindGroupLayoutDescriptor(bindings=bindings_layout, bindings_length=len(bindings_layout)),
        )

        # the bind group is used during draw
        bind_group = ctx.device_create_bind_group(
            device_id,
            ctx.create_BindGroupDescriptor(
                layout=bind_group_layout, bindings=bindings, bindings_length=len(bindings)
            ),
        )
        ##

        pipeline_layout = ctx.device_create_pipeline_layout(
            device_id,
            ctx.create_PipelineLayoutDescriptor(bind_group_layouts=[bind_group_layout], bind_group_layouts_length=1),
        )

        # print(bind_group_layout, bind_group, pipeline_layout)

        # todo: a lot of these functions have device_id as first arg - this smells like a class, perhaps
        # todo: several descriptor args have a list, and another arg to provide the length of that list, because C

        pipeline = ctx.device_create_render_pipeline(
            device_id,
            ctx.create_RenderPipelineDescriptor(
                layout=pipeline_layout,
                vertex_stage=ctx.create_ProgrammableStageDescriptor(
                    module=vs_module, entry_point="main"
                ),
                # fragment_stage: Some(ctx::ProgrammableStageDescriptor {
                fragment_stage=ctx.create_ProgrammableStageDescriptor(
                    module=fs_module, entry_point="main"
                ),
                primitive_topology=ctx.PrimitiveTopology_TriangleList,
                # rasterization_state: Some(ctx::RasterizationStateDescriptor {
                rasterization_state=ctx.create_RasterizationStateDescriptor(
                    front_face=ctx.FrontFace_Ccw,
                    cull_mode=ctx.CullMode_None,
                    depth_bias=0,
                    depth_bias_slope_scale=0.0,
                    depth_bias_clamp=0.0,
                ),
                # color_states: &[ctx::ColorStateDescriptor {
                color_states=ctx.create_ColorStateDescriptor(
                    format=ctx.TextureFormat_Bgra8UnormSrgb,
                    alpha_blend=ctx.create_BlendDescriptor(
                        src_factor=ctx.BlendFactor_One,
                        dst_factor=ctx.BlendFactor_Zero,
                        operation=ctx.BlendOperation_Add,
                    ),
                    color_blend=ctx.create_BlendDescriptor(
                        src_factor=ctx.BlendFactor_SrcAlpha,
                        dst_factor=ctx.BlendFactor_OneMinusSrcAlpha,
                        operation=ctx.BlendOperation_Add,
                    ),
                    write_mask=ctx.ColorWrite_ALL,  # write_mask: ctx::ColorWrite::ALL,
                ),
                color_states_length=1,
                depth_stencil_state=None,
                vertex_input=ctx.create_VertexInputDescriptor(
                    index_format=ctx.IndexFormat_Uint16,
                    vertex_buffers=(),  # todo: VBO's go here
                    vertex_buffers_length=0,
                ),
                sample_count=1,
                sample_mask=1,  # todo: or FFFFFFFFFF-ish?
                alpha_to_coverage_enabled=False,
            ),
        )
        return pipeline, bind_group

    def _create_swapchain(self, surface_id, width, height):
        ctx = self._ctx
        device_id = self._device_id

        self._swap_chain = ctx.device_create_swap_chain(
            device_id=device_id,
            surface_id=surface_id,
            desc=ctx.create_SwapChainDescriptor(
                usage=ctx.TextureUsage_OUTPUT_ATTACHMENT,  # usage
                format=ctx.TextureFormat_Bgra8UnormSrgb,  # format: ctx::TextureFormat::Bgra8UnormSrgb,
                width=width,  # width: size.width.round() as u32,
                height=height,  # height: size.height.round() as u32,
                present_mode=ctx.PresentMode_Vsync,  # present_mode: ctx::PresentMode::Vsync,
            ),
        )

    def draw_frame(self, figure):
        # When resizing, re-create the swapchain
        cur_size = figure.size
        if cur_size != self._surface_size:
            self._surface_size = cur_size
            self._create_swapchain(figure.get_surface_id(self._ctx), *cur_size)

        wobjects = self.collect_from_figure(figure)
        if len(wobjects) != len(self._pipelines):
            self._pipelines = [self.compose_pipeline(wo) for wo in wobjects]

        if not self._pipelines:
            return

        ctx = self._ctx
        device_id = self._device_id
        swap_chain = self._swap_chain

        next_texture = ctx.swap_chain_get_next_texture(swap_chain)
        queue = ctx.device_get_queue(device_id)
        # todo: what do I need to duplicate if I have two objects to draw???

        command_buffers = []


        command_encoder = ctx.device_create_command_encoder(
            device_id, ctx.create_CommandEncoderDescriptor(todo=0)
        )

        rpass = ctx.command_encoder_begin_render_pass(
            command_encoder,
            ctx.create_RenderPassDescriptor(
                color_attachments=[
                    ctx.create_RenderPassColorAttachmentDescriptor(
                        # attachment=next_texture["view_id"],
                        # todo: arg! need struct2dict function in ffi implementation
                        attachment=next_texture["view_id"]
                        if isinstance(next_texture, dict)
                        else next_texture.view_id,
                        resolve_target=None,  # resolve_target: None,
                        load_op=ctx.LoadOp_Clear,  # load_op: ctx::LoadOp::Clear,
                        store_op=ctx.StoreOp_Store,  # store_op: ctx::StoreOp::Store,
                        clear_color=dict(
                            r=0.0, g=0.1, b=0.2, a=1.0
                        ),  # clear_color: ctx::Color::GREEN,
                    ),
                ],
                color_attachments_length=1,
                depth_stencil_attachment=None,  # depth_stencil_attachement
            ),
        )

        for pipeline, bind_group in self._pipelines:
            ctx.render_pass_set_pipeline(rpass, pipeline)

            # Set uniforms, vertex buffers, index buffer
            offsets = [0] # as long as the bindings in the group, I think
            ctx.render_pass_set_bind_group(rpass, 0, bind_group, offsets, len(offsets))
            # ctx.render_pass_set_index_buffer(&self.index_buf, 0);
            # ctx.render_pass_set_vertex_buffers(0, &[(&self.vertex_buf, 0)]);

            ctx.render_pass_draw(rpass, 3, 1, 0, 0)

        ctx.render_pass_end_pass(rpass)

        cmd_buf = ctx.command_encoder_finish(command_encoder, None)
        command_buffers.append(cmd_buf)

        ctx.queue_submit(queue, command_buffers, len(command_buffers))
        ctx.swap_chain_present(swap_chain)
