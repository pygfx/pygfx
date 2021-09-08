import wgpu  # only for flags/enums

from ...utils import array_from_shadertype
from ._shadercomposer import BaseShader


class RenderTexture:
    """Class used internally to store a texture and meta data."""

    def __init__(self, format):
        self.format = format
        self.texture = None
        self.texture_view = None
        self.size = (0, 0, 0)

    def ensure_size(self, device, size):
        """Make sure that the texture has the given size. If necessary,
        recreates the texture and texture-view objects.
        """
        if size != self.size:
            self.size = size
            usage = wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.COPY_SRC
            if self.format.startswith(("rgb", "bgr")):
                usage |= wgpu.TextureUsage.TEXTURE_BINDING
            self.texture = device.create_texture(
                size=size, usage=usage, dimension="2d", format=self.format
            )
            self.texture_view = self.texture.create_view()


class FinalShader(BaseShader):
    """The shader for the final render step (the flushing to a texture)."""

    def __init__(self):
        super().__init__()
        self["tex_coord_map"] = ""
        self["color_map"] = ""

    def get_code(self):
        return (
            self.get_definitions()
            + """

        struct VertexOutput {
            [[location(0)]] texcoord: vec2<f32>;
            [[builtin(position)]] pos: vec4<f32>;
        };

        [[group(0), binding(1)]]
        var r_sampler: sampler;
        [[group(0), binding(2)]]
        var r_tex: texture_2d<f32>;

        [[stage(vertex)]]
        fn vs_main([[builtin(vertex_index)]] index: u32) -> VertexOutput {
            var positions = array<vec2<f32>, 4>(vec2<f32>(0.0, 1.0), vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 1.0), vec2<f32>(1.0, 0.0));
            let pos = positions[index];
            var out: VertexOutput;
            out.texcoord = vec2<f32>(pos.x, 1.0 - pos.y);
            out.pos = vec4<f32>(pos * 2.0 - 1.0, 0.0, 1.0);
            return out;
        }

        [[stage(fragment)]]
        fn fs_main(in: VertexOutput) -> [[location(0)]] vec4<f32> {
            // Get info about the smoothing
            let sigma = u_render.sigma;
            let support = min(5, u_render.support);

            // Determine distance between pixels in src texture
            let stepp = vec2<f32>(1.0 / u_render.size.x, 1.0 / u_render.size.y);
            // Get texcoord, and round it to the center of the source pixels.
            // Thus, whether the sampler is linear or nearest, we get equal results.
            var tex_coord = in.texcoord.xy;
            {{ tex_coord_map }}
            let ref_coord = vec2<f32>(vec2<i32>(tex_coord / stepp)) * stepp + 0.5 * stepp;

            // Convolve. Here we apply a Gaussian kernel, the weight is calculated
            // for each pixel individually based on the distance to the actual texture
            // coordinate. This means that the textures don't even need to align.
            var val: vec4<f32> = vec4<f32>(0.0, 0.0, 0.0, 0.0);
            var weight: f32 = 0.0;
            for (var y:i32 = -support; y <= support; y = y + 1) {
                for (var x:i32 = -support; x <= support; x = x + 1) {
                    let coord = ref_coord + vec2<f32>(f32(x), f32(y)) * stepp;
                    let dist = length((tex_coord - coord) / stepp);  // in src pixels
                    let t = dist / sigma;
                    let w = exp(-0.5 * t * t);
                    val = val + textureSample(r_tex, r_sampler, coord) * w;
                    weight = weight + w;
                }
            }
            var out = val / weight;
            {{ color_map }}
            return out;
        }
    """
        )


class RenderFlusher:
    """
    Utility to flush (render) the current state of a renderer into a texture.
    """

    # todo: Once we also have the depth here, we can support things like fog

    uniform_type = dict(
        size=("float32", 2),
        sigma=("float32",),
        support=("int32",),
    )

    def __init__(self, device):
        self._shader = FinalShader()
        self._device = device
        self._pipelines = {}

        self._uniform_data = array_from_shadertype(self.uniform_type)
        self._uniform_buffer = self._device.create_buffer(
            size=self._uniform_data.nbytes,
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
        )

        self._sampler = self._device.create_sampler(
            label="render sampler",
            mag_filter="nearest",
            min_filter="nearest",
        )

    def render(self, src_color_tex, src_depth_tex, dst_color_tex, dst_format):
        """Render the (internal) result of the renderer into a texture."""
        # NOTE: cannot actually use src_depth_tex as a sample texture (BindingCollision)
        assert src_depth_tex is None
        assert isinstance(src_color_tex, wgpu.base.GPUTextureView)
        assert isinstance(dst_color_tex, wgpu.base.GPUTextureView)

        # Recreate pipeline? Use ._internal as a true identifier of the texture view
        hash = src_color_tex.size, src_color_tex._internal
        stored_hash = self._pipelines.get(dst_format, ["invalidhash"])[0]
        if hash != stored_hash:
            bind_group, render_pipeline = self._create_pipeline(
                src_color_tex, dst_format
            )
            self._pipelines[dst_format] = hash, bind_group, render_pipeline

        self._update_uniforms(src_color_tex, dst_color_tex)
        self._render(dst_color_tex, dst_format)

    def _update_uniforms(self, src_color_tex, dst_color_tex):
        # Get factor between texture sizes
        factor_x = src_color_tex.size[0] / dst_color_tex.size[0]
        factor_y = src_color_tex.size[1] / dst_color_tex.size[1]
        factor = (factor_x + factor_y) / 2

        if factor > 1:
            # The src has higher res, we can do ssaa.
            sigma = 0.5 * factor
            support = min(5, int(sigma * 3))
        else:
            # The src has lower res, interpolate + smooth.
            # Smoothing a bit more helps reduce the blockiness.
            sigma = 1
            support = 2

        self._uniform_data["size"] = src_color_tex.size[:2]
        self._uniform_data["sigma"] = sigma
        self._uniform_data["support"] = support

    def _render(self, dst_color_tex, dst_format):
        device = self._device
        _, bind_group, render_pipeline = self._pipelines[dst_format]

        command_encoder = device.create_command_encoder()

        tmp_buffer = device.create_buffer_with_data(
            data=self._uniform_data,
            usage=wgpu.BufferUsage.COPY_SRC,
        )
        command_encoder.copy_buffer_to_buffer(
            tmp_buffer, 0, self._uniform_buffer, 0, self._uniform_data.nbytes
        )

        render_pass = command_encoder.begin_render_pass(
            color_attachments=[
                {
                    "view": dst_color_tex,
                    "resolve_target": None,
                    "load_value": (0, 0, 0, 0),  # LoadOp.load or color
                    "store_op": wgpu.StoreOp.store,
                }
            ],
            depth_stencil_attachment=None,
            occlusion_query_set=None,
        )
        render_pass.set_pipeline(render_pipeline)
        render_pass.set_bind_group(0, bind_group, [], 0, 99)
        render_pass.draw(4, 1)
        render_pass.end_pass()
        device.queue.submit([command_encoder.finish()])

    def _create_pipeline(self, src_texture_view, dst_format):

        device = self._device

        shader = self._shader
        shader.define_uniform(0, 0, "u_render", self._uniform_data.dtype)
        shader_module = device.create_shader_module(code=shader.generate_wgsl())

        binding_layouts = [
            {
                "binding": 0,
                "visibility": wgpu.ShaderStage.FRAGMENT,
                "buffer": {"type": wgpu.BufferBindingType.uniform},
            },
            {
                "binding": 1,
                "visibility": wgpu.ShaderStage.FRAGMENT,
                "sampler": {"type": wgpu.SamplerBindingType.filtering},
            },
            {
                "binding": 2,
                "visibility": wgpu.ShaderStage.FRAGMENT,
                "texture": {
                    "sample_type": wgpu.TextureSampleType.float,
                    "view_dimension": wgpu.TextureViewDimension.d2,
                    "multisampled": False,
                },
            },
        ]
        bindings = [
            {
                "binding": 0,
                "resource": {
                    "buffer": self._uniform_buffer,
                    "offset": 0,
                    "size": self._uniform_data.nbytes,
                },
            },
            {"binding": 1, "resource": self._sampler},
            {"binding": 2, "resource": src_texture_view},
        ]

        bind_group_layout = device.create_bind_group_layout(entries=binding_layouts)
        pipeline_layout = device.create_pipeline_layout(
            bind_group_layouts=[bind_group_layout]
        )
        bind_group = device.create_bind_group(
            layout=bind_group_layout, entries=bindings
        )

        render_pipeline = device.create_render_pipeline(
            layout=pipeline_layout,
            vertex={
                "module": shader_module,
                "entry_point": "vs_main",
                "buffers": [],
            },
            primitive={
                "topology": wgpu.PrimitiveTopology.triangle_strip,
                "strip_index_format": wgpu.IndexFormat.uint32,
            },
            depth_stencil=None,
            multisample=None,
            fragment={
                "module": shader_module,
                "entry_point": "fs_main",
                "targets": [
                    {
                        "format": dst_format,
                        "blend": {
                            "alpha": (
                                wgpu.BlendFactor.one,
                                wgpu.BlendFactor.zero,
                                wgpu.BlendOperation.add,
                            ),
                            "color": (
                                wgpu.BlendFactor.src_alpha,
                                wgpu.BlendFactor.one_minus_src_alpha,
                                wgpu.BlendOperation.add,
                            ),
                        },
                    }
                ],
            },
        )

        return bind_group, render_pipeline
