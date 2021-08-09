import wgpu  # only for flags/enums

from ...utils import array_from_shadertype


class RenderTexture:
    """Class used internally to store a render texture and meta data."""

    def __init__(self, format):
        self.format = format
        self.texture = None
        self.texture_view = None
        self.size = (0, 0, 0)

    def set_texture_view(self, texture_view):
        """Set from a texture view. Intended when the texture comes
        from elsewhere. Make sure it matches the format!
        """
        self.texture = None
        self.texture_view = texture_view
        self.size = self.texture_view.size

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

    @property
    def bytes_per_pixel(self):
        """The number of bytes per pixel."""
        format_map = {
            "depth24plus_stencil8": 4,
            "depth24plus": 3,  # ?
            "depth32": 4,
            "r8": 1,
            "r16": 2,
            "r32": 4,
            "rg8": 2,
            "rg16": 4,
            "rg32": 8,
            "rgba8": 4,
            "rgba16": 8,
            "rgba32": 16,
        }
        for key, val in format_map.items():
            if self.format.startswith(key):
                return val
        else:
            raise ValueError(f"Could not determine bbp of {self.format}")


class PostProcessingStep:
    """
    Utility object to render one texture to another, as a post-processing step.
    """

    def __init__(self, device, *, vertex_shader=None, fragment_shader, uniform_type):
        self._device = device
        self._vertex_shader = (
            default_vertex_shader if vertex_shader is None else vertex_shader
        )
        self._fragment_shader = fragment_shader
        self._hash = None

        self._uniform_type = uniform_type
        self._uniform_data = array_from_shadertype(self._uniform_type)
        self._uniform_buffer = self._device.create_buffer(
            size=self._uniform_data.nbytes,
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
        )

        self._sampler = self._device.create_sampler(
            label="render sampler",
            mag_filter="nearest",
            min_filter="nearest",
        )

    def render(self, src, dst):
        """Render one texture to another. If the src.texture_view and the dst.format
        have not changed, the pipeline is re-used.
        """
        assert isinstance(src, RenderTexture)
        assert isinstance(dst, RenderTexture)
        device = self._device

        # Recreate pipeline?
        hash = src.size, id(src.texture_view), dst.format
        if hash != self._hash:
            self._hash = hash
            self._create_pipeline(
                self._vertex_shader, self._fragment_shader, src.texture_view, dst.format
            )

        command_encoder = device.create_command_encoder()

        # Copy data to tmp buffer
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
                    "view": dst.texture_view,
                    "resolve_target": None,
                    "load_value": (0, 0, 0, 0),  # LoadOp.load or color
                    "store_op": wgpu.StoreOp.store,
                }
            ],
            depth_stencil_attachment=None,
            occlusion_query_set=None,
        )
        render_pass.set_pipeline(self._render_pipeline)
        render_pass.set_bind_group(0, self._bind_group, [], 0, 99)
        render_pass.draw(4, 1)
        render_pass.end_pass()
        device.queue.submit([command_encoder.finish()])

    def _create_pipeline(
        self, vertex_shader, fragment_shader, src_texture_view, dst_format
    ):
        device = self._device

        vs_module = device.create_shader_module(code=vertex_shader)
        fs_module = device.create_shader_module(code=fragment_shader)

        binding_layouts = [
            {
                "binding": 0,
                "visibility": wgpu.ShaderStage.FRAGMENT,
                "sampler": {"type": wgpu.SamplerBindingType.filtering},
            },
            {
                "binding": 1,
                "visibility": wgpu.ShaderStage.FRAGMENT,
                "texture": {
                    "sample_type": wgpu.TextureSampleType.float,
                    "view_dimension": wgpu.TextureViewDimension.d2,
                    "multisampled": False,
                },
            },
            {
                "binding": 2,
                "visibility": wgpu.ShaderStage.FRAGMENT,
                "buffer": {"type": wgpu.BufferBindingType.uniform},
            },
        ]
        bindings = [
            {"binding": 0, "resource": self._sampler},
            {"binding": 1, "resource": src_texture_view},
            {
                "binding": 2,
                "resource": {
                    "buffer": self._uniform_buffer,
                    "offset": 0,
                    "size": self._uniform_data.nbytes,
                },
            },
        ]

        bind_group_layout = device.create_bind_group_layout(entries=binding_layouts)
        pipeline_layout = device.create_pipeline_layout(
            bind_group_layouts=[bind_group_layout]
        )
        self._bind_group = device.create_bind_group(
            layout=bind_group_layout, entries=bindings
        )

        self._render_pipeline = device.create_render_pipeline(
            layout=pipeline_layout,
            vertex={
                "module": vs_module,
                "entry_point": "main",
                "buffers": [],
            },
            primitive={
                "topology": wgpu.PrimitiveTopology.triangle_strip,
                "strip_index_format": wgpu.IndexFormat.uint32,
            },
            depth_stencil=None,
            multisample=None,
            fragment={
                "module": fs_module,
                "entry_point": "main",
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


class SSAAPostProcessingStep(PostProcessingStep):
    """A texture renderer that supports SSAA."""

    def __init__(self, device):
        super().__init__(
            device, fragment_shader=ssaa_fragment_shader, uniform_type=ssaa_uniform_type
        )

    def render(self, src, dst):

        # Get factor between texture sizes
        factor_x = src.size[0] / dst.size[0]
        factor_y = src.size[1] / dst.size[1]
        factor = (factor_x + factor_y) / 2

        if factor > 1:
            # src has higher res, we can do ssaa
            sigma = 0.5 * factor
            support = min(5, int(sigma * 3))
        else:
            # src has lower res. smooth those pixels a bit
            sigma = 1
            support = 2

        self._uniform_data["size"] = src.size[:2]
        self._uniform_data["sigma"] = sigma
        self._uniform_data["support"] = support

        super().render(src, dst)


# %% Shaders

ssaa_uniform_type = dict(
    size=("float32", 2),
    sigma=("float32",),
    support=("int32",),
)


default_vertex_shader = """

struct VertexOutput {
    [[location(0)]] texcoord: vec2<f32>;
    [[builtin(position)]] pos: vec4<f32>;
};

[[stage(vertex)]]
fn main([[builtin(vertex_index)]] index: u32) -> VertexOutput {
    var positions = array<vec2<f32>, 4>(vec2<f32>(0.0, 1.0), vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 1.0), vec2<f32>(1.0, 0.0));
    let pos = positions[index];
    var out: VertexOutput;
    out.texcoord = vec2<f32>(pos.x, 1.0 - pos.y);
    out.pos = vec4<f32>(pos * 2.0 - 1.0, 0.0, 1.0);
    return out;
}

"""

ssaa_fragment_shader = """
struct VertexOutput {
    [[location(0)]] texcoord: vec2<f32>;
    [[builtin(position)]] pos: vec4<f32>;
};

[[block]]
struct Render {
    size: vec2<f32>;
    sigma: f32;
    support: i32;
};

[[group(0), binding(0)]]
var r_sampler: sampler;
[[group(0), binding(1)]]
var r_tex: texture_2d<f32>;
[[group(0), binding(2)]]
var u_render: Render;

[[stage(fragment)]]
fn main(in: VertexOutput) -> [[location(0)]] vec4<f32> {
    // Get info about the smoothing
    let sigma = u_render.sigma;
    let support = min(5, u_render.support);

    // Determine distance between pixels in src texture
    let stepp = vec2<f32>(1.0 / u_render.size.x, 1.0 / u_render.size.y);
    // Get texcoord, and round it to the center of the source pixels.
    // Thus, whether the sampler is linear or nearest, we get equal results.
    let ori_coord = in.texcoord.xy;
    let ref_coord = vec2<f32>(vec2<i32>(ori_coord / stepp)) * stepp + 0.5 * stepp;

    // Convolve. Here we apply a Gaussian kernel, the weight is calculated
    // for each pixel individually based on the distance to the actual texture
    // coordinate. This means that the textures don't even need to align.
    var val: vec4<f32> = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    var weight: f32 = 0.0;
    for (var y:i32 = -support; y <= support; y = y + 1) {
        for (var x:i32 = -support; x <= support; x = x + 1) {
            let coord = ref_coord + vec2<f32>(f32(x), f32(y)) * stepp;
            let dist = length((ori_coord - coord) / stepp);  // in src pixels
            let t = dist / sigma;
            let w = exp(-0.5 * t * t);
            val = val + textureSample(r_tex, r_sampler, coord) * w;
            weight = weight + w;
        }
    }
    return val / weight;
}
"""
