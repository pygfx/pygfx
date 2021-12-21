import wgpu
import numpy as np


FULL_QUAD_SHADER = """
    struct VertexInput {
        [[builtin(vertex_index)]] index: u32;
    };
    struct Varyings {
        [[location(0)]] texcoord: vec2<f32>;
        [[builtin(position)]] position: vec4<f32>;
    };
    struct FragmentOutput {
        [[location(0)]] color: vec4<f32>;
    };

    BINDINGS_CODE

    [[stage(vertex)]]
    fn vs_main(in: VertexInput) -> Varyings {
        var positions = array<vec2<f32>,4>(
            vec2<f32>(0.0, 1.0), vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 1.0), vec2<f32>(1.0, 0.0)
        );
        let pos = positions[in.index];
        var varyings: Varyings;
        varyings.texcoord = vec2<f32>(pos.x, 1.0 - pos.y);
        varyings.position = vec4<f32>(pos * 2.0 - 1.0, 0.0, 1.0);
        return varyings;
    }

    [[stage(fragment)]]
    fn fs_main(varyings: Varyings) -> FragmentOutput {
        var out : FragmentOutput;
        let texcoord = varyings.texcoord;  // for textureSample
        let texindex = vec2<i32>(varyings.position.xy);  // for textureLoad

        FRAGMENT_CODE

        return out;
    }
"""


def _create_pipeline(device, binding_layouts, bindings, targets, wgsl):

    shader_module = device.create_shader_module(code=wgsl)

    bind_group_layout = device.create_bind_group_layout(entries=binding_layouts)
    pipeline_layout = device.create_pipeline_layout(
        bind_group_layouts=[bind_group_layout]
    )
    bind_group = device.create_bind_group(layout=bind_group_layout, entries=bindings)

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
            "targets": targets,
        },
    )

    return bind_group, render_pipeline


# %%%%%%%%%%


class RenderFlusher:
    """
    Utility to flush (render) the current state of a renderer into a texture.
    """

    # todo: Once we also have the depth here, we can support things like fog

    def __init__(self, device):
        self._device = device
        self._pipelines = {}

        dtype = [("size", "float32", (2,)), ("sigma", "float32"), ("support", "int32")]
        self._uniform_data = np.zeros((), dtype=dtype)
        self._uniform_buffer = self._device.create_buffer(
            size=self._uniform_data.nbytes,
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
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
        return self._render(dst_color_tex, dst_format)

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
        )
        render_pass.set_pipeline(render_pipeline)
        render_pass.set_bind_group(0, bind_group, [], 0, 99)
        render_pass.draw(4, 1)
        render_pass.end_pass()

        return [command_encoder.finish()]

    def _create_pipeline(self, src_texture_view, dst_format):
        device = self._device

        bindings_code = """
            struct Render {
                size: vec2<f32>;
                sigma: f32;
                support: i32;
            };
            [[group(0), binding(0)]]
            var<uniform> u_render: Render;
            [[group(0), binding(1)]]
            var r_color: texture_2d<f32>;
        """

        fragment_code = """
            // Get info about the smoothing
            let sigma = u_render.sigma;
            let support = min(5, u_render.support);

            // The reference index is the subpixel index in the source texture that
            // represents the location of this fragment.
            let ref_index = texcoord * u_render.size;

            // For the sampling, we work with integer coords. Also use min/max for the edges.
            let base_index = vec2<i32>(ref_index);
            let min_index = vec2<i32>(0, 0);
            let max_index = vec2<i32>(u_render.size - 1.0);

            // Convolve. Here we apply a Gaussian kernel, the weight is calculated
            // for each pixel individually based on the distance to the ref_index.
            // This means that the textures don't need to align.
            var val: vec4<f32> = vec4<f32>(0.0, 0.0, 0.0, 0.0);
            var weight: f32 = 0.0;
            for (var y:i32 = -support; y <= support; y = y + 1) {
                for (var x:i32 = -support; x <= support; x = x + 1) {
                    let step = vec2<i32>(x, y);
                    let index = clamp(base_index + step, min_index, max_index);
                    let dist = length(ref_index - vec2<f32>(index) - 0.5);
                    let t = dist / sigma;
                    let w = exp(-0.5 * t * t);
                    val = val + textureLoad(r_color, index, 0) * w;
                    weight = weight + w;
                }
            }
            out.color = val / weight;
        """

        wgsl = FULL_QUAD_SHADER
        wgsl = wgsl.replace("BINDINGS_CODE", bindings_code)
        wgsl = wgsl.replace("FRAGMENT_CODE", fragment_code)

        binding_layouts = [
            {
                "binding": 0,
                "visibility": wgpu.ShaderStage.FRAGMENT,
                "buffer": {"type": wgpu.BufferBindingType.uniform},
            },
            {
                "binding": 1,
                "visibility": wgpu.ShaderStage.FRAGMENT,
                "texture": {
                    "sample_type": wgpu.TextureSampleType.unfilterable_float,
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
            {"binding": 1, "resource": src_texture_view},
        ]

        targets = [
            {
                "format": dst_format,
                "blend": {
                    "alpha": (
                        wgpu.BlendFactor.src_alpha,
                        wgpu.BlendFactor.one_minus_src_alpha,
                        wgpu.BlendOperation.add,
                    ),
                    "color": (
                        wgpu.BlendFactor.src_alpha,
                        wgpu.BlendFactor.one_minus_src_alpha,
                        wgpu.BlendOperation.add,
                    ),
                },
            },
        ]

        return _create_pipeline(device, binding_layouts, bindings, targets, wgsl)
