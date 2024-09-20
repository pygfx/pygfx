"""
The flusher is responsible for rendering the renderers internal image
to the canvas.
"""

import wgpu
import numpy as np

from .utils import GpuCache, hash_from_value
from .shared import get_shared

# This cache enables sharing some gpu objects between code that uses
# full-quad shaders. The gain here won't be large in general, but can
# still be worthwhile in situations where many canvases are created,
# such as in Jupyter notebooks, or e.g. when generating screenshots.
# It should also reduce the required work during canvas resizing.
FULL_QUAD_CACHE = GpuCache("full_quad_objects")


FULL_QUAD_SHADER = """
    struct VertexInput {
        @builtin(vertex_index) index: u32,
    };
    struct Varyings {
        @location(0) texcoord: vec2<f32>,
        @builtin(position) position: vec4<f32>,
    };
    struct FragmentOutput {
        @location(0) color: vec4<f32>,
    };

    BINDINGS_CODE

    @vertex
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

    @fragment
    fn fs_main(varyings: Varyings) -> FragmentOutput {
        var out : FragmentOutput;
        let texcoord = varyings.texcoord;  // for textureSample
        let texindex = vec2<i32>(varyings.position.xy);  // for textureLoad

        FRAGMENT_CODE

        return out;
    }
"""


def create_full_quad_pipeline(targets, binding_layout, bindings_code, fragment_code):
    device = get_shared().device

    # Get bind group layout
    key1 = hash_from_value(binding_layout)
    bind_group_layout = FULL_QUAD_CACHE.get(key1)
    if bind_group_layout is None:
        bind_group_layout = device.create_bind_group_layout(entries=binding_layout)
        FULL_QUAD_CACHE.set(key1, bind_group_layout)

    # Get render pipeline
    key2 = hash_from_value([bind_group_layout, targets, bindings_code, fragment_code])
    render_pipeline = FULL_QUAD_CACHE.get(key2)
    if render_pipeline is None:
        wgsl = FULL_QUAD_SHADER
        wgsl = wgsl.replace("BINDINGS_CODE", bindings_code)
        wgsl = wgsl.replace("FRAGMENT_CODE", fragment_code)
        shader_module = device.create_shader_module(code=wgsl)

        pipeline_layout = device.create_pipeline_layout(
            bind_group_layouts=[bind_group_layout]
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
                "targets": targets,
            },
        )

        # Bind shader module object to the lifetime of the pipeline object
        render_pipeline._gfx_module = shader_module

        FULL_QUAD_CACHE.set(key2, render_pipeline)

    return render_pipeline


# %%%%%%%%%%


class RenderFlusher:
    """
    Utility to flush (render) the current state of a renderer into a texture.
    """

    def __init__(self, target_format):
        self._device = get_shared().device
        self._target_format = target_format

        dtype = [
            ("size", "float32", (2,)),
            ("sigma", "float32"),
            ("support", "int32"),
            ("gamma", "float32"),
            ("_padding", "float32"),
        ]
        self._uniform_data = np.zeros((), dtype=dtype)
        self._uniform_buffer = self._device.create_buffer(
            size=self._uniform_data.nbytes,
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
        )

        self._render_pipeline = None
        self._bind_group = None
        self._bind_group_hash = None

    def render(
        self,
        src_color_tex,
        src_depth_tex,
        dst_color_tex,
        gamma=1.0,
        filter_strength=1.0,
    ):
        """Render the (internal) result of the renderer to a texture view."""

        # NOTE: src_depth_tex is not used yet, see #492
        # NOTE: cannot actually use src_depth_tex as a sample texture (BindingCollision)?
        assert src_depth_tex is None
        assert isinstance(src_color_tex, wgpu.GPUTextureView)
        assert isinstance(dst_color_tex, wgpu.GPUTextureView)

        # Make sure we have the render_pipeline
        if self._render_pipeline is None:
            self._render_pipeline = self._create_pipeline()

        # Same for bind group. Needs to be recreated when the source texture changes.
        hash = id(src_color_tex._internal)
        if self._bind_group is None or hash != self._bind_group_hash:
            self._bind_group_hash = hash
            self._bind_group = self._create_bind_group(
                self._render_pipeline.get_bind_group_layout(0), src_color_tex
            )

        # Ready to go!
        self._update_uniforms(src_color_tex, dst_color_tex, gamma, filter_strength)
        return self._render(dst_color_tex)

    def _update_uniforms(self, src_color_tex, dst_color_tex, gamma, filter_strength):
        # Get factor between texture sizes
        factor_x = src_color_tex.size[0] / dst_color_tex.size[0]
        factor_y = src_color_tex.size[1] / dst_color_tex.size[1]
        factor = (factor_x + factor_y) / 2

        if factor == 1:
            # With equal res, we smooth a tiny bit. A bit less crisp, but also less flicker.
            ref_sigma = 0.5
        elif factor > 1:
            # With src a higher res, we will do SSAA.
            ref_sigma = 0.5 * factor
        else:
            # With src a lower res, the output is interpolated. But we also smooth to reduce blockiness.
            ref_sigma = 0.5

        # Determine kernel sigma and support
        sigma = ref_sigma * float(filter_strength)
        support = int(sigma * 3)  # is limited in shader

        # Compose
        self._uniform_data["size"] = src_color_tex.size[:2]
        self._uniform_data["sigma"] = sigma
        self._uniform_data["support"] = support
        self._uniform_data["gamma"] = gamma

        # Sync to gpu
        self._device.queue.write_buffer(
            self._uniform_buffer, 0, self._uniform_data, 0, self._uniform_data.nbytes
        )

    def _render(self, dst_color_tex):
        command_encoder = self._device.create_command_encoder()

        render_pass = command_encoder.begin_render_pass(
            color_attachments=[
                {
                    "view": dst_color_tex,
                    "resolve_target": None,
                    "clear_value": (0, 0, 0, 0),
                    "load_op": wgpu.LoadOp.clear,
                    "store_op": wgpu.StoreOp.store,
                }
            ],
            depth_stencil_attachment=None,
        )
        render_pass.set_pipeline(self._render_pipeline)
        render_pass.set_bind_group(0, self._bind_group, [], 0, 99)
        render_pass.draw(4, 1)
        render_pass.end()

        return [command_encoder.finish()]

    def _create_pipeline(self):
        bindings_code = """
            struct Render {
                size: vec2<f32>,
                sigma: f32,
                support: i32,
                gamma: f32,
            };
            @group(0) @binding(0)
            var<uniform> u_render: Render;
            @group(0) @binding(1)
            var r_color: texture_2d<f32>;
        """

        fragment_code = """
            // Get info about the smoothing
            // The limits here may give the compiler info on max iters of the loop below.
            let sigma = max(0.1, u_render.sigma);
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
            let gamma3 = vec3<f32>(u_render.gamma);
            out.color = vec4<f32>(pow(val.rgb / weight, gamma3), val.a / weight);

            // Note that the final opacity is not necessarily one. This means that
            // the framebuffer can be blended with the background, or one can render
            // images that can be better combined with other content in a document.
            // It also means that most examples benefit from a gfx.Background.
        """

        binding_layout = [
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

        targets = [
            {
                "format": self._target_format,
                "blend": {
                    "alpha": {
                        "operation": wgpu.BlendOperation.add,
                        "src_factor": wgpu.BlendFactor.src_alpha,
                        "dst_factor": wgpu.BlendFactor.one_minus_src_alpha,
                    },
                    "color": {
                        "operation": wgpu.BlendOperation.add,
                        "src_factor": wgpu.BlendFactor.src_alpha,
                        "dst_factor": wgpu.BlendFactor.one_minus_src_alpha,
                    },
                },
            },
        ]

        return create_full_quad_pipeline(
            targets, binding_layout, bindings_code, fragment_code
        )

    def _create_bind_group(self, bind_group_layout, src_texture_view):
        # This must match the binding_layout above
        bind_group_entries = [
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
        return self._device.create_bind_group(
            layout=bind_group_layout, entries=bind_group_entries
        )
