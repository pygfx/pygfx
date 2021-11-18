import wgpu  # only for flags/enums

from ...utils import array_from_shadertype
from ._shadercomposer import Binding, BaseShader


class BaseFragmentBlender:
    """Manage how fragments are blended and end up in the final target.
    All aspects of blending are defined on this class. Subclasses implement
    a specific blend mode. Each renderer has one blender object.
    """

    def __init__(self):

        # The size (2D in pixels) of the frame textures.
        self.size = (0, 0)
        self.msaa = 1

        # The color texture is rgba8 unorm - not srgb, that's only for the last step.
        self.color_format = wgpu.TextureFormat.rgba8unorm
        # The depth buffer is 32 bit - we need that precision.
        self.depth_format = wgpu.TextureFormat.depth32float
        # The pick texture has 4 channels: object id, and then 3 more, e.g.
        # the instance nr, vertex nr and weights.
        self.pick_format = wgpu.TextureFormat.rgba32sint

    def ensure_target_size(self, device, size):
        """If necessary, resize render buffers/textures to match the target size."""

        # todo: I feel that the logic to contain and resize the render targets may be better off in a separate class.
        # I'll look at this in step 2 of OIT, when we get render targets specific to a blend mode.
        # Then, this class almost static ...

        assert len(size) == 2
        size = size[0], size[1]
        if size == self.size:
            return

        self.size = size
        tex_size = size + (1,)
        usg = wgpu.TextureUsage

        for tex_name, usage in [
            ("color", usg.RENDER_ATTACHMENT | usg.COPY_SRC | usg.TEXTURE_BINDING),
            ("depth", usg.RENDER_ATTACHMENT | usg.COPY_SRC),
            ("pick", usg.RENDER_ATTACHMENT | usg.COPY_SRC),
        ]:
            format = getattr(self, tex_name + "_format")
            texture = device.create_texture(
                size=tex_size, usage=usage, dimension="2d", format=format
            )
            setattr(self, tex_name + "_tex", texture)
            setattr(self, tex_name + "_view", texture.create_view())

    def get_pipeline_targets1(self):
        """Get the list of fragment targets for device.create_render_pipeline(),
        for the first render pass.
        """
        return [
            {
                "format": self.color_format,
                "blend": {
                    "alpha": (
                        wgpu.BlendFactor.one,
                        wgpu.BlendFactor.zero,
                        wgpu.BlendOperation.add,
                    ),
                    "color": (
                        wgpu.BlendFactor.one,
                        wgpu.BlendFactor.zero,
                        wgpu.BlendOperation.add,
                    ),
                },
                "write_mask": wgpu.ColorWrite.ALL,
            },
            {
                "format": self.pick_format,
                "blend": None,
                "write_mask": wgpu.ColorWrite.ALL,
            },
        ]

    def get_color_attachments1(self, clear_color):
        """Get the list of color_attachments for command_encoder.begin_render_pass(),
        for the first render pass.
        """
        # Unlike most methods, this affects the rendering, but not the wobject's pipeline.
        if clear_color:
            color_load_value = 0, 0, 0, 0
            pick_load_value = 0, 0, 0, 0
        else:
            color_load_value = wgpu.LoadOp.load
            pick_load_value = wgpu.LoadOp.load

        return [
            {
                "view": self.color_view,
                "resolve_target": None,
                "load_value": color_load_value,
                "store_op": wgpu.StoreOp.store,
            },
            {
                "view": self.pick_view,
                "resolve_target": None,
                "load_value": pick_load_value,
                "store_op": wgpu.StoreOp.store,
            },
        ]

    def get_pipeline_targets2(self):
        """Get the list of fragment targets for device.create_render_pipeline(),
        for the second render pass.
        """
        return []  # Returning [] prevents this pipeline from being created

    def get_color_attachments2(self, clear_color):
        """Get the list of color_attachments for command_encoder.begin_render_pass(),
        for the second render pass.
        """
        # Unlike most methods, this affects the rendering, but not the wobject's pipeline.
        return []  # Returning [] prevents this pass from running

    def get_shader_code(self):
        """Get the shader code to include in the shaders.

        Notes:

        * This code gets injected into the shader, so the material shaders
          can use add_fragment and finalize_fragment.
        * This code should define:
          * FragmentOutput2
          * add_fragment1, add_fragment2
          * finalize_fragment1, finalize_fragment2
        * That <private> means its a mutable global, scoped to the current thread.
          Like a normal `var`, but sharable across functions.
        """

        # Default is very minimalistic, no blending nor depth check, just a stub.
        return """
        var<private> p_fragment_color : vec4<f32> = vec4<f32>(0.0, 0.0, 0.0, 0.0);

        $$ if render_pass == 1

            struct FragmentOutput {
                [[location(0)]] color: vec4<f32>;
                [[location(1)]] pick: vec4<i32>;
            };
            fn add_fragment(depth: f32, color: vec4<f32>) {
                p_fragment_color = color;
            }
            fn finalize_fragment() -> FragmentOutput {
                var out : FragmentOutput;
                out.color = p_fragment_color;
                return out;
            }

        $$ endif
        """


class OpaqueFragmentBlender(BaseFragmentBlender):
    """A fragment blender that pretends that all surfaces are opaque,
    even if they're not.
    """

    def get_shader_code(self):
        return """
        var<private> p_fragment_color : vec4<f32> = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        var<private> p_fragment_depth : f32 = 1.1;

        $$ if render_pass == 1

            struct FragmentOutput {
                [[location(0)]] color: vec4<f32>;
                [[location(1)]] pick: vec4<i32>;
            };
            fn add_fragment(depth: f32, color: vec4<f32>) {
                if (depth < p_fragment_depth) {
                    p_fragment_color = color;
                    p_fragment_depth = depth;
                }
            }
            fn finalize_fragment() -> FragmentOutput {
                if (p_fragment_depth > 1.0) { discard; }
                var out : FragmentOutput;
                out.color = vec4<f32>(p_fragment_color.rgb, 1.0);  // opaque
                return out;
            }

        $$ endif
        """


class Simple1FragmentBlender(BaseFragmentBlender):
    """A minimal fragment blender that does the minimal approach to blending:
    use the OVER blend operation, without differentiating between opaque and
    transparent objects. This is a common approach for applications using a single pass.
    """

    def get_pipeline_targets1(self):
        return [
            {
                "format": self.color_format,
                "blend": {
                    "alpha": (
                        wgpu.BlendFactor.one,
                        wgpu.BlendFactor.one_minus_src_alpha,
                        wgpu.BlendOperation.add,
                    ),
                    "color": (
                        wgpu.BlendFactor.src_alpha,
                        wgpu.BlendFactor.one_minus_src_alpha,
                        wgpu.BlendOperation.add,
                    ),
                },
                "write_mask": wgpu.ColorWrite.ALL,
            },
            {
                "format": self.pick_format,
                "blend": None,
                "write_mask": wgpu.ColorWrite.ALL,
            },
        ]

    def get_shader_code(self):
        return """
        var<private> p_fragment_color : vec4<f32> = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        var<private> p_fragment_depth : f32 = 1.1;

        $$ if render_pass == 1

            struct FragmentOutput {
                [[location(0)]] color: vec4<f32>;
                [[location(1)]] pick: vec4<i32>;
            };
            fn add_fragment(depth: f32, color: vec4<f32>) {
                if (depth < p_fragment_depth) {
                    p_fragment_color = color;
                    p_fragment_depth = depth;
                }
            }
            fn finalize_fragment() -> FragmentOutput {
                if (p_fragment_depth > 1.0) { discard; }
                var out : FragmentOutput;
                out.color = p_fragment_color;
                return out;
            }

        $$ endif
        """


class Simple2FragmentBlender(BaseFragmentBlender):
    """A first step towards better blending: separating the opaque
    from the transparent fragments, and blending the latter using the
    OVER operator. The second step has depth-testing, but no depth-writing.
    Not order-independent.
    """

    def get_pipeline_targets2(self):
        return [
            {
                "format": self.color_format,
                "blend": {
                    "alpha": (
                        wgpu.BlendFactor.one,
                        wgpu.BlendFactor.one_minus_src_alpha,
                        wgpu.BlendOperation.add,
                    ),
                    "color": (
                        wgpu.BlendFactor.one,
                        wgpu.BlendFactor.one_minus_src_alpha,
                        wgpu.BlendOperation.add,
                    ),
                },
                "write_mask": wgpu.ColorWrite.ALL,
            },
        ]

    def get_color_attachments2(self, clear_color):
        return [
            {
                "view": self.color_view,
                "resolve_target": None,
                "load_value": wgpu.LoadOp.load,
                "store_op": wgpu.StoreOp.store,
            },
        ]

    def get_shader_code(self):
        return """
        var<private> p_fragment_color : vec4<f32> = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        var<private> p_fragment_depth : f32 = 1.1;

        $$ if render_pass == 1

            struct FragmentOutput {
                [[location(0)]] color: vec4<f32>;
                [[location(1)]] pick: vec4<i32>;
            };
            fn add_fragment(depth: f32, color: vec4<f32>) {
                if (color.a >= 1.0 && depth < p_fragment_depth) {
                    p_fragment_color = color;
                    p_fragment_depth = depth;
                }
            }
            fn finalize_fragment() -> FragmentOutput {
                if (p_fragment_depth > 1.0) { discard; }
                var out : FragmentOutput;
                out.color = vec4<f32>(p_fragment_color.rgb, 1.0);
                return out;
            }

        $$ elif render_pass == 2

            struct FragmentOutput {
                [[location(0)]] color: vec4<f32>;
            };
            fn add_fragment(depth: f32, color: vec4<f32>) {
                let rgb = (1.0 - color.a) * p_fragment_color.rgb + color.a * color.rgb;
                let a = (1.0 - color.a) * p_fragment_color.a + color.a;
                p_fragment_color = vec4<f32>(rgb, a);
            }
            fn finalize_fragment() -> FragmentOutput {
                if (p_fragment_color.a <= 0.0) { discard; }
                var out : FragmentOutput;
                out.color = p_fragment_color;
                return out;
            }

        $$ endif
        """


# TODO: I think that the blender will always write the end-result back into the color texture
# which means that the flush step is unaffected. In that case we should move the code elsewhere.
# There is a small chance that the flush becomes part of the blend resolve step though.


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
            [[builtin(position)]] position: vec4<f32>;
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
            out.position = vec4<f32>(pos * 2.0 - 1.0, 0.0, 1.0);
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
        size="2xf4",
        sigma="f4",
        support="i4",
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
        shader.define_binding(
            0, 0, Binding("u_render", "buffer/uniform", self._uniform_data.dtype)
        )
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
