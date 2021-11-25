import wgpu  # only for flags/enums

from ._flusher import FULL_QUAD_SHADER, _create_pipeline


# Notes:
# - The user code provides color as-is in rgba.
# - In the add_fragment logic defined in the shaders here, the color
#   is pre-multiplied with the alpha.
# - All fixed-pipeling blending option assume premultiplied alpha.


standard_texture_des = {
    "sample_type": wgpu.TextureSampleType.float,
    "view_dimension": wgpu.TextureViewDimension.d2,
    "multisampled": False,
}


# %%%%%%%%%%  Define passes


class BasePass:
    """The base pass class, defining and documenting the API that a pass must provide."""

    depth_write_enabled = True

    def get_pipeline_targets(self, blender):
        """Get the list of fragment targets for device.create_render_pipeline()."""
        # The result affects the wobject's pipeline.
        # Returning [] prevents this pipeline from being created.
        return []

    def get_color_attachments(self, blender, clear):
        """Get the list of color_attachments for command_encoder.begin_render_pass()."""
        # The result affects the rendering, but not the wobject's pipeline.
        # Returning [] prevents this pass from running.
        return []

    def get_shader_code(self, blender):
        """Get the fragment-write shader code.

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
        return ""


class OpaquePass(BasePass):
    """A pass that renders opaque fragments with depth testing, while
    discarting transparent fragments. This functions as the first pass
    in all multi-pass blenders.
    """

    depth_write_enabled = True

    def get_pipeline_targets(self, blender):
        BF, BO = wgpu.BlendFactor, wgpu.BlendOperation
        return [
            {
                "format": blender.color_format,
                "blend": {
                    "alpha": (BF.one, BF.zero, BO.add),
                    "color": (BF.one, BF.zero, BO.add),
                },
                "write_mask": wgpu.ColorWrite.ALL,
            },
            {
                "format": blender.pick_format,
                "blend": None,
                "write_mask": wgpu.ColorWrite.ALL,
            },
        ]

    def get_color_attachments(self, blender, clear):
        if clear:
            color_load_value = 0, 0, 0, 0
            pick_load_value = 0, 0, 0, 0
        else:
            color_load_value = wgpu.LoadOp.load
            pick_load_value = wgpu.LoadOp.load

        return [
            {
                "view": blender.color_view,
                "resolve_target": None,
                "load_value": color_load_value,
                "store_op": wgpu.StoreOp.store,
            },
            {
                "view": blender.pick_view,
                "resolve_target": None,
                "load_value": pick_load_value,
                "store_op": wgpu.StoreOp.store,
            },
        ]

    def get_shader_code(self, blender):
        return """
        var<private> p_fragment_color : vec4<f32> = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        var<private> p_fragment_depth : f32 = 1.1;

        struct FragmentOutput {
            [[location(0)]] color: vec4<f32>;
            [[location(1)]] pick: vec4<i32>;
        };
        fn add_fragment(depth: f32, color: vec4<f32>) {
            if (color.a >= 1.0 && depth < p_fragment_depth) {
                p_fragment_color = vec4<f32>(color.rgb, 1.0);
                p_fragment_depth = depth;
            }
        }
        fn finalize_fragment() -> FragmentOutput {
            if (p_fragment_depth > 1.0) { discard; }
            var out : FragmentOutput;
            out.color = p_fragment_color;
            return out;
        }
        """


class FullOpaquePass(OpaquePass):
    """A pass that considers all fragments opaque."""

    depth_write_enabled = True

    def get_shader_code(self, blender):
        return """
        var<private> p_fragment_color : vec4<f32> = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        var<private> p_fragment_depth : f32 = 1.1;

        struct FragmentOutput {
            [[location(0)]] color: vec4<f32>;
            [[location(1)]] pick: vec4<i32>;
        };
        fn add_fragment(depth: f32, color: vec4<f32>) {
            if (depth < p_fragment_depth) {
                p_fragment_color = vec4<f32>(color.rgb, 1.0);  // always opaque
                p_fragment_depth = depth;
            }
        }
        fn finalize_fragment() -> FragmentOutput {
            if (p_fragment_depth > 1.0) { discard; }
            var out : FragmentOutput;
            out.color = p_fragment_color;
            return out;
        }
        """


class SimpleSinglePass(OpaquePass):
    """A pass that blends opaque and transparent fragments in a single pass."""

    depth_write_enabled = True

    def get_pipeline_targets(self, blender):
        BF, BO = wgpu.BlendFactor, wgpu.BlendOperation
        return [
            {
                "format": blender.color_format,
                "blend": {
                    "alpha": (BF.one, BF.one_minus_src_alpha, BO.add),
                    "color": (BF.one, BF.one_minus_src_alpha, BO.add),
                },
                "write_mask": wgpu.ColorWrite.ALL,
            },
            {
                "format": blender.pick_format,
                "blend": None,
                "write_mask": wgpu.ColorWrite.ALL,
            },
        ]

    def get_shader_code(self, blender):
        # Take depth into account, but don't treat transparent fragments differently
        return """
        var<private> p_fragment_color : vec4<f32> = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        var<private> p_fragment_depth : f32 = 1.1;

        struct FragmentOutput {
            [[location(0)]] color: vec4<f32>;
            [[location(1)]] pick: vec4<i32>;
        };
        fn add_fragment(depth: f32, color: vec4<f32>) {
            if (depth < p_fragment_depth) {
                p_fragment_color = vec4<f32>(color.rgb * color.a, color.a);
                p_fragment_depth = depth;
            }
        }
        fn finalize_fragment() -> FragmentOutput {
            if (p_fragment_depth > 1.0) { discard; }
            var out : FragmentOutput;
            out.color = p_fragment_color;
            return out;
        }
        """


class SimpleTransparencyPass(BasePass):
    """A pass that only renders transparent fragments, blending them
    with the classic recursive alpha blending equation (a.k.a. the OVER
    operator).
    """

    depth_write_enabled = False

    def get_pipeline_targets(self, blender):
        BF, BO = wgpu.BlendFactor, wgpu.BlendOperation
        return [
            {
                "format": blender.color_format,
                "blend": {
                    "alpha": (BF.one, BF.one_minus_src_alpha, BO.add),
                    "color": (BF.one, BF.one_minus_src_alpha, BO.add),
                },
                "write_mask": wgpu.ColorWrite.ALL,
            },
        ]

    def get_color_attachments(self, blender, clear):
        return [
            {
                "view": blender.color_view,
                "resolve_target": None,
                "load_value": wgpu.LoadOp.load,
                "store_op": wgpu.StoreOp.store,
            },
        ]

    def get_shader_code(self, blender):
        return """
        var<private> p_fragment_color : vec4<f32> = vec4<f32>(0.0, 0.0, 0.0, 0.0);

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
        """


class WeightedTransparencyPass(BasePass):
    """A pass that implements weighted blended order-independed
    blending for transparent fragments, as proposed by McGuire in 2013.
    Multiple weight functions are supported.
    """

    depth_write_enabled = False

    def __init__(self, weight_func):

        if weight_func == "alpha":
            weight_code = """
                let weight = alpha;
            """
        elif weight_func == "depth":
            # The "generic purpose" depth function proposed by McGuire in
            # http://casual-effects.blogspot.com/2015/03/implemented-weighted-blended-order.html
            weight_code = """
                let a = min(1.0, alpha) * 8.0 + 0.01;
                let b = (1.0 - 0.99999 * depth);
                let weight = clamp(a * a * a * 1e8 * b * b * b, 1e-2, 3e2);
            """
        else:
            raise ValueError(
                f"Unknown WeightedTransparencyPass weight_func: {weight_func!r}"
            )

        self._weight_code = weight_code.strip()

    def get_pipeline_targets(self, blender):
        BF, BO = wgpu.BlendFactor, wgpu.BlendOperation
        return [
            {
                "format": blender.accum_format,
                "blend": {
                    "alpha": (BF.one, BF.one, BO.add),
                    "color": (BF.one, BF.one, BO.add),
                },
                "write_mask": wgpu.ColorWrite.ALL,
            },
            {
                "format": blender.reveal_format,
                "blend": {
                    "alpha": (BF.zero, BF.one_minus_src_alpha, BO.add),
                    "color": (BF.zero, BF.one_minus_src, BO.add),
                },
                "write_mask": wgpu.ColorWrite.ALL,
            },
        ]

    def get_color_attachments(self, blender, clear):
        if clear:
            accum_load_value = 0, 0, 0, 0
            reveal_load_value = 1, 0, 0, 0
        else:
            accum_load_value = wgpu.LoadOp.load
            reveal_load_value = wgpu.LoadOp.load

        return [
            {
                "view": blender.accum_view,
                "resolve_target": None,
                "load_value": accum_load_value,
                "store_op": wgpu.StoreOp.store,
            },
            {
                "view": blender.reveal_view,
                "resolve_target": None,
                "load_value": reveal_load_value,
                "store_op": wgpu.StoreOp.store,
            },
        ]

    def get_shader_code(self, blender):
        return """
        var<private> p_fragment_accum : vec4<f32> = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        var<private> p_fragment_reveal : f32 = 0.0;

        struct FragmentOutput {
            [[location(0)]] accum: vec4<f32>;
            [[location(1)]] reveal: f32;
        };
        fn add_fragment(depth: f32, color: vec4<f32>) {
            let premultiplied = color.rgb * color.a;
            let alpha = color.a;  // could take user-specified transmittance into account
            WEIGHT_CODE
            p_fragment_accum = vec4<f32>(premultiplied, alpha) * weight;
            p_fragment_reveal = alpha;
        }
        fn finalize_fragment() -> FragmentOutput {
            if (p_fragment_reveal <= 0.0) { discard; }
            var out : FragmentOutput;
            out.accum = p_fragment_accum;
            out.reveal = p_fragment_reveal;
            return out;
        }
        """.replace(
            "WEIGHT_CODE", self._weight_code
        )


class FrontmostTransparencyPass(BasePass):
    """A render pass that renders the front-most transparent layer to
    a custom render target. This can then later be used in the combine-pass.
    """

    depth_write_enabled = True

    def get_pipeline_targets(self, blender):
        BF, BO = wgpu.BlendFactor, wgpu.BlendOperation
        return [
            {
                "format": blender.frontcolor_format,
                "blend": {
                    "alpha": (BF.one, BF.zero, BO.add),
                    "color": (BF.one, BF.zero, BO.add),
                },
                "write_mask": wgpu.ColorWrite.ALL,
            },
        ]

    def get_color_attachments(self, blender, clear):
        if clear:
            color_load_value = 0, 0, 0, 0
        else:
            color_load_value = wgpu.LoadOp.load

        return [
            {
                "view": blender.frontcolor_view,
                "resolve_target": None,
                "load_value": color_load_value,
                "store_op": wgpu.StoreOp.store,
            },
        ]

    def get_shader_code(self, blender):
        return """
        var<private> p_fragment_color : vec4<f32> = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        var<private> p_fragment_depth : f32 = 1.1;

        struct FragmentOutput {
            [[location(0)]] color: vec4<f32>;
            [[location(1)]] pick: vec4<i32>;
        };
        fn add_fragment(depth: f32, color: vec4<f32>) {
            if (color.a > 0.0 && color.a < 1.0) {
                p_fragment_color = vec4<f32>(color.rgb * color.a, color.a);
                p_fragment_depth = depth;
            }
        }
        fn finalize_fragment() -> FragmentOutput {
            if (p_fragment_depth > 1.0) { discard; }
            var out : FragmentOutput;
            out.color = p_fragment_color;
            return out;
        }
        """


# %%%%%%%%%% Define blenders


class BaseFragmentBlender:
    """Manage how fragments are blended and end up in the final target.
    Each renderer has one blender object.
    """

    # A list of passes
    passes = []

    def __init__(self):

        # The size (2D in pixels) of the frame textures.
        self.size = (0, 0)

        # Pipeline objects
        self._combine_pass_info = None

        # A dict that contains the metadata for all render targets.
        self._texture_info = {}

        # The below targets are always present, and the renderer expects their
        # format, texture, and view to be present.

        usg = wgpu.TextureUsage

        # The color texture is rgba8 unorm - not srgb, that's only for the last step.
        self._texture_info["color"] = (
            wgpu.TextureFormat.rgba8unorm,
            usg.RENDER_ATTACHMENT | usg.COPY_SRC | usg.TEXTURE_BINDING,
        )

        # The depth buffer is 32 bit - we need that precision.
        self._texture_info["depth"] = (
            wgpu.TextureFormat.depth32float,
            usg.RENDER_ATTACHMENT | usg.COPY_SRC,
        )

        # The pick texture has 4 channels: object id, and then 3 more, e.g.
        # the instance nr, vertex nr and weights.
        self._texture_info["pick"] = (
            wgpu.TextureFormat.rgba32sint,
            usg.RENDER_ATTACHMENT | usg.COPY_SRC,
        )

    def ensure_target_size(self, device, size):
        """If necessary, resize render-textures to match the target size."""

        assert len(size) == 2
        size = size[0], size[1]
        if size == self.size:
            return

        # Set new size
        self.size = size
        tex_size = size + (1,)

        # Any pipelines are now invalid because they include render targets.
        self._combine_pass_info = None

        # Recreate internal textures
        for name, (format, usage) in self._texture_info.items():
            texture = device.create_texture(
                size=tex_size, usage=usage, dimension="2d", format=format
            )
            setattr(self, name + "_format", format)
            setattr(self, name + "_tex", texture)
            setattr(self, name + "_view", texture.create_view())

    # The five methods below represent the API that the render system uses.

    def get_pipeline_targets(self, pass_index):
        return self.passes[pass_index].get_pipeline_targets(self)

    def get_color_attachments(self, pass_index, clear):
        return self.passes[pass_index].get_color_attachments(self, clear)

    def get_shader_code(self, pass_index):
        return self.passes[pass_index].get_shader_code(self)

    def get_depth_write_enabled(self, pass_index):
        return self.passes[pass_index].depth_write_enabled

    def iter_pass_indices(self):
        """Get an iterator that yield pass_index in the appropriate order.
        Note that pass_index 0 (zero) is reserved for the "solid" pass in which
        the pick info must be written. The index 0 is not necessarily the first
        index of this iterator.
        """
        return range(len(self.passes))

    def perform_combine_pass(self, device, command_encoder):
        """Perform a render-pass to combine any multi-pass results, if needed."""
        # Get bindgroup and pipeline
        if not self._combine_pass_info:
            self._combine_pass_info = self._create_combination_pipeline(device)
        bind_group, render_pipeline = self._combine_pass_info
        if not render_pipeline:
            return

        # Render
        render_pass = command_encoder.begin_render_pass(
            color_attachments=[
                {
                    "view": self.color_view,
                    "resolve_target": None,
                    "load_value": wgpu.LoadOp.load,
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

    def _create_combination_pipeline(self, device):
        """Overload this to setup the specific combiner-pass."""
        return None, None


class OpaqueFragmentBlender(BaseFragmentBlender):
    """A fragment blender that pretends that all surfaces are opaque,
    even if they're not.
    """

    passes = [FullOpaquePass()]


class Simple1FragmentBlender(BaseFragmentBlender):
    """A minimal fragment blender that uses the classic alpha blending
    equation, without differentiating between opaque and transparent
    objects. This is a common approach for applications using a single
    pass.
    """

    passes = [SimpleSinglePass()]


class Simple2FragmentBlender(BaseFragmentBlender):
    """A first step towards better blending: separating the opaque from
    the transparent fragments, and blending the latter using the alpha
    blending equation. The second step has depth-testing, but no
    depth-writing. Not order-independent.
    """

    passes = [OpaquePass(), SimpleTransparencyPass()]


class WeightedFragmentBlender(BaseFragmentBlender):
    """Weighted blended order independent transparency (McGuire 2013),
    using a weight function based only on alpha.

    The opaque pass is followed by as pass that accumulates the
    transparent fragments in two targets (4 channels) in a way that
    weights the fragments. These are combined in the combine-pass,
    realizing order independent blending.
    """

    passes = [OpaquePass(), WeightedTransparencyPass("alpha")]

    def __init__(self):
        super().__init__()

        usg = wgpu.TextureUsage

        # The accumulation buffer collects weighted fragments
        self._texture_info["accum"] = (
            wgpu.TextureFormat.rgba16float,
            usg.RENDER_ATTACHMENT | usg.TEXTURE_BINDING,
        )

        # The reveal buffer collects the weights
        self._texture_info["reveal"] = (
            wgpu.TextureFormat.r16float,
            usg.RENDER_ATTACHMENT | usg.TEXTURE_BINDING,
        )

    def _create_combination_pipeline(self, device):

        binding_layouts = [
            {
                "binding": 0,
                "visibility": wgpu.ShaderStage.FRAGMENT,
                "sampler": {"type": wgpu.SamplerBindingType.filtering},
            },
            {
                "binding": 1,
                "visibility": wgpu.ShaderStage.FRAGMENT,
                "texture": standard_texture_des,
            },
            {
                "binding": 2,
                "visibility": wgpu.ShaderStage.FRAGMENT,
                "texture": standard_texture_des,
            },
        ]

        sampler = device.create_sampler(mag_filter="nearest", min_filter="nearest")

        bindings = [
            {"binding": 0, "resource": sampler},
            {"binding": 1, "resource": self.accum_view},
            {"binding": 2, "resource": self.reveal_view},
        ]

        BF, BO = wgpu.BlendFactor, wgpu.BlendOperation
        targets = [
            {
                "format": self.color_format,
                "blend": {
                    "alpha": (BF.one, BF.one_minus_src_alpha, BO.add),
                    "color": (BF.one, BF.one_minus_src_alpha, BO.add),
                },
            },
        ]

        bindings_code = """
            [[group(0), binding(0)]]
            var r_sampler: sampler;
            [[group(0), binding(1)]]
            var r_accum: texture_2d<f32>;
            [[group(0), binding(2)]]
            var r_reveal: texture_2d<f32>;
        """

        fragment_code = """
            let epsilon = 0.00001;

            // Sample
            let accum = textureSample(r_accum, r_sampler, texcoord).rgba;
            let reveal = textureSample(r_reveal, r_sampler, texcoord).r;

            // Exit if no transparent fragments was written
            if (reveal >= 1.0) { discard; }  // no transparent fragments here

            // Reconstruct the color and alpha, and set final color, with premultiplied alpha
            let avg_color = accum.rgb / max(accum.a, epsilon);
            let alpha = 1.0 - reveal;
            out.color = vec4<f32>(avg_color * alpha, alpha);
        """

        wgsl = FULL_QUAD_SHADER
        wgsl = wgsl.replace("BINDINGS_CODE", bindings_code)
        wgsl = wgsl.replace("FRAGMENT_CODE", fragment_code)

        return _create_pipeline(device, binding_layouts, bindings, targets, wgsl)


class WeightedDepthFragmentBlender(WeightedFragmentBlender):
    """Weighted blended order independent transparency (McGuire 2013),
    using a general purpose depth weight function.
    """

    passes = [OpaquePass(), WeightedTransparencyPass("depth")]


class WeightedPlusFragmentBlender(WeightedFragmentBlender):
    """Three-pass weighted blended order independent transparency (McGuire 2013),
    using a depth function based on alpha, and in which the top-most
    transparent layer is actually in front.

    This uses the same approach as WeightedFragmentBlender, but in a
    3d pass we draw the front-most transparent layer. In the
    combine-pass, we subtract the front layer from the accum and reveal
    buffer, and add it again using the blend equation. In effect, the
    front-most layer is actually correct, and all transparent fragments
    behind it follow McGuire's approach. This looks a bit like a
    single-layer depth peeling.

    That 3d pass is actually drawn first, so that the final depth buffer
    state matches the opaque content of the scene. Since we have a rule
    that pass_index 0 means the opaque pass, we need to juggle the
    indices in iter_pass_indices.
    """

    passes = [
        OpaquePass(),
        WeightedTransparencyPass("alpha"),
        FrontmostTransparencyPass(),
    ]

    def __init__(self):
        super().__init__()

        usg = wgpu.TextureUsage

        # One extra color buffer for the front-most semitransparent layer
        self._texture_info["frontcolor"] = (
            wgpu.TextureFormat.rgba8unorm,
            usg.RENDER_ATTACHMENT | usg.TEXTURE_BINDING,
        )

    def iter_pass_indices(self):
        return (2, 0, 1)

    def _create_combination_pipeline(self, device):

        binding_layouts = [
            {
                "binding": 0,
                "visibility": wgpu.ShaderStage.FRAGMENT,
                "sampler": {"type": wgpu.SamplerBindingType.filtering},
            },
            {
                "binding": 1,
                "visibility": wgpu.ShaderStage.FRAGMENT,
                "texture": standard_texture_des,
            },
            {
                "binding": 2,
                "visibility": wgpu.ShaderStage.FRAGMENT,
                "texture": standard_texture_des,
            },
            {
                "binding": 3,
                "visibility": wgpu.ShaderStage.FRAGMENT,
                "texture": standard_texture_des,
            },
        ]

        sampler = device.create_sampler(mag_filter="nearest", min_filter="nearest")

        bindings = [
            {"binding": 0, "resource": sampler},
            {"binding": 1, "resource": self.accum_view},
            {"binding": 2, "resource": self.reveal_view},
            {"binding": 3, "resource": self.frontcolor_view},
        ]

        BF, BO = wgpu.BlendFactor, wgpu.BlendOperation
        targets = [
            {
                "format": self.color_format,
                "blend": {
                    "alpha": (BF.one, BF.one_minus_src_alpha, BO.add),
                    "color": (BF.one, BF.one_minus_src_alpha, BO.add),
                },
            },
        ]

        bindings_code = """
            [[group(0), binding(0)]]
            var r_sampler: sampler;
            [[group(0), binding(1)]]
            var r_accum: texture_2d<f32>;
            [[group(0), binding(2)]]
            var r_reveal: texture_2d<f32>;
            [[group(0), binding(3)]]
            var r_frontcolor: texture_2d<f32>;
        """

        fragment_code = """
            let epsilon = 0.00001;

            // Sample
            var accum = textureSample(r_accum, r_sampler, texcoord).rgba;
            var reveal = textureSample(r_reveal, r_sampler, texcoord).r;
            let front = textureSample(r_frontcolor, r_sampler, texcoord).rgba;

            // Exit if no transparent fragments was written - there also not be a front.
            if (reveal >= 1.0) { discard; }  // no transparent fragments here

            // Undo contribution of the front
            let weight = front.a;  // This must match with the other pass!
            accum = accum - front.rgba * weight;  // front is already pre-multiplied
            reveal = reveal / (1.0 - front.a);

            // Reconstruct the color and alpha, and set final color, with premultiplied alpha
            let avg_color = accum.rgb / max(accum.a, epsilon);
            let alpha = 1.0 - reveal;
            out.color = vec4<f32>(avg_color * alpha, alpha);

            // Blend in the front color (front is already premultiplied)
            let out_rgb = (1.0 - front.a) * out.color.rgb + front.rgb;
            let out_a = (1.0 - front.a) * out.color.a + front.a;
            out.color = vec4<f32>(out_rgb, out_a);
        """

        wgsl = FULL_QUAD_SHADER
        wgsl = wgsl.replace("BINDINGS_CODE", bindings_code)
        wgsl = wgsl.replace("FRAGMENT_CODE", fragment_code)

        return _create_pipeline(device, binding_layouts, bindings, targets, wgsl)
