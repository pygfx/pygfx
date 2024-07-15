"""
Defines the classes for the different render passes and the blender
objects that contain them. A blender becomes part of the environment
object.
"""

import wgpu  # only for flags/enums

from ....utils.enums import RenderMask
from .flusher import create_full_quad_pipeline
from .shared import get_shared

# Notes:
# - The user code provides color as-is in rgba.
# - In the get_fragment_output logic defined in the shaders here, the color
#   is pre-multiplied with the alpha.
# - All fixed-pipeling blending options assume premultiplied alpha.


standard_texture_des = {
    "sample_type": wgpu.TextureSampleType.unfilterable_float,
    "view_dimension": wgpu.TextureViewDimension.d2,
    "multisampled": False,
}


def blend_dict(src_factor, dst_factor, operation):
    return {
        "operation": operation,
        "src_factor": src_factor,
        "dst_factor": dst_factor,
    }


# %%%%%%%%%%  Define passes


class BasePass:
    """The base pass class, defining and documenting the API that a pass must provide."""

    render_mask = RenderMask.opaque | RenderMask.transparent
    write_pick = True

    def get_color_descriptors(self, blender, material_write_pick):
        """Get the list of fragment targets for device.create_render_pipeline()."""
        # The result affects the wobject's pipeline.
        # Returning [] prevents this pipeline from being created.
        return []

    def get_color_attachments(self, blender):
        """Get the list of color_attachments for command_encoder.begin_render_pass()."""
        # The result affects the rendering, but not the wobject's pipeline.
        # Returning [] prevents this pass from running.
        return []

    def get_depth_descriptor(self, blender):
        """Get dict that has the depth-specific fields for the depth_stencil field
        in device.create_render_pipeline().
        """
        return {}

    def get_depth_attachment(self, blender):
        """Get a dict that has the depth-specific fields for the
        depth_stencil_attachment argument of command_encoder.begin_render_pass().
        """
        return {}

    def get_shader_code(self, blender):
        """Get the fragment-write shader code.

        Notes:

        * This code gets injected into the shader, so the material shaders
          can use get_fragment_output.
        * This code should define FragmentOutput and get_fragment_output.
        """
        return ""


class OpaquePass(BasePass):
    """A pass that renders opaque fragments with depth testing, while
    discarting transparent fragments. This functions as the first pass
    in all multi-pass blenders.
    """

    render_mask = RenderMask.opaque
    write_pick = True

    def get_color_descriptors(self, blender, material_write_pick):
        bf, bo = wgpu.BlendFactor, wgpu.BlendOperation
        return [
            {
                "format": blender.color_format,
                "blend": {
                    "alpha": blend_dict(bf.one, bf.zero, bo.add),
                    "color": blend_dict(bf.one, bf.zero, bo.add),
                },
                "write_mask": wgpu.ColorWrite.ALL,
            },
            {
                "format": blender.pick_format,
                "blend": None,
                "write_mask": wgpu.ColorWrite.ALL if material_write_pick else 0,
            },
        ]

    def get_color_attachments(self, blender):
        color_load_op = pick_load_op = wgpu.LoadOp.load
        if blender.color_clear:
            blender.color_clear = False
            color_load_op = wgpu.LoadOp.clear
        if blender.pick_clear:
            blender.pick_clear = False
            pick_load_op = wgpu.LoadOp.clear

        return [
            {
                "view": blender.color_view,
                "resolve_target": None,
                "clear_value": (0, 0, 0, 0),
                "load_op": color_load_op,
                "store_op": wgpu.StoreOp.store,
            },
            {
                "view": blender.pick_view,
                "resolve_target": None,
                "clear_value": (0, 0, 0, 0),
                "load_op": pick_load_op,
                "store_op": wgpu.StoreOp.store,
            },
        ]

    def get_depth_descriptor(self, blender):
        return {
            "format": blender.depth_format,
            "depth_write_enabled": True,
            "depth_compare": wgpu.CompareFunction.less,
        }

    def get_depth_attachment(self, blender):
        depth_load_op = wgpu.LoadOp.load
        if blender.depth_clear:
            blender.depth_clear = False
            depth_load_op = wgpu.LoadOp.clear
        return {
            "view": blender.depth_view,
            "depth_clear_value": 1.0,
            "depth_load_op": depth_load_op,
            "depth_store_op": wgpu.StoreOp.store,
        }

    def get_shader_code(self, blender):
        return """
        struct FragmentOutput {
            @location(0) color: vec4<f32>,
            @location(1) pick: vec4<u32>,
        };
        fn get_fragment_output(depth: f32, color: vec4<f32>) -> FragmentOutput {
            if (color.a < 1.0 - ALPHA_COMPARE_EPSILON ) { discard; }
            var out : FragmentOutput;
            out.color = vec4<f32>(color.rgb, 1.0);
            return out;
        }
        """


class FullOpaquePass(OpaquePass):
    """A pass that considers all fragments opaque."""

    render_mask = RenderMask.opaque | RenderMask.transparent
    write_pick = True

    def get_shader_code(self, blender):
        return """
        struct FragmentOutput {
            @location(0) color: vec4<f32>,
            @location(1) pick: vec4<u32>,
        };
        fn get_fragment_output(depth: f32, color: vec4<f32>) -> FragmentOutput {
            var out : FragmentOutput;
            out.color = vec4<f32>(color.rgb, 1.0);  // always opaque
            return out;
        }
        """


class SimpleSinglePass(OpaquePass):
    """A pass that blends opaque and transparent fragments in a single pass."""

    render_mask = RenderMask.opaque | RenderMask.transparent
    write_pick = True

    def get_color_descriptors(self, blender, material_write_pick):
        bf, bo = wgpu.BlendFactor, wgpu.BlendOperation
        return [
            {
                "format": blender.color_format,
                "blend": {
                    "alpha": blend_dict(bf.one, bf.one_minus_src_alpha, bo.add),
                    "color": blend_dict(bf.one, bf.one_minus_src_alpha, bo.add),
                },
                "write_mask": wgpu.ColorWrite.ALL,
            },
            {
                "format": blender.pick_format,
                "blend": None,
                "write_mask": wgpu.ColorWrite.ALL if material_write_pick else 0,
            },
        ]

    def get_shader_code(self, blender):
        # Take depth into account, but don't treat transparent fragments differently
        return """
        struct FragmentOutput {
            @location(0) color: vec4<f32>,
            @location(1) pick: vec4<u32>,
        };
        fn get_fragment_output(depth: f32, color: vec4<f32>) -> FragmentOutput {
            var out : FragmentOutput;
            out.color = vec4<f32>(color.rgb * color.a, color.a);
            return out;
        }
        """


class SimpleTransparencyPass(BasePass):
    """A pass that only renders transparent fragments, blending them
    with the classic recursive alpha blending equation (a.k.a. the OVER
    operator).
    """

    render_mask = RenderMask.transparent
    write_pick = False

    def get_color_descriptors(self, blender, material_write_pick):
        bf, bo = wgpu.BlendFactor, wgpu.BlendOperation
        return [
            {
                "format": blender.color_format,
                "blend": {
                    "alpha": blend_dict(bf.one, bf.one_minus_src_alpha, bo.add),
                    "color": blend_dict(bf.one, bf.one_minus_src_alpha, bo.add),
                },
                "write_mask": wgpu.ColorWrite.ALL,
            },
        ]

    def get_color_attachments(self, blender):
        color_load_op = wgpu.LoadOp.load
        if blender.color_clear:
            blender.color_clear = False
            color_load_op = wgpu.LoadOp.clear

        return [
            {
                "view": blender.color_view,
                "resolve_target": None,
                "load_op": color_load_op,
                "store_op": wgpu.StoreOp.store,
            },
        ]

    def get_depth_descriptor(self, blender):
        return {
            "format": blender.depth_format,
            "depth_write_enabled": False,
            "depth_compare": wgpu.CompareFunction.less,
        }

    def get_depth_attachment(self, blender):
        depth_load_op = wgpu.LoadOp.load
        if blender.depth_clear:
            blender.depth_clear = False
            depth_load_op = wgpu.LoadOp.clear
        return {
            "view": blender.depth_view,
            "depth_load_op": depth_load_op,
            "depth_store_op": wgpu.StoreOp.store,
        }

    def get_shader_code(self, blender):
        return """
        struct FragmentOutput {
            @location(0) color: vec4<f32>,
        };
        fn get_fragment_output(depth: f32, color: vec4<f32>) -> FragmentOutput {
            if (color.a <= ALPHA_COMPARE_EPSILON) { discard; }
            var out : FragmentOutput;
            out.color = vec4<f32>(color.rgb * color.a, color.a);
            return out;
        }
        """


class WeightedTransparencyPass(BasePass):
    """A pass that implements weighted blended order-independent
    blending for transparent fragments, as proposed by McGuire in 2013.
    Multiple weight functions are supported.
    """

    render_mask = RenderMask.transparent
    write_pick = False

    def __init__(self, weight_func):
        if weight_func == "alpha":
            weight_code = """
                let weight = alpha;
            """
        elif weight_func == "depth":
            # The "generic purpose" weight function proposed by McGuire in
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

    def get_color_descriptors(self, blender, material_write_pick):
        bf, bo = wgpu.BlendFactor, wgpu.BlendOperation
        return [
            {
                "format": blender.accum_format,
                "blend": {
                    "alpha": blend_dict(bf.one, bf.one, bo.add),
                    "color": blend_dict(bf.one, bf.one, bo.add),
                },
                "write_mask": wgpu.ColorWrite.ALL,
            },
            {
                "format": blender.reveal_format,
                "blend": {
                    "alpha": blend_dict(bf.zero, bf.one_minus_src_alpha, bo.add),
                    "color": blend_dict(bf.zero, bf.one_minus_src, bo.add),
                },
                "write_mask": wgpu.ColorWrite.ALL,
            },
        ]

    def get_color_attachments(self, blender):
        # We always clear the textures used in this pass
        accum_load_op = reveal_load_op = wgpu.LoadOp.clear
        accum_clear_value = 0, 0, 0, 0
        reveal_clear_value = 1, 0, 0, 0
        return [
            {
                "view": blender.accum_view,
                "resolve_target": None,
                "clear_value": accum_clear_value,
                "load_op": accum_load_op,
                "store_op": wgpu.StoreOp.store,
            },
            {
                "view": blender.reveal_view,
                "resolve_target": None,
                "clear_value": reveal_clear_value,
                "load_op": reveal_load_op,
                "store_op": wgpu.StoreOp.store,
            },
        ]

    def get_depth_descriptor(self, blender):
        return {
            "format": blender.depth_format,
            "depth_write_enabled": False,
            "depth_compare": wgpu.CompareFunction.less,
        }

    def get_depth_attachment(self, blender):
        # We never clear the depth buffer in this pass
        depth_load_op = wgpu.LoadOp.load
        return {
            "view": blender.depth_view,
            "depth_load_op": depth_load_op,
            "depth_store_op": wgpu.StoreOp.store,
        }

    def get_shader_code(self, blender):
        return """
        struct FragmentOutput {
            @location(0) accum: vec4<f32>,
            @location(1) reveal: f32,
        };
        fn get_fragment_output(depth: f32, color: vec4<f32>) -> FragmentOutput {
            let alpha = color.a;
            if (alpha <= ALPHA_COMPARE_EPSILON) { discard; }
            let premultiplied = color.rgb * alpha;
            WEIGHT_CODE
            var out : FragmentOutput;
            out.accum = vec4<f32>(premultiplied, alpha) * weight;
            out.reveal = alpha;  // yes, alpha, not weight
            return out;
            // Note 1: could also take user-specified transmittance into account.
            // Note 2: its also possible to undo a fragment contribution. For this the accum
            // and reveal buffer must be float to avoid clamping. And we'd do `abs(color.a)` above.
            // The formula would then be:
            //    out.accum = - out.accum;
            //    out.reveal = 1.0 - 1.0 / (1.0 - alpha);
        }
        """.replace(
            "WEIGHT_CODE", self._weight_code
        )


class FrontmostTransparencyPass(BasePass):
    """A render pass that renders the front-most transparent layer to
    a custom render target. This can then later be used in the combine-pass.
    """

    write_pick = True

    def get_color_descriptors(self, blender, material_write_pick):
        bf, bo = wgpu.BlendFactor, wgpu.BlendOperation
        return [
            {
                "format": blender.frontcolor_format,
                "blend": {
                    "alpha": blend_dict(bf.one, bf.zero, bo.add),
                    "color": blend_dict(bf.one, bf.zero, bo.add),
                },
                "write_mask": wgpu.ColorWrite.ALL,
            },
            {
                "format": blender.pick_format,
                "blend": None,
                "write_mask": wgpu.ColorWrite.ALL if material_write_pick else 0,
            },
        ]

    def get_color_attachments(self, blender):
        # We always clear the dedicated color textire, but we share the
        # pick texture, so we check the clear-flag.
        color_load_op = wgpu.LoadOp.clear
        pick_load_op = wgpu.LoadOp.load
        if blender.pick_clear:
            blender.pick_clear = False
            pick_load_op = wgpu.LoadOp.clear
        return [
            {
                "view": blender.frontcolor_view,
                "resolve_target": None,
                "clear_value": (0, 0, 0, 0),
                "load_op": color_load_op,
                "store_op": wgpu.StoreOp.store,
            },
            {
                "view": blender.pick_view,
                "resolve_target": None,
                "clear_value": (0, 0, 0, 0),
                "load_op": pick_load_op,
                "store_op": wgpu.StoreOp.store,
            },
        ]

    def get_depth_descriptor(self, blender):
        return {
            "format": blender.depth_format,
            "depth_write_enabled": True,
            "depth_compare": wgpu.CompareFunction.less,
        }

    def get_depth_attachment(self, blender):
        depth_load_op = wgpu.LoadOp.load
        if blender.depth_clear:
            blender.depth_clear = False
            depth_load_op = wgpu.LoadOp.clear
        return {
            "view": blender.depth_view,
            "depth_clear_value": 1.0,
            "depth_load_op": depth_load_op,
            "depth_store_op": wgpu.StoreOp.store,
        }

    def get_shader_code(self, blender):
        return """
        struct FragmentOutput {
            @location(0) color: vec4<f32>,
            @location(1) pick: vec4<u32>,
        };
        fn get_fragment_output(depth: f32, color: vec4<f32>) -> FragmentOutput {
            if (color.a <= ALPHA_COMPARE_EPSILON || color.a >= 1.0 - ALPHA_COMPARE_EPSILON) { discard; }
            var out : FragmentOutput;
            out.color = vec4<f32>(color.rgb * color.a, color.a);
            return out;
        }
        """


# %%%%%%%%%% Define blenders


class BaseFragmentBlender:
    """Manage how fragments are blended and end up in the final target.
    Each renderer has one blender object.
    """

    passes = []

    def __init__(self):
        self.device = get_shared().device

        # The size (2D in pixels) of the frame textures.
        self.size = (0, 0)

        # Objects for the combination pass
        self._combine_pass_pipeline = None
        self._combine_pass_bind_group = None

        # A dict that contains the metadata for all render targets.
        self._texture_info = {}

        # The below targets are always present, and the renderer expects their
        # format, texture, and view to be present.
        # These contribute to 4+4+8 = 16 bytes per pixel

        usg = wgpu.TextureUsage

        # The color texture is in srgb, because in the shaders we work with physical
        # values, but we want to store as srgb to make effective use of the available bits.
        self._texture_info["color"] = (
            wgpu.TextureFormat.rgba8unorm_srgb,
            usg.RENDER_ATTACHMENT | usg.COPY_SRC | usg.TEXTURE_BINDING,
        )

        # The depth buffer is 32 bit - we need that precision.
        # Note that there is also depth32float-stencil8, but it needs the
        # (webgpu) extension with the same name.
        self._texture_info["depth"] = (
            wgpu.TextureFormat.depth32float,
            usg.RENDER_ATTACHMENT | usg.COPY_SRC,
        )

        # The pick texture has 4 16bit channels, adding up to 64 bits.
        # These bits are divided over the pick data, e.g. 20 for the wobject id.
        self._texture_info["pick"] = (
            wgpu.TextureFormat.rgba16uint,
            usg.RENDER_ATTACHMENT | usg.COPY_SRC,
        )

    def clear(self):
        """Clear the buffers."""
        for key in self._texture_info.keys():
            setattr(self, key + "_clear", True)

    def clear_depth(self):
        """Clear the deph buffer only."""
        self.depth_clear = True

    def ensure_target_size(self, size):
        """If necessary, resize render-textures to match the target size."""

        assert len(size) == 2
        size = size[0], size[1]
        if size == self.size:
            return

        # Set new size
        self.size = size
        tex_size = size + (1,)

        # Any bind group is now invalid because they include source textures.
        self._combine_pass_bind_group = None

        # Recreate internal textures
        for name, (format, usage) in self._texture_info.items():
            wgpu_texture = self.device.create_texture(
                size=tex_size, usage=usage, dimension="2d", format=format
            )
            setattr(self, name + "_format", format)
            setattr(self, name + "_tex", wgpu_texture)
            setattr(self, name + "_view", wgpu_texture.create_view())
            setattr(self, name + "_clear", True)

    # The five methods below represent the API that the render system uses.

    def get_color_descriptors(self, pass_index, material_write_pick):
        return self.passes[pass_index].get_color_descriptors(self, material_write_pick)

    def get_color_attachments(self, pass_index):
        return self.passes[pass_index].get_color_attachments(self)

    def get_depth_descriptor(self, pass_index, depth_test=True):
        des = self.passes[pass_index].get_depth_descriptor(self)
        if not depth_test:
            des["depth_compare"] = wgpu.CompareFunction.always
            des["depth_write_enabled"] = False
        return {
            **des,
            "stencil_read_mask": 0,
            "stencil_write_mask": 0,
            "stencil_front": {},  # use defaults
            "stencil_back": {},  # use defaults
        }

    def get_depth_attachment(self, pass_index):
        return {
            **self.passes[pass_index].get_depth_attachment(self),
            "stencil_read_only": True,
            "stencil_load_op": wgpu.LoadOp.clear,
            "stencil_store_op": wgpu.StoreOp.discard,
        }

    def get_shader_kwargs(self, pass_index):
        return {
            "blending_code": self.passes[pass_index].get_shader_code(self),
            "write_pick": self.passes[pass_index].write_pick,
        }

    def get_pass_count(self):
        """Get the number of passes for this blender."""
        return len(self.passes)

    def perform_combine_pass(self):
        """Perform a render-pass to combine any multi-pass results, if needed."""

        # Get bindgroup and pipeline. The creation should only happens once per blender lifetime.
        if not self._combine_pass_pipeline:
            self._combine_pass_pipeline = self._create_combination_pipeline()
        if not self._combine_pass_pipeline:
            return []

        # Get the bind group. A new one is needed when the source textures resize.
        if not self._combine_pass_bind_group:
            self._combine_pass_bind_group = self._create_combination_bind_group(
                self._combine_pass_pipeline.get_bind_group_layout(0)
            )

        command_encoder = self.device.create_command_encoder()

        # Render
        render_pass = command_encoder.begin_render_pass(
            color_attachments=[
                {
                    "view": self.color_view,
                    "resolve_target": None,
                    "load_op": wgpu.LoadOp.load,
                    "store_op": wgpu.StoreOp.store,
                }
            ],
            depth_stencil_attachment=None,
            occlusion_query_set=None,
        )
        render_pass.set_pipeline(self._combine_pass_pipeline)
        render_pass.set_bind_group(0, self._combine_pass_bind_group, [], 0, 99)
        render_pass.draw(4, 1)
        render_pass.end()

        return [command_encoder.finish()]

    def _create_combination_pipeline(self):
        """Overload this to setup the specific combiner-pass."""
        return None


class OpaqueFragmentBlender(BaseFragmentBlender):
    """A fragment blender that pretends that all surfaces are opaque,
    even if they're not.
    """

    passes = [FullOpaquePass()]


class Ordered1FragmentBlender(BaseFragmentBlender):
    """A minimal fragment blender that uses the classic alpha blending
    equation, without differentiating between opaque and transparent
    objects. This is a common approach for applications using a single
    pass. Order dependent.
    """

    passes = [SimpleSinglePass()]


class Ordered2FragmentBlender(BaseFragmentBlender):
    """A first step towards better blending: separating the opaque from
    the transparent fragments, and blending the latter using the alpha
    blending equation. The second step has depth-testing, but no
    depth-writing. Order dependent.
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

        # Create two additional render targets.
        # These contribute 8+1 = 9 bytes per pixel
        # So the total = 16 + 9 = 25 bytes per pixel

        # The accumulation buffer collects weighted fragments
        self._texture_info["accum"] = (
            wgpu.TextureFormat.rgba16float,
            usg.RENDER_ATTACHMENT | usg.TEXTURE_BINDING,
        )

        # The reveal buffer collects the weights.
        # McGuire: "Using R16F for the revealage render target will give slightly better
        # precision and make it easier to tune the algorithm, but a 2x savings on bandwidth
        # and memory footprint for that texture may make it worth compressing into R8 format."
        self._texture_info["reveal"] = (
            wgpu.TextureFormat.r8unorm,
            usg.RENDER_ATTACHMENT | usg.TEXTURE_BINDING,
        )

    def _create_combination_pipeline(self):
        binding_layout = [
            {
                "binding": 0,
                "visibility": wgpu.ShaderStage.FRAGMENT,
                "texture": standard_texture_des,
            },
            {
                "binding": 1,
                "visibility": wgpu.ShaderStage.FRAGMENT,
                "texture": standard_texture_des,
            },
        ]

        bf, bo = wgpu.BlendFactor, wgpu.BlendOperation
        targets = [
            {
                "format": self.color_format,
                "blend": {
                    "alpha": blend_dict(bf.one, bf.one_minus_src_alpha, bo.add),
                    "color": blend_dict(bf.one, bf.one_minus_src_alpha, bo.add),
                },
            },
        ]

        bindings_code = """
            @group(0) @binding(0)
            var r_accum: texture_2d<f32>;
            @group(0) @binding(1)
            var r_reveal: texture_2d<f32>;
        """

        fragment_code = """
            let epsilon = 1e-6;
            // Sample
            let accum = textureLoad(r_accum, texindex, 0).rgba;
            let reveal = textureLoad(r_reveal, texindex, 0).r;

            // Exit if no transparent fragments was written
            if (reveal >= 1.0) { discard; }  // no transparent fragments here

            // Reconstruct the color and alpha, and set final color, with premultiplied alpha
            let avg_color = accum.rgb / max(accum.a, epsilon);
            let alpha = 1.0 - reveal;
            out.color = vec4<f32>(avg_color * alpha, alpha);
        """

        return create_full_quad_pipeline(
            targets, binding_layout, bindings_code, fragment_code
        )

    def _create_combination_bind_group(self, bind_group_layout):
        # This must match the binding_layout above
        bind_group_entries = [
            {"binding": 0, "resource": self.accum_view},
            {"binding": 1, "resource": self.reveal_view},
        ]
        return self.device.create_bind_group(
            layout=bind_group_layout, entries=bind_group_entries
        )


class WeightedDepthFragmentBlender(WeightedFragmentBlender):
    """Weighted blended order independent transparency (McGuire 2013),
    using a general purpose depth weight function.
    """

    passes = [OpaquePass(), WeightedTransparencyPass("depth")]


class WeightedPlusFragmentBlender(WeightedFragmentBlender):
    """Three-pass weighted blended order independent transparency (McGuire 2013),
    using a weight function based on alpha, and in which the top-most
    transparent layer is actually in front.

    This uses the same approach as WeightedFragmentBlender, but in a
    3d pass we draw the front-most transparent layer. In the
    combine-pass, we subtract the front layer from the accum and reveal
    buffer, and add it again using the blend equation. In effect, the
    front-most layer is actually correct, and all transparent fragments
    behind it follow McGuire's approach. This looks a bit like a
    single-layer depth peeling.
    """

    # Since the front-most transparency pass also writes to the depth
    # buffer, we want to do the opaque pass first. And the weighed pass
    # should come ater the opaque pass.
    #
    # In an earlier version we did the front-most pass first, but then
    # each pass cleared the depth buffer. We should not do that if we
    # want to leave the depth buffer for post-processing or picking.
    #
    # Note that doing the front-post last means that the depth buffer
    # *does* include transparent fragments, while this is not the case
    # for the other blend modes.

    passes = [
        OpaquePass(),
        WeightedTransparencyPass("alpha"),
        FrontmostTransparencyPass(),
    ]

    def __init__(self):
        super().__init__()

        usg = wgpu.TextureUsage

        # Create one additional render target.
        # These contribute 4 bytes per pixel
        # So the total = 16 + 9 + 4 = 29 bytes per pixel

        # Color buffer for the front-most semitransparent layer
        self._texture_info["frontcolor"] = (
            wgpu.TextureFormat.rgba8unorm_srgb,
            usg.RENDER_ATTACHMENT | usg.TEXTURE_BINDING,
        )

    def _create_combination_pipeline(self):
        binding_layout = [
            {
                "binding": 0,
                "visibility": wgpu.ShaderStage.FRAGMENT,
                "texture": standard_texture_des,
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

        bf, bo = wgpu.BlendFactor, wgpu.BlendOperation
        targets = [
            {
                "format": self.color_format,
                "blend": {
                    "alpha": blend_dict(bf.one, bf.one_minus_src_alpha, bo.add),
                    "color": blend_dict(bf.one, bf.one_minus_src_alpha, bo.add),
                },
            },
        ]

        bindings_code = """
            @group(0) @binding(0)
            var r_accum: texture_2d<f32>;
            @group(0) @binding(1)
            var r_reveal: texture_2d<f32>;
            @group(0) @binding(2)
            var r_frontcolor: texture_2d<f32>;
        """

        fragment_code = """
            let epsilon = 1e-6;

            // Sample
            var accum = textureLoad(r_accum, texindex, 0).rgba;
            var reveal = textureLoad(r_reveal, texindex, 0).r;
            let front = textureLoad(r_frontcolor, texindex, 0).rgba;

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

        return create_full_quad_pipeline(
            targets, binding_layout, bindings_code, fragment_code
        )

    def _create_combination_bind_group(self, bind_group_layout):
        # This must match the binding_layout above
        bind_group_entries = [
            {"binding": 0, "resource": self.accum_view},
            {"binding": 1, "resource": self.reveal_view},
            {"binding": 2, "resource": self.frontcolor_view},
        ]
        return self.device.create_bind_group(
            layout=bind_group_layout, entries=bind_group_entries
        )


class AdditivePass(BasePass):
    """A pass that renders fragments with additive blending."""

    render_mask = RenderMask.opaque | RenderMask.transparent
    write_pick = False

    def get_color_descriptors(self, blender, material_write_pick):
        bf, bo = wgpu.BlendFactor, wgpu.BlendOperation
        return [
            {
                "format": blender.color_format,
                "blend": {
                    "alpha": blend_dict(bf.one, bf.one, bo.add),
                    "color": blend_dict(bf.one, bf.one, bo.add),
                },
                "write_mask": wgpu.ColorWrite.ALL,
            },
        ]

    def get_color_attachments(self, blender):
        color_load_op = wgpu.LoadOp.load
        if blender.color_clear:
            blender.color_clear = False
            color_load_op = wgpu.LoadOp.clear

        return [
            {
                "view": blender.color_view,
                "resolve_target": None,
                "load_op": color_load_op,
                "store_op": wgpu.StoreOp.store,
            },
        ]

    def get_depth_descriptor(self, blender):
        return {
            "format": blender.depth_format,
            "depth_write_enabled": False,
            "depth_compare": wgpu.CompareFunction.always,
        }

    def get_depth_attachment(self, blender):
        depth_load_op = wgpu.LoadOp.load
        if blender.depth_clear:
            blender.depth_clear = False
            depth_load_op = wgpu.LoadOp.clear
        return {
            "view": blender.depth_view,
            "depth_load_op": depth_load_op,
            "depth_store_op": wgpu.StoreOp.store,
        }

    def get_shader_code(self, blender):
        return """
        struct FragmentOutput {
            @location(0) color: vec4<f32>,
        };
        fn get_fragment_output(depth: f32, color: vec4<f32>) -> FragmentOutput {
            var out : FragmentOutput;
            out.color = color;
            return out;
        }
        """


class AdditiveFragmentBlender(BaseFragmentBlender):
    """A fragment blender that uses additive blending for all fragments."""

    passes = [AdditivePass()]
