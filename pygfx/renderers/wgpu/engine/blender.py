"""
Defines the classes for the different render passes and the blender
objects that contain them. A blender becomes part of the renderstate
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


class TheOneAndOnlyBlender:
    """Manage how fragments are blended and end up in the final target.
    Each renderer has one blender object.
    """

    # Each blender should define a unique name so that it may be
    # correctly registered in the renderer.
    name = None
    passes = []

    def __init__(self):
        self.device = get_shared().device

        # The size (2D in pixels) of the frame textures.
        self.size = (0, 0)

        # Objects for the combination pass
        self._weighted_blending_was_used = False
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

    def clear(self):
        """Clear the buffers."""
        self._weighted_blending_was_used = False
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
        tex_size = (*size, 1)

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

        # TODO: only create accum and reveal textures when needed!!!

    # The five methods below represent the API that the render system uses.

    def get_color_descriptors(self, pass_index, material_write_pick, blending):
        # todo: remove pass_index
        bf, bo = wgpu.BlendFactor, wgpu.BlendOperation
        color_blend = None
        if isinstance(blending, str):
            if blending == "no":
                color_blend = {
                    "alpha": blend_dict(bf.one, bf.zero, bo.add),
                    "color": blend_dict(bf.one, bf.zero, bo.add),
                }
            elif blending == "normal":
                color_blend = {
                    "alpha": blend_dict(bf.one, bf.one_minus_src_alpha, bo.add),
                    "color": blend_dict(bf.src_alpha, bf.one_minus_src_alpha, bo.add),
                }
            elif blending == "add":
                color_blend = {
                    "alpha": blend_dict(bf.one, bf.one, bo.add),
                    "color": blend_dict(bf.one, bf.one, bo.add),
                }
            elif blending == "subtract":
                # todo: is this correct?
                color_blend = {
                    "alpha": blend_dict(bf.one, bf.one, bo.subtract),
                    "color": blend_dict(bf.one, bf.one, bo.subtract),
                }
            elif blending == "dither":
                color_blend = {
                    "alpha": blend_dict(bf.one, bf.zero, bo.add),
                    "color": blend_dict(bf.one, bf.zero, bo.add),
                }
            elif blending == "weighted":
                pass  # handled below
            else:
                raise RuntimeError(f"Unexpected blending string {blending:!r}")
        else:
            raise RuntimeError(f"Unexpected blending {blending:!r}")

        color_descriptor = {
            "format": self.color_format,
            "blend": color_blend,
            "write_mask": wgpu.ColorWrite.ALL,
        }
        pick_descriptor = {
            "format": self.pick_format,
            "blend": None,
            "write_mask": wgpu.ColorWrite.ALL if material_write_pick else 0,
        }
        descriptors = [color_descriptor, pick_descriptor]

        if blending == "weighted":
            accum_descriptor = {
                "format": self.accum_format,
                "blend": {
                    "alpha": blend_dict(bf.one, bf.one, bo.add),
                    "color": blend_dict(bf.one, bf.one, bo.add),
                },
                "write_mask": wgpu.ColorWrite.ALL,
            }
            reveal_descriptor = {
                "format": self.reveal_format,
                "blend": {
                    "alpha": blend_dict(bf.zero, bf.one_minus_src_alpha, bo.add),
                    "color": blend_dict(bf.zero, bf.one_minus_src, bo.add),
                },
                "write_mask": wgpu.ColorWrite.ALL,
            }
            descriptors = [accum_descriptor, reveal_descriptor, pick_descriptor]

        return descriptors

    def get_depth_descriptor(self, pass_index, depth_test=True, depth_write=True):
        depth_write = bool(depth_write)
        depth_compare = (
            wgpu.CompareFunction.less if depth_test else wgpu.CompareFunction.always
        )
        return {
            "format": self.depth_format,
            "depth_write_enabled": depth_write,
            "depth_compare": depth_compare,
            "stencil_read_mask": 0,
            "stencil_write_mask": 0,
            "stencil_front": {},  # use defaults
            "stencil_back": {},  # use defaults
        }

    def get_color_attachments(self, pass_index, pass_type):
        color_load_op = pick_load_op = wgpu.LoadOp.load
        if self.color_clear:
            self.color_clear = False
            color_load_op = wgpu.LoadOp.clear
        if self.pick_clear:
            self.pick_clear = False
            pick_load_op = wgpu.LoadOp.clear

        color_attachment = {
            "view": self.color_view,
            "resolve_target": None,
            "clear_value": (0, 0, 0, 0),
            "load_op": color_load_op,
            "store_op": wgpu.StoreOp.store,
        }
        pick_attachment = {
            "view": self.pick_view,
            "resolve_target": None,
            "clear_value": (0, 0, 0, 0),
            "load_op": pick_load_op,
            "store_op": wgpu.StoreOp.store,
        }
        attachments = [color_attachment, pick_attachment]

        if pass_type == "weighted":
            self._weighted_blending_was_used = True
            # TODO: this'd be a good time to make sure the accum and reveal texture are ready
            # We always clear the textures used in this pass
            # TODO: always clear??
            accum_load_op = reveal_load_op = wgpu.LoadOp.clear
            accum_clear_value = 0, 0, 0, 0
            reveal_clear_value = 1, 0, 0, 0
            accum_attachment = {
                "view": self.accum_view,
                "resolve_target": None,
                "clear_value": accum_clear_value,
                "load_op": accum_load_op,
                "store_op": wgpu.StoreOp.store,
            }
            reveal_attachment = {
                "view": self.reveal_view,
                "resolve_target": None,
                "clear_value": reveal_clear_value,
                "load_op": reveal_load_op,
                "store_op": wgpu.StoreOp.store,
            }
            attachments = [accum_attachment, reveal_attachment, pick_attachment]

        return attachments

    def get_depth_attachment(self, pass_index):
        # We don't use the stencil yet, but when we do, we will also have to specify
        # "stencil_read_only", "stencil_load_op", and "stencil_store_op"
        depth_load_op = wgpu.LoadOp.load
        if self.depth_clear:
            self.depth_clear = False
            depth_load_op = wgpu.LoadOp.clear
        return {
            "view": self.depth_view,
            "depth_clear_value": 1.0,
            "depth_load_op": depth_load_op,
            "depth_store_op": wgpu.StoreOp.store,
        }

    def get_shader_kwargs(self, pass_index, blending):
        # Take depth into account, but don't treat transparent fragments differently

        code = """
        struct FragmentOutput {
            @location(0) color: vec4<f32>,
            @location(1) pick: vec4<u32>,
        };
        fn get_fragment_output(position: vec4<f32>, color: vec4<f32>) -> FragmentOutput {
            var out : FragmentOutput;
            SUBCODE
            return out;
        }
        """

        if blending == "no":
            subcode = "out.color = vec4<f32>(color.rgb, 1.0);"
        elif blending in ("normal", "add", "subtract"):
            subcode = "out.color = vec4<f32>(color.rgb, color.a);"
        elif blending == "dither":
            # We want the seed for the random function to be such that the result is
            # deterministic, so that rendered images can be visually compared. This
            # is why the object-id should not be used. Using the xy ndc coord is a
            # no-brainer seed. Using only these will give an effect often observed
            # in games, where the pattern is "stuck to the screen". We also seed with
            # the depth, since this covers *a lot* of cases, e.g. different objects
            # behind each-other, as well as the same object having different parts
            # at the same screen pixel. This only does not cover cases where objects
            # are exactly on top of each other. Therefore we use rgba as another seed.
            # So the only case where the same pattern may be used for different
            # fragments if an object is at the same depth and has the same color.
            subcode = """
            let seed1 = position.x * position.y * position.z;
            let seed2 = color.r * 0.12 + color.g * 0.34 + color.b * 0.56 + color.a * 0.78;
            let rand = random2(vec2<f32>(seed1, seed2));
            if ( color.a < 1.0 - ALPHA_COMPARE_EPSILON && color.a < rand ) { discard; }
            out.color = vec4<f32>(color.rgb, 1.0);  // fragments that pass through are opaque
            """.strip()

        elif blending == "weighted":
            weight_func = "alpha"  # TODO: parametrize, maybe blending="weighted: depth", or a dict
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
                    f"Unknown weighted-blending weight_func: {weight_func!r}"
                )

            code = """
            struct FragmentOutput {
                @location(0) accum: vec4<f32>,
                @location(1) reveal: f32,
            };
            fn get_fragment_output(position: vec4<f32>, color: vec4<f32>) -> FragmentOutput {
                let depth = position.z;
                let alpha = color.a;
                if (alpha <= ALPHA_COMPARE_EPSILON) { discard; }
                let premultiplied = color.rgb * alpha;
                SUBCODE
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
            """
            subcode = weight_code
        else:
            raise RuntimeError(f"Unexpected blending {blending!r}")

        return {
            "blending_code": code.replace("SUBCODE", subcode),
            "write_pick": True,  # todo: factore this out
        }

    def get_pass_count(self):
        """Get the number of passes for this blender."""
        return 1

    def perform_combine_pass(self, command_encoder):
        """Perform a render-pass to combine any multi-pass results, if needed."""

        # This is only needed for weighted blending
        if not self._weighted_blending_was_used:
            return

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
