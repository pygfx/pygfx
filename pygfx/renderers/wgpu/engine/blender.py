"""
The blender is the part of the renderer that manages output targets and blending configuration, based on material.blending.
"""

import wgpu  # only for flags/enums

from .flusher import create_full_quad_pipeline
from .shared import get_shared


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


DEPTH_MAP = {
    "": wgpu.CompareFunction.always,
    "<": wgpu.CompareFunction.less,
    "<=": wgpu.CompareFunction.less_equal,
    "==": wgpu.CompareFunction.equal,
    "!=": wgpu.CompareFunction.not_equal,
    ">=": wgpu.CompareFunction.greater_equal,
    ">": wgpu.CompareFunction.greater,
}

usg = wgpu.TextureUsage
default_targets = {
    # The color texture is in srgb, because in the shaders we work with physical
    # values, but we want to store as srgb to make effective use of the available bits. It's 4 bytes per pixel.
    "color": (
        wgpu.TextureFormat.rgba8unorm_srgb,
        usg.RENDER_ATTACHMENT | usg.COPY_SRC | usg.TEXTURE_BINDING,
    ),
    # The depth buffer should preferably at least 24bit - we need that precision. It's 4 bytes per pixel.
    # 32 but is cool, but we may want stencil at some point, so depth24plus_stencil8 seems like a good default.
    # The depth24plus is either depth32float or depth24unorm, depending on the backend.
    # Note that there is also depth32float-stencil8, but it needs the (webgpu) extension with the same name.
    "depth": (
        wgpu.TextureFormat.depth32float,
        usg.RENDER_ATTACHMENT | usg.COPY_SRC,
    ),
    # The pick texture has 4 16bit channels. It's 8 bytes per pixel.
    # These bits are divided over the pick data, e.g. 20 for the wobject id.
    "pick": (
        wgpu.TextureFormat.rgba16uint,
        usg.RENDER_ATTACHMENT | usg.COPY_SRC,
    ),
    # The accumulation buffer collects weighted fragments. It's 8 bytes per pixel.
    "accum": (
        wgpu.TextureFormat.rgba16float,
        usg.RENDER_ATTACHMENT | usg.TEXTURE_BINDING,
    ),
    # The reveal buffer collects the weights. It's 1 byte per pixel.
    # McGuire: "Using R16F for the revealage render target will give slightly better
    # precision and make it easier to tune the algorithm, but a 2x savings on bandwidth
    # and memory footprint for that texture may make it worth compressing into R8 format."
    "reveal": (
        wgpu.TextureFormat.r8unorm,
        usg.RENDER_ATTACHMENT | usg.TEXTURE_BINDING,
    ),
}


class Blender:
    """Manage how fragments are blended and end up in the final target.
    Each renderer has one blender object.
    """

    def __init__(self, *, enable_pick=True, enable_depth=True):
        self.device = get_shared().device

        # We could allow custom targets, but this is not yet implemented in the methods.
        # The code in this init shows the first steps of what that could look like.
        custom_targets = {}  # name -> (format, usage)  could allow users to specify this

        # The size (2D in pixels) of the textures.
        self.size = (0, 0)

        # Objects for the combination pass
        self._weighted_blending_was_used_in_last_pass = False
        self._weighted_resolve_pass_pipeline = None
        self._weighted_resolve_pass_bind_group = None

        # A dict that contains the metadata for all render targets.
        self._texture_info = {}  # name -> dict

        # A dict with the actual textures
        self._textures = {}  # name -> Texture

        # -- setup targets

        # Collect render targets
        all_targets = default_targets.copy()
        for name in sorted(custom_targets):
            all_targets[name] = custom_targets[name]
        if not enable_pick:
            all_targets.pop("pick")
        if not enable_depth:
            all_targets.pop("depth")

        # Convert to final dict
        for name, (format, usage) in all_targets.items():
            assert isinstance(name, str)
            assert format in wgpu.TextureFormat
            assert isinstance(usage, int)
            self._texture_info[name] = {
                "name": name,
                "format": format,
                "usage": usage,
                "is_used": False,
                "clear": True,
            }

    @property
    def hash(self):
        """The hash for this blender.

        This is used by the renderstate. If an object is rendered with two renderers,
        we still want this to work, even if the corresponding blenders results in incompatible pipelines.

        Use cases are a blender without a pick or depth target. Or blenders with different color formats.

        In such cases, their blender hashes will be different, resulting in a separate pipeline object for each.
        Similar to rendering with a different number of lights will result in separate pipeline objects.
        """
        return ", ".join(
            info["name"] + ": " + info["format"] for info in self._texture_info.values()
        )

    @property
    def texture_info(self):
        """A dict of dicts, containing per-rendertarget info."""
        return self._texture_info

    def get_texture(self, name):
        """Get the texture object for the given name."""
        return self._textures.get(name)

    def get_texture_view(self, name, usage):
        """Get a texture view for the given name, with the given usage. Returns None if the texture does not exist."""
        texture = self._textures.get(name)
        if texture is None:
            return None
        else:
            return texture.create_view(usage=usage)

    def _get_texture_view_for_rendering(self, name):
        """Get a texture view for the given name with rendert-attachment usage. Creates the texture if it does not exist."""
        texture = self._textures.get(name)

        if texture is None:
            texinfo = self._texture_info[name]
            texture = self.device.create_texture(
                dimension="2d",
                size=(*self.size, 1),
                usage=texinfo["usage"],
                format=texinfo["format"],
            )
            self._textures[name] = texture

        return texture.create_view(usage=wgpu.TextureUsage.RENDER_ATTACHMENT)

    def clear(self):
        """Clear all the buffers (on the next time they're attached)."""
        for texstate in self._texture_info.values():
            texstate["clear"] = True

    def ensure_target_size(self, size):
        """Make sure that the textures are resized if necessary."""

        assert len(size) == 2
        size = int(size[0]), int(size[1])
        if size == self.size:
            return

        # Set new size
        self.size = size

        # Any bind group is now invalid because they include source textures.
        self._weighted_resolve_pass_bind_group = None

        # All textures are invalidated too
        self._textures = {}

    def get_color_descriptors(self, material_pick_write, blending):
        """Get the dict color-descriptors that pipeline.py needs to create the render pipeline.

        Called per object when the pipeline is created.
        """
        bf, bo = wgpu.BlendFactor, wgpu.BlendOperation

        blending_mode = blending["mode"]

        # Get the color_blend mini-dict
        if blending_mode == "classic":
            color_blend = {
                "color": blend_dict(
                    blending["color_src"],
                    blending["color_dst"],
                    blending.get("color_op", "add"),
                ),
                "alpha": blend_dict(
                    blending["alpha_src"],
                    blending["alpha_dst"],
                    blending.get("alpha_op", "add"),
                ),
            }
        elif blending_mode == "dither":
            color_blend = {
                "alpha": blend_dict(bf.one, bf.zero, bo.add),
                "color": blend_dict(bf.one, bf.zero, bo.add),
            }
        elif blending_mode == "weighted":
            color_blend = None  # handled below
        else:
            raise RuntimeError(f"Unexpected blending mode {blending_mode:!r}")

        # Build target state dicts
        color_target_state = {
            "name": "color",
            "blend": color_blend,
            "write_mask": wgpu.ColorWrite.ALL,
        }
        target_states = [color_target_state]

        # More work for weighted blending
        if blending_mode == "weighted":
            accum_target_state = {
                "name": "accum",
                "blend": {
                    "alpha": blend_dict(bf.one, bf.one, bo.add),
                    "color": blend_dict(bf.one, bf.one, bo.add),
                },
                "write_mask": wgpu.ColorWrite.ALL,
            }
            reveal_target_state = {
                "name": "reveal",
                "blend": {
                    "alpha": blend_dict(bf.zero, bf.one_minus_src_alpha, bo.add),
                    "color": blend_dict(bf.zero, bf.one_minus_src, bo.add),
                },
                "write_mask": wgpu.ColorWrite.ALL,
            }
            target_states = [accum_target_state, reveal_target_state]

        # Add pick target if the blender supports it. All pipelines must have matching (target states in their) pipelines.
        # Whether or not the pick target is used is determined by the material using the write_mask.
        if "pick" in self._texture_info:
            target_state = {
                "name": "pick",
                "blend": None,
                "write_mask": wgpu.ColorWrite.ALL if material_pick_write else 0,
            }
            target_states.append(target_state)

        # Note: in a similar way we could allow custom additional render targets.

        # Set format for each target, and register what targets were used
        for target_state in target_states:
            name = target_state.pop("name")
            texinfo = self._texture_info[name]
            target_state["format"] = texinfo["format"]
            texinfo["is_used"] = True

        return target_states

    def get_depth_descriptor(self, depth_test, depth_compare, depth_write):
        """Get the dict depth-descriptors that pipeline.py needs to create the render pipeline.

        Called per object when the pipeline is created.
        """

        if "depth" not in self._texture_info:
            return None

        depth_write = bool(depth_write)
        if depth_test:
            wgpu_depth_compare = DEPTH_MAP[depth_compare]
        else:
            wgpu_depth_compare = wgpu.CompareFunction.always

        texinfo = self._texture_info["depth"]
        texinfo["is_used"] = True

        return {
            "format": texinfo["format"],
            "depth_write_enabled": depth_write,
            "depth_compare": wgpu_depth_compare,
            "stencil_read_mask": 0,
            "stencil_write_mask": 0,
            "stencil_front": {},  # use defaults
            "stencil_back": {},  # use defaults
        }

    def get_color_attachments(self, pass_type):
        """Get the texture render targets for color.

        These are dynamically attached right before rendering a batch of objects.
        This is called by the renderer for each 'batch' of objects (of a particular pass_type) is being rendered.
        """

        color_attachment = {
            "name": "color",
            "resolve_target": None,
            "clear_value": (0, 0, 0, 0),
            "load_op": wgpu.LoadOp.load,  # maybe set to clear at the end of this func
            "store_op": wgpu.StoreOp.store,
        }
        pick_attachment = {
            "name": "pick",
            "resolve_target": None,
            "clear_value": (0, 0, 0, 0),
            "load_op": wgpu.LoadOp.load,
            "store_op": wgpu.StoreOp.store,
        }
        attachments = [color_attachment]

        self._weighted_blending_was_used_in_last_pass = False

        if pass_type == "weighted":
            self._weighted_blending_was_used_in_last_pass = True
            # We always clear the textures at the beginning of a pass, because at the end of
            # that pass it will be merged with the color buffer using the combine pass.
            accum_clear_value = 0, 0, 0, 0
            reveal_clear_value = 1, 0, 0, 0
            accum_attachment = {
                "name": "accum",
                "resolve_target": None,
                "clear_value": accum_clear_value,
                "load_op": wgpu.LoadOp.clear,
                "store_op": wgpu.StoreOp.store,
            }
            reveal_attachment = {
                "name": "reveal",
                "resolve_target": None,
                "clear_value": reveal_clear_value,
                "load_op": wgpu.LoadOp.clear,
                "store_op": wgpu.StoreOp.store,
            }
            attachments = [accum_attachment, reveal_attachment]

        # Add pick attachment if this blender supports picking.
        if "pick" in self._texture_info:
            attachments.append(pick_attachment)

        # Finalize attachments
        for attachment in attachments:
            name = attachment.pop("name")
            texinfo = self._texture_info[name]

            if texinfo["clear"]:
                texinfo["clear"] = False
                attachment["load_op"] = wgpu.LoadOp.clear

            attachment["view"] = self._get_texture_view_for_rendering(name)

        return attachments

    def get_depth_attachment(self):
        """Get the texture render targets for depth/stencil. These are dynamically attached right before rendering a batch of objects."""

        if "depth" not in self._texture_info:
            return None

        texinfo = self._texture_info["depth"]

        load_op = wgpu.LoadOp.load
        if texinfo["clear"]:
            texinfo["clear"] = False
            load_op = wgpu.LoadOp.clear

        view = self._get_texture_view_for_rendering("depth")

        # We don't use the stencil yet, but when we do, we will also have to specify
        # "stencil_read_only", "stencil_load_op", and "stencil_store_op"
        return {
            "view": view,
            "depth_clear_value": 1.0,
            "depth_load_op": load_op,
            "depth_store_op": wgpu.StoreOp.store,
        }

    def get_shader_kwargs(self, material_pick_write, blending):
        """Get the shader templating variables for the given blending."""

        blending_mode = blending["mode"]

        if blending_mode == "classic":
            blending_code = """
            struct FragmentOutput {
                @location(0) color: vec4<f32>,
                MAYBE_PICK@location(1) pick: vec4<u32>,
            };
            """

        elif blending_mode == "dither":
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
            blending_code = """
            struct FragmentOutput {
                // virtualfield seed1 : f32 = varyings.position.x * varyings.position.y * varyings.position.z
                // virtualfield seed2 : f32 = out.color.r * 0.12 + out.color.g * 0.34 + out.color.b * 0.56 + out.color.a * 0.78
                @location(0) color: vec4<f32>,
                MAYBE_PICK@location(1) pick: vec4<u32>,
            };

            fn apply_virtual_fields_of_fragment_output(outp: ptr<function,FragmentOutput>, seed1:f32, seed2:f32) {
                let rand = random2(vec2<f32>(seed1, seed2));
                let alpha = (*outp).color.a;
                if ( alpha < 1.0 - ALPHA_COMPARE_EPSILON && alpha < rand ) { discard; }
                (*outp).color.a = 1.0;  // fragments that pass through are opaque
            }
            """

            # Optimization for the case when the object is known (or determined) to be opaque.
            # The discard will not happen then. By removing it from the shader, we enable early-z optimizations.
            # By disabling the whole function, it behaves as classic blending for opaque-declared objects.
            if blending.get("no_discard"):
                blending_code, selector, rest = blending_code.partition("let rand")
                assert selector, "looks like dither code changed, selector not present"
                blending_code = blending_code.rstrip() + "}"

        elif blending_mode == "weighted":
            use_alpha = "alpha", "use_alpha", "weighted_blending_use_alpha"
            weight_default = blending.get("weight", "alpha")
            if weight_default in use_alpha:
                weight_default = "weighted_blending_use_alpha"
            alpha_default = blending.get("alpha", "alpha")
            if alpha_default in use_alpha:
                alpha_default = "weighted_blending_use_alpha"

            blending_code = """
            struct FragmentOutput {
                // virtualfield color : vec4<f32> = vec4<f32>(0.0)
                // virtualfield alpha : f32 = ALPHA_DEFAULT
                // virtualfield weight : f32 = WEIGHT_DEFAULT
                @location(0) accum: vec4<f32>,
                @location(1) reveal: f32,
                MAYBE_PICK@location(2) pick: vec4<u32>,
            };
            const weighted_blending_use_alpha: f32 = -42.0;

            fn apply_virtual_fields_of_fragment_output(outp: ptr<function,FragmentOutput>, color: vec4<f32>,  _alpha: f32, _weight: f32) {
                let alpha = select(_alpha, color.a, _alpha == weighted_blending_use_alpha);
                let weight = select(_weight, color.a, _weight == weighted_blending_use_alpha);
                (*outp).accum = vec4<f32>(color.rgb * alpha, alpha) * weight;
                (*outp).reveal = alpha;  // yes, alpha, not weight
            }
            """.replace("WEIGHT_DEFAULT", weight_default).replace(
                "ALPHA_DEFAULT", alpha_default
            )

            # Note 1: could also take user-specified transmittance into account.
            # Note 2: its also possible to undo a fragment contribution. For this the accum
            # and reveal buffer must be float to avoid clamping. And we'd do `abs(color.a)` above.
            # The formula would then be:
            #    out.accum = - out.accum;
            #    out.reveal = 1.0 - 1.0 / (1.0 - alpha);

        else:
            raise RuntimeError(f"Unexpected blending mode {blending_mode!r}")

        # Enable/disable picking in the shader
        do_pick = material_pick_write and "pick" in self._texture_info
        blending_code = blending_code.replace("MAYBE_PICK", "" if do_pick else "// ")

        return {"blending_code": blending_code, "write_pick": do_pick}

    def perform_weighted_resolve_pass(self, command_encoder):
        """Perform a render-pass to resolve the result of weighted blending."""

        # This is only needed if objects were rendered with weighted blending
        if not self._weighted_blending_was_used_in_last_pass:
            return

        # Get bindgroup and pipeline. The creation should only happens once per blender lifetime.
        if not self._weighted_resolve_pass_pipeline:
            self._weighted_resolve_pass_pipeline = self._create_combination_pipeline()
        if not self._weighted_resolve_pass_pipeline:
            return []

        # Get the bind group. A new one is needed when the source textures resize.
        if not self._weighted_resolve_pass_bind_group:
            self._weighted_resolve_pass_bind_group = (
                self._create_combination_bind_group(
                    self._weighted_resolve_pass_pipeline.get_bind_group_layout(0)
                )
            )

        # Get info on the color target texture
        texinfo = self._texture_info["color"]
        load_op = wgpu.LoadOp.load
        if texinfo["clear"]:
            texinfo["clear"] = False
            load_op = wgpu.LoadOp.clear

        # Render
        render_pass = command_encoder.begin_render_pass(
            color_attachments=[
                {
                    "view": self._get_texture_view_for_rendering("color"),
                    "resolve_target": None,
                    "load_op": load_op,
                    "store_op": wgpu.StoreOp.store,
                }
            ],
            depth_stencil_attachment=None,
            occlusion_query_set=None,
        )
        render_pass.set_pipeline(self._weighted_resolve_pass_pipeline)
        render_pass.set_bind_group(0, self._weighted_resolve_pass_bind_group, [], 0, 99)
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
                "format": self._texture_info["color"]["format"],
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
            // This discard does not brake early-z, because with weighted blending you're rendering each object anyway.
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
            {
                "binding": 0,
                "resource": self.get_texture_view(
                    "accum", wgpu.TextureUsage.TEXTURE_BINDING
                ),
            },
            {
                "binding": 1,
                "resource": self.get_texture_view(
                    "reveal", wgpu.TextureUsage.TEXTURE_BINDING
                ),
            },
        ]
        return self.device.create_bind_group(
            layout=bind_group_layout, entries=bind_group_entries
        )
