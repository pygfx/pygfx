"""
The blender is the part of the renderer that manages output targets and alpha_config.
"""

import wgpu  # only for flags/enums

from .effectpasses import create_full_quad_pipeline
from .shared import get_shared
from ..wgsl import load_wgsl


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
    # Inside all our shaders, as well as in the EffectPass shaders, we follow a
    # "linear workflow"; inside the shaders all colors are assumed to be in linear
    # space (colors read from colormaps are converted when read). The final color,
    # displayed on screen (or stored to PNG, etc.) should be sRGB so that the colors
    # are perceived correctly. This conversion should preferably be done at the end.
    #
    # As for storing intermediate results, using 8bit colors in linear space would
    # mean that we lose precision in regions where the human eye is sensitive.
    # Ideally we'd use 'rgba16float', and maybe we should, it's 2025, but let's
    # change that in a self-contained commit, because it will affect memory usage,
    # and needs a feature?
    #
    # Using 'rgba8unorm_srgb' means that in each post-effect step, the colors are
    # stored in srgb, and auto-conveted to linear when read/written. This seems a
    # bit weird, because from the pov of math that works on linear values, it loses
    # precision in a non-linear way. But that way we lose precision "in a way that's
    # linear to the human eye", which results in a better end-result than using
    # 'rgba8unorm'. Also see https://github.com/pmndrs/postprocessing
    #
    # This is 4 bytes per pixel.
    # TODO: use half-floats
    "color": (
        wgpu.TextureFormat.rgba8unorm_srgb,
        usg.RENDER_ATTACHMENT | usg.COPY_SRC | usg.TEXTURE_BINDING,
    ),
    # A texture of the same size, to allow post-processing effects.
    # When applying effects, the color and altcolor texture are ping-ponged.
    "altcolor": (
        wgpu.TextureFormat.rgba8unorm_srgb,
        usg.RENDER_ATTACHMENT | usg.COPY_SRC | usg.TEXTURE_BINDING,
    ),
    # The depth buffer should preferably at least 24bit - we need that precision. It's 4 bytes per pixel.
    # 32 but is cool, but we may want stencil at some point, so depth24plus_stencil8 seems like a good default.
    # The depth24plus is either depth32float or depth24unorm, depending on the backend.
    # Note that there is also depth32float-stencil8, but it needs the (webgpu) extension with the same name.
    "depth": (
        wgpu.TextureFormat.depth32float,
        usg.RENDER_ATTACHMENT | usg.COPY_SRC | usg.TEXTURE_BINDING,
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

    def get_texture_view(self, name, usage, *, create_if_not_exist=False):
        """Get a texture view for the given name, with the given usage. Returns None if the texture does not exist."""
        texture = self._textures.get(name)

        if texture is None:
            if not create_if_not_exist:
                return None
            else:
                texinfo = self._texture_info[name]
                texture = self.device.create_texture(
                    dimension="2d",
                    size=(*self.size, 1),
                    usage=texinfo["usage"],
                    format=texinfo["format"],
                )
            self._textures[name] = texture

        return texture.create_view(usage=usage)

    def _get_texture_view_for_rendering(self, name):
        """Get a texture view for the given name with rendert-attachment usage. Creates the texture if it does not exist."""
        return self.get_texture_view(
            name, wgpu.TextureUsage.RENDER_ATTACHMENT, create_if_not_exist=True
        )

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

    def get_color_descriptors(self, material_pick_write, alpha_config):
        """Get the dict color-descriptors that pipeline.py needs to create the render pipeline.

        Called per object when the pipeline is created.
        """
        bf, bo = wgpu.BlendFactor, wgpu.BlendOperation

        alpha_method = alpha_config["method"]

        # Get the color_blend mini-dict
        if alpha_method == "opaque":
            color_blend = {
                "alpha": blend_dict(bf.one, bf.zero, bo.add),
                "color": blend_dict(bf.one, bf.zero, bo.add),
            }
        elif alpha_method == "blended":
            # Note that color_constant and alpha_constant are not yet supported.
            # We'd need to call GPURenderPassEncoder.set_blend_constant(rgba), close to where we call GPURenderPassEncoder.set_viewport() in renderer.py
            color_blend = {
                "color": blend_dict(
                    alpha_config["color_src"],
                    alpha_config["color_dst"],
                    alpha_config.get("color_op", "add"),
                ),
                "alpha": blend_dict(
                    alpha_config["alpha_src"],
                    alpha_config["alpha_dst"],
                    alpha_config.get("alpha_op", "add"),
                ),
            }
        elif alpha_method == "stochastic":
            color_blend = {
                "alpha": blend_dict(bf.one, bf.zero, bo.add),
                "color": blend_dict(bf.one, bf.zero, bo.add),
            }
        elif alpha_method == "weighted":
            color_blend = None  # handled below
        else:
            raise RuntimeError(f"Unexpected alpha_method {alpha_method!r}")

        # Build target state dicts
        color_target_state = {
            "name": "color",
            "blend": color_blend,
            "write_mask": wgpu.ColorWrite.ALL,
        }
        target_states = [color_target_state]

        # More work for 'weighted' method5
        if alpha_method == "weighted":
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
                "load_op": wgpu.LoadOp.load,
                "store_op": wgpu.StoreOp.store,
            }
            reveal_attachment = {
                "name": "reveal",
                "resolve_target": None,
                "clear_value": reveal_clear_value,
                "load_op": wgpu.LoadOp.load,
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

    def get_shader_kwargs(self, material_pick_write, alpha_config):
        """Get the shader templating variables for the given alpha config."""

        alpha_method = alpha_config["method"]

        if alpha_method == "opaque":
            make_rgb = "color.rgb"
            if alpha_config["premultiply_alpha"]:
                make_rgb = "color.rgb * color.a"

            fragment_output_code = """
            struct FragmentOutput {
                // virtualfield stub: bool = false;
                @location(0) color: vec4<f32>,
                MAYBE_PICK@location(1) pick: vec4<u32>,
            };

            fn apply_virtual_fields_of_fragment_output(outp: ptr<function,FragmentOutput>, stub: bool) {
                let color = (*outp).color;
                let rgb = MAKE_RGB;
                (*outp).color = vec4f(rgb, 1.0);  // force alpha 1
            }
            """.replace("MAKE_RGB", make_rgb)

        elif alpha_method == "blended":
            fragment_output_code = """
            struct FragmentOutput {
                @location(0) color: vec4<f32>,
                MAYBE_PICK@location(1) pick: vec4<u32>,
            };
            """

        elif alpha_method == "stochastic":
            # We want the seed for the random function to be such that the
            # result is deterministic, so that rendered images can be visually
            # compared. We also (in general) want the seed to be stable for e.g.
            # movement of the mouse or object.
            #
            # So, let's *not* use:
            #
            # * The global-id: because it's determined by how many objects have
            #   been created earlier (in the current process).
            # * Object position, because then it will flicker when it moves.
            # * varyings.position.z, because then it will flicker when the
            #   camera moves.
            #
            # We can use these:
            #
            # * varyings.position.xy (the screen coord) is a common seed. It
            #   will give an effect often observed in games, where the pattern
            #   is "stuck to the screen".
            # * the renderer-id to distinguish objects while being reproducable
            #   in a scene.
            # * varyings.elementIndex to prevent two regions of the same object
            #   to get the same pattern, e.g. two points in a points object.

            seed = alpha_config.get("seed", "screen").lower()
            upos = "upos" + str({"screen": 0, "object": 1, "element": 2}[seed])
            pattern = alpha_config.get("pattern", "blue").lower().replace("_", "-")
            if pattern in ("blue-noise", "blue"):
                random_call = f"blueNoise2({upos})"
            elif pattern in ("white-noise", "white"):
                random_call = f"whiteNoise({upos})"
            elif pattern in ("bayer8", "bayer"):
                # Using upos2 becomes too noisy, but upos1 (i.e. including the per-object seed) seems quite alright!
                random_call = f"bayerPattern({upos})"
            else:
                raise ValueError(f"Unexpected stochastic pattern: {pattern!r}")

            fragment_output_code = load_wgsl("noise.wgsl")  # cannot use include here

            fragment_output_code += """
            struct FragmentOutput {
                // virtualfield position: vec3f = varyings.position.xyz;
                // virtualfield objectId: u32 = u_wobject.renderer_id;
                // virtualfield elementIndex: u32 = varyings.elementIndex;
                @location(0) color: vec4<f32>,
                MAYBE_PICK@location(1) pick: vec4<u 32>,
            };

            fn apply_virtual_fields_of_fragment_output(outp: ptr<function,FragmentOutput>, position: vec3f, objectId: u32, elementIndex: u32) {

                // Early exit
                let alpha = (*outp).color.a;
                if (alpha >= 1.0) { return; }

                let screenSize = u_stdinfo.physical_size.xy;

                // Compose seeds
                let seed1 = hashu(objectId);
                let seed2 = hashu(objectId) ^ hashu(elementIndex+1);

                // Compose positions
                let upos0 = vec2u(position.xy);
                let upos1 = upos0 + vec2u(seed1 >> 16, seed1 & 0xffff);
                let upos2 = upos0 + vec2u(seed2 >> 16, seed2 & 0xffff);

                // Generate a random number with a blue-noise distribution, i.e. resulting in uniformly sampled points with very little 'structure'
                // Blue noise has is great for sampling problems like this, because it has few low-frequency components, so the noise is very 'fine'.
                // The bayer pattern looks nicer than the noise, but is not suited for mixing multiple transparent layers.
                var rand = 0.0;

                if true {  // set to false to use split-screen debug mode
                    rand = RANDOM_CALL;
                } else if position.x < 0.5 * screenSize.x {
                    //rand = blueNoise2(upos2);
                    //rand = random(position.x * position.y * position.z);  // more or less original white noise version
                    rand = bayerPattern(upos1);
                } else {
                    rand = blueNoise2(upos2);
                }

                // Render or drop the fragment?
                if ( alpha < 1.0 - ALPHA_COMPARE_EPSILON && alpha < rand ) { discard; }
                (*outp).color.a = 1.0;  // fragments that pass through are opaque
            }
            """.replace("RANDOM_CALL", random_call)

        elif alpha_method == "weighted":
            use_alpha = "alpha", "use_alpha", "weighted_blending_use_alpha"
            weight_default = alpha_config.get("weight", "alpha")
            if weight_default in use_alpha:
                weight_default = "weighted_blending_use_alpha"
            alpha_default = alpha_config.get("alpha", "alpha")
            if alpha_default in use_alpha:
                alpha_default = "weighted_blending_use_alpha"

            fragment_output_code = """
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
            raise RuntimeError(f"Unexpected alpha_method {alpha_method!r}")

        # Enable/disable picking in the shader
        do_pick = material_pick_write and "pick" in self._texture_info
        fragment_output_code = fragment_output_code.replace(
            "MAYBE_PICK", "" if do_pick else "// "
        )

        return {
            "alpha_method": alpha_method,
            "fragment_output_code": fragment_output_code,
            "write_pick": do_pick,
        }

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

        wgsl = """
            @group(0) @binding(0)
            var r_accum: texture_2d<f32>;
            @group(0) @binding(1)
            var r_reveal: texture_2d<f32>;

            @fragment
            fn fs_main(varyings: Varyings) -> @location(0) vec4<f32> {
                let epsilon = 1e-6;
                let texIndex = vec2i(varyings.position.xy);

                // Sample
                let accum = textureLoad(r_accum, texIndex, 0).rgba;
                let reveal = textureLoad(r_reveal, texIndex, 0).r;

                // Exit if no transparent fragments was written
                // This discard does not brake early-z, because with weighted blending you're rendering each object anyway.
                if (reveal >= 1.0) { discard; }  // no transparent fragments here

                // Reconstruct the color and alpha, and set final color, with premultiplied alpha
                let avgColor = accum.rgb / max(accum.a, epsilon);
                let alpha = 1.0 - reveal;
                return vec4<f32>(avgColor * alpha, alpha);
            }
        """

        return create_full_quad_pipeline(targets, binding_layout, wgsl)

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
