import wgpu
from .effectpasses import (
    EffectPass,
    FullQuadPass,
    CopyPass,
    create_full_quad_pipeline,
    apply_templating,
)
from .shared import get_shared


class PhysicalBasedBloomPass(EffectPass):
    """
    Physically-based bloom effect pass based on the technique from Call of Duty: Advanced Warfare.

    This implementation uses progressive downsampling and upsampling without thresholding,
    making it suitable for HDR rendering pipelines. The algorithm:

    1. Progressively downsamples the HDR buffer using a 13-tap filter
    2. Applies 3x3 tent filter during upsampling
    3. Accumulates bloom across all mip levels
    4. Produces natural-looking bloom that affects the entire scene

    Reference: https://learnopengl.com/Guest-Articles/2022/Phys.-Based-Bloom

    Parameters
    ----------
    bloom_strength : float
        Strength of the bloom effect. Default 0.04.
    mip_levels : int
        Max number of mip levels for downsampling/upsampling chain. Default 5.
    filter_radius : float
        Filter radius for upsampling in texture coordinates. Default 0.005.
    use_karis_average : bool
        Whether to use Karis average for the first downsample pass. Default false.
    """

    uniform_type = dict(
        EffectPass.uniform_type,
        bloom_strength="f4",
    )

    class _DownsamplePass(FullQuadPass):
        """Internal downsampling pass using 13-tap filter."""

        uniform_type = dict(
            use_karis_average="i4",
        )

        wgsl = """
            // Utility functions for Karis average (prevent fireflies)
            fn rgb_to_luminance(color: vec3<f32>) -> f32 {
                return dot(color, vec3<f32>(0.2126, 0.7152, 0.0722));
            }

            fn karis_average(color: vec3<f32>) -> f32 {
                let luma = rgb_to_luminance(color) * 0.25;
                return 1.0 / (1.0 + luma);
            }

            @fragment
            fn fs_main(varyings: Varyings) -> @location(0) vec4<f32> {
                let coord = varyings.texCoord;

                // Take 13 samples around current texel:
                // a - b - c
                // - j - k -
                // d - e - f
                // - l - m -
                // g - h - i
                // === ('e' is the current texel) ===

                let a = textureSample(colorTex, texSampler, coord, vec2i(-2, 2)).rgb;
                let b = textureSample(colorTex, texSampler, coord, vec2i(0, 2)).rgb;
                let c = textureSample(colorTex, texSampler, coord, vec2i(2, 2)).rgb;

                let d = textureSample(colorTex, texSampler, coord, vec2i(-2, 0)).rgb;
                let e = textureSample(colorTex, texSampler, coord).rgb;
                let f = textureSample(colorTex, texSampler, coord, vec2i(2, 0)).rgb;

                let g = textureSample(colorTex, texSampler, coord, vec2i(-2, -2)).rgb;
                let h = textureSample(colorTex, texSampler, coord, vec2i(0, -2)).rgb;
                let i = textureSample(colorTex, texSampler, coord, vec2i(2, -2)).rgb;

                let j = textureSample(colorTex, texSampler, coord, vec2i(-1, 1)).rgb;
                let k = textureSample(colorTex, texSampler, coord, vec2i(1, 1)).rgb;
                let l = textureSample(colorTex, texSampler, coord, vec2i(-1, -1)).rgb;
                let m = textureSample(colorTex, texSampler, coord, vec2i(1, -1)).rgb;

                // Apply Karis average for the first downsample to prevent fireflies
                if (u_effect.use_karis_average != 0) {
                    // Calculate 5 groups with proper weights
                    var groups: array<vec3<f32>, 5>;

                    groups[0] = (a + b + d + e) / 4.0;
                    groups[1] = (b + c + e + f) / 4.0;
                    groups[2] = (d + e + g + h) / 4.0;
                    groups[3] = (e + f + h + i) / 4.0;
                    groups[4] = (j + k + l + m) / 4.0;

                    let kw0 = karis_average(groups[0]);
                    let kw1 = karis_average(groups[1]);
                    let kw2 = karis_average(groups[2]);
                    let kw3 = karis_average(groups[3]);
                    let kw4 = karis_average(groups[4]);

                    var downsample = groups[0] * kw0 + groups[1] * kw1 + groups[2] * kw2 + groups[3] * kw3 + groups[4] * kw4;
                    downsample /= (kw0 + kw1 + kw2 + kw3 + kw4);

                    downsample = max(downsample, vec3<f32>(0.0001)); // Prevent pure black
                    return vec4<f32>(downsample, 1.0);
                }

                // Standard weighted distribution for other mip levels
                // Apply weighted distribution:
                // 0.125*5 + 0.03125*4 + 0.0625*4 = 1
                var downsample = e * 0.125;
                downsample += (a + c + g + i) * 0.03125;
                downsample += (b + d + f + h) * 0.0625;
                downsample += (j + k + l + m) * 0.125;

                downsample = max(downsample, vec3<f32>(0.0001)); // Prevent pure black
                return vec4<f32>(downsample, 1.0);
            }
        """

    class _UpsamplePass(FullQuadPass):
        """Internal upsampling pass using 3x3 tent filter."""

        uniform_type = dict(
            filter_radius="f4",
        )

        wgsl = """
            @fragment
            fn fs_main(varyings: Varyings) -> @location(0) vec4<f32> {
                let x = u_effect.filter_radius;
                let aspect = f32(textureDimensions(colorTex).x) / f32(textureDimensions(colorTex).y);
                let y = u_effect.filter_radius * aspect;
                let coord = varyings.texCoord;

                // Take 9 samples around current texel using tent filter:
                // a - b - c
                // d - e - f
                // g - h - i
                // === ('e' is the current texel) ===

                let a = textureSample(colorTex, texSampler, coord + vec2<f32>(-x, y)).rgb;
                let b = textureSample(colorTex, texSampler, coord + vec2<f32>(0.0, y)).rgb;
                let c = textureSample(colorTex, texSampler, coord + vec2<f32>(x, y)).rgb;

                let d = textureSample(colorTex, texSampler, coord + vec2<f32>(-x, 0.0)).rgb;
                let e = textureSample(colorTex, texSampler, coord).rgb;
                let f = textureSample(colorTex, texSampler, coord + vec2<f32>(x, 0.0)).rgb;

                let g = textureSample(colorTex, texSampler, coord + vec2<f32>(-x, -y)).rgb;
                let h = textureSample(colorTex, texSampler, coord + vec2<f32>(0.0, -y)).rgb;
                let i = textureSample(colorTex, texSampler, coord + vec2<f32>(x, -y)).rgb;

                // Apply weighted distribution using 3x3 tent filter:
                //  1   | 1 2 1 |
                // -- * | 2 4 2 |
                // 16   | 1 2 1 |
                var upsample = e * 4.0;
                upsample += (b + d + f + h) * 2.0;
                upsample += (a + c + g + i);
                upsample *= (1.0 / 16.0);

                // Ensure we don't get negative values that could cause issues
                upsample = max(upsample, vec3<f32>(0.0));

                return vec4<f32>(upsample, 1.0);
            }
        """

        def _render(self, command_encoder, source_textures, target_textures):
            # Create bind group. This is very light and can be done every time.
            # Chances are we get new views on every call anyway.
            bind_group_entries = [
                self._uniform_binding_entry,
                self._sampler_binding_entry,
            ]
            for i, tex in enumerate(source_textures, 2):
                bind_group_entries.append({"binding": i, "resource": tex})
            bind_group = self._device.create_bind_group(
                layout=self._render_pipeline.get_bind_group_layout(0),
                entries=bind_group_entries,
            )

            # Create attachments
            color_attachments = []
            for tex in target_textures:
                color_attachments.append(
                    {
                        "view": tex,
                        "resolve_target": None,
                        "clear_value": (0, 0, 0, 0),
                        "load_op": wgpu.LoadOp.load,
                        "store_op": wgpu.StoreOp.store,
                    }
                )

            render_pass = command_encoder.begin_render_pass(
                color_attachments=color_attachments,
                depth_stencil_attachment=None,
            )
            render_pass.set_pipeline(self._render_pipeline)
            render_pass.set_bind_group(0, bind_group, [], 0, 99)
            render_pass.draw(4, 1)
            render_pass.end()

        def _create_pipeline(self, source_names, target_formats):
            binding_layout = []
            definitions_code = ""

            # Uniform buffer
            binding_layout.append(
                {
                    "binding": 0,
                    "visibility": wgpu.ShaderStage.FRAGMENT,
                    "buffer": {"type": wgpu.BufferBindingType.uniform},
                }
            )
            definitions_code += self._uniform_binding_definition

            # Sampler
            binding_layout.append(
                {
                    "binding": 1,
                    "visibility": wgpu.ShaderStage.FRAGMENT,
                    "sampler": {"type": wgpu.SamplerBindingType.filtering},
                }
            )
            definitions_code += self._sampler_binding_definition

            # Source textures
            for i, name in enumerate(source_names, 2):
                sample_type = wgpu.TextureSampleType.float
                wgsl_type = "texture_2d<f32>"
                if "depth" in name.lower():
                    sample_type = wgpu.TextureSampleType.depth
                    wgsl_type = "texture_depth_2d"
                binding_layout.append(
                    {
                        "binding": i,
                        "visibility": wgpu.ShaderStage.FRAGMENT,
                        "texture": {
                            "sample_type": sample_type,
                            "view_dimension": wgpu.TextureViewDimension.d2,
                            "multisampled": False,
                        },
                    }
                )
                definitions_code += f"""
                    @group(0) @binding({i})
                    var {name}: {wgsl_type};
                """

            # Render targets
            targets = []
            for format in target_formats:
                targets.append(
                    {
                        "format": format,
                        "blend": {
                            "alpha": {
                                "operation": wgpu.BlendOperation.add,
                                "src_factor": wgpu.BlendFactor.one,
                                "dst_factor": wgpu.BlendFactor.zero,
                            },
                            "color": {
                                "operation": wgpu.BlendOperation.add,
                                "src_factor": wgpu.BlendFactor.one,
                                "dst_factor": wgpu.BlendFactor.one,
                            },
                        },
                    }
                )

            wgsl = definitions_code
            wgsl += apply_templating(self.wgsl, **self._template_vars)
            return create_full_quad_pipeline(targets, binding_layout, wgsl)

    def __init__(
        self,
        *,
        bloom_strength=0.04,
        mip_levels=5,
        filter_radius=0.005,
        use_karis_average=False,
    ):
        """
        Initialize the Physical Based Bloom Pass.

        Parameters:
        -----------
        bloom_strength : float, default 0.04
            The strength of the bloom effect. Lower values create more subtle bloom.
        mip_levels : int, default 5
            Max number of mip levels for downsampling/upsampling chain.
        filter_radius : float, default 0.005
            Filter radius for upsampling in texture coordinates.
        use_karis_average : bool, default True
            Whether to use Karis average on first downsample to prevent fireflies.
        """
        super().__init__()

        self._bloom_strength = bloom_strength
        self._mip_levels = max(1, min(mip_levels, 10))  # Clamp between 1-10
        self._filter_radius = filter_radius
        self._use_karis_average = use_karis_average

        # Initialize uniform data
        self._uniform_data["bloom_strength"] = float(bloom_strength)

        # Create internal passes
        self._downsample_pass = self._DownsamplePass()
        self._upsample_pass = self._UpsamplePass()
        self._composite_pass = self._CompositePass(self.bloom_strength)

        # Store mip chain data
        self._mip_textures = []
        self._current_source_size = (0, 0)

    @property
    def bloom_strength(self):
        """The strength of the bloom effect."""
        return float(self._uniform_data["bloom_strength"])

    @bloom_strength.setter
    def bloom_strength(self, value):
        self._uniform_data["bloom_strength"] = float(value)
        self._bloom_strength = float(value)

    @property
    def mip_levels(self):
        """Number of mip levels in the bloom chain."""
        return self._mip_levels

    @mip_levels.setter
    def mip_levels(self, value):
        self._mip_levels = max(1, min(int(value), 10))

    @property
    def filter_radius(self):
        """Filter radius for upsampling."""
        return self._filter_radius

    @filter_radius.setter
    def filter_radius(self, value):
        self._filter_radius = float(value)

    @property
    def use_karis_average(self):
        """Whether to use Karis average to prevent fireflies."""
        return self._use_karis_average

    @use_karis_average.setter
    def use_karis_average(self, value):
        self._use_karis_average = bool(value)

    def _create_mip_textures(self, source_texture):
        """Create mip chain textures for bloom processing."""
        device = get_shared().device

        # Clean up old textures
        self._mip_textures.clear()

        # Get source dimensions
        source_size = source_texture.size
        current_size = [source_size[0], source_size[1]]

        # Create mip chain
        for _ in range(self._mip_levels):
            # Halve the size for each mip level
            current_size[0] = max(1, current_size[0] // 2)
            current_size[1] = max(1, current_size[1] // 2)

            # Create texture for this mip level
            mip_texture = device.create_texture(
                size=(current_size[0], current_size[1], 1),
                format=source_texture.format,
                usage=wgpu.TextureUsage.RENDER_ATTACHMENT
                | wgpu.TextureUsage.TEXTURE_BINDING,
                sample_count=1,
                mip_level_count=1,
                dimension=wgpu.TextureDimension.d2,
            )

            self._mip_textures.append(mip_texture)

            # Stop if we reach 1x1
            if current_size[0] == 1 and current_size[1] == 1:
                break

    def _perform_downsampling(self, command_encoder, source_texture):
        """Perform the downsampling phase."""
        # First downsample: from source to first mip
        # self._downsample_pass._uniform_data["src_resolution"] = [source_texture.size[0], source_texture.size[1]]
        # self._downsample_pass._uniform_data["mip_level"] = 0
        self._downsample_pass._uniform_data["use_karis_average"] = int(
            self._use_karis_average
        )

        source_view = (
            source_texture.create_view()
            if hasattr(source_texture, "create_view")
            else source_texture
        )
        target_view = self._mip_textures[0].create_view()

        self._downsample_pass.render(
            command_encoder, colorTex=source_view, targetTex=target_view
        )

        # Subsequent downsamples: mip to mip
        for i in range(1, len(self._mip_textures)):
            # self._downsample_pass._uniform_data["src_resolution"] = [
            #     self._mip_textures[i-1].size[0],
            #     self._mip_textures[i-1].size[1]
            # ]
            # self._downsample_pass._uniform_data["mip_level"] = i
            self._downsample_pass._uniform_data["use_karis_average"] = (
                0  # Only first mip uses Karis
            )

            source_view = self._mip_textures[i - 1].create_view()
            target_view = self._mip_textures[i].create_view()

            self._downsample_pass.render(
                command_encoder, colorTex=source_view, targetTex=target_view
            )

    def _perform_upsampling(self, command_encoder):
        """Perform the upsampling phase with accumulation."""
        self._upsample_pass._uniform_data["filter_radius"] = self._filter_radius

        # For proper physical-based bloom, we need to accumulate all mip levels
        # Start with the smallest mip as base
        if len(self._mip_textures) < 2:
            return

        # Create a temporary texture for accumulation if needed

        # Work from smallest to largest, accumulating results
        for i in range(len(self._mip_textures) - 1, 0, -1):
            source_mip = self._mip_textures[i]  # Smaller mip (source)
            target_mip = self._mip_textures[i - 1]  # Larger mip (target)

            source_view = source_mip.create_view()
            target_view = target_mip.create_view()

            # For now, use simple upsampling. In a complete implementation,
            # we would use additive blending to accumulate with existing content
            self._upsample_pass.render(
                command_encoder, colorTex=source_view, targetTex=target_view
            )

    def render(self, command_encoder, color_tex, depth_tex, target_tex):
        """
        Render the physical-based bloom effect using standard EffectPass interface.

        Parameters:
        -----------
        command_encoder : wgpu.GPUCommandEncoder
            The command encoder for GPU commands
        color_tex : wgpu.GPUTextureView
            The HDR color texture to apply bloom to
        depth_tex : wgpu.GPUTextureView
            The depth texture (ignored for bloom)
        target_tex : wgpu.GPUTextureView
            The target texture to render the bloom result to
        """
        # Ensure we have the right mip textures
        source_texture = (
            color_tex.texture if hasattr(color_tex, "texture") else color_tex
        )
        source_size = (source_texture.size[0], source_texture.size[1])

        if source_size != self._current_source_size or not self._mip_textures:
            self._create_mip_textures(source_texture)
            self._current_source_size = source_size

        if not self._mip_textures:
            # Fallback: just copy the original image
            self._render_copy_fallback(command_encoder, color_tex, target_tex)
            return

        # Phase 1: Downsampling - create bloom mip chain
        self._perform_downsampling(command_encoder, source_texture)

        # Phase 2: Upsampling - accumulate bloom across mips
        self._perform_upsampling(command_encoder)

        # Phase 3: Final composition - blend original + bloom
        self._perform_final_composition(command_encoder, color_tex, target_tex)

    def _render_copy_fallback(self, command_encoder, color_tex, target_tex):
        """Fallback to simply copy the original image when no mip textures."""
        # Create a simple copy pass
        copy_pass = CopyPass()
        copy_pass.render(command_encoder, color_tex, None, target_tex)

    def _perform_final_composition(self, command_encoder, original_tex, target_tex):
        """Perform final composition: original + bloom."""
        # Update bloom strength uniform
        self._composite_pass._uniform_data["bloom_strength"] = float(
            self.bloom_strength
        )

        # Get bloom result from first mip texture
        bloom_view = self._mip_textures[0].create_view()

        # Render with both original and bloom textures
        self._composite_pass.render(
            command_encoder,
            originalTex=original_tex,
            bloomTex=bloom_view,
            targetTex=target_tex,
        )

    class _CompositePass(FullQuadPass):
        """Internal composite pass for blending original + bloom."""

        uniform_type = dict(
            bloom_strength="f4",
        )

        wgsl = """
            @fragment
            fn fs_main(varyings: Varyings) -> @location(0) vec4<f32> {
                let original = textureSample(originalTex, texSampler, varyings.texCoord);
                let bloom = textureSample(bloomTex, texSampler, varyings.texCoord);

                // Mix original and bloom based on strength
                let result = mix(original, bloom, u_effect.bloom_strength);

                return vec4<f32>(result.rgb, original.a);
            }
        """

        def __init__(self, bloom_strength):
            super().__init__()
            self._uniform_data["bloom_strength"] = float(bloom_strength)

    def __repr__(self):
        return f"<{self.__class__.__name__} strength={self.bloom_strength} mips={self.mip_levels} at {hex(id(self))}>"
