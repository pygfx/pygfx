import wgpu
from .effectpasses import (
    EffectPass,
    FullQuadPass,
)
from .shared import get_shared
import numpy as np


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
                let luma = rgb_to_luminance(color);
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

                    downsample = max(downsample, vec3<f32>(0.0)); // Prevent NaNs
                    return vec4<f32>(downsample, 1.0);
                }

                // Standard weighted distribution for other mip levels
                // Apply weighted distribution:
                // 0.125*5 + 0.03125*4 + 0.0625*4 = 1
                var downsample = e * 0.125;
                downsample += (a + c + g + i) * 0.03125;
                downsample += (b + d + f + h) * 0.0625;
                downsample += (j + k + l + m) * 0.125;

                downsample = max(downsample, vec3<f32>(0.0)); // Prevent NaNs
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

        load_op = wgpu.LoadOp.load

        blend_op = {
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
        }

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

    def __init__(
        self,
        *,
        bloom_strength=0.04,
        max_mip_levels=6,
        filter_radius=0.005,
        use_karis_average=False,
    ):
        """
        Initialize the Physical Based Bloom Pass.

        Parameters:
        -----------
        bloom_strength : float, default 0.04
            The strength of the bloom effect. Lower values create more subtle bloom.
        max_mip_levels : int, default 6
            Max number of mip levels for downsampling/upsampling chain.
        filter_radius : float, default 0.005
            Filter radius for upsampling in texture coordinates.
        use_karis_average : bool, default True
            Whether to use Karis average on first downsample to prevent fireflies.
        """
        super().__init__()

        self._bloom_strength = bloom_strength
        self._max_mip_levels = max_mip_levels
        self._filter_radius = filter_radius
        self._use_karis_average = use_karis_average

        # Create internal passes
        self._downsample_pass = self._DownsamplePass()
        self._upsample_pass = self._UpsamplePass()
        self._composite_pass = self._CompositePass(self.bloom_strength)

        # Store mip chain data
        self._bloom_texture = None
        self._current_source_size = (0, 0)

    @property
    def bloom_strength(self):
        """The strength of the bloom effect."""
        return self._bloom_strength

    @bloom_strength.setter
    def bloom_strength(self, value):
        self._bloom_strength = float(value)

    @property
    def max_mip_levels(self):
        """Maximum number of mip levels in the bloom chain."""
        return self._max_mip_levels

    @max_mip_levels.setter
    def max_mip_levels(self, value):
        self._max_mip_levels = int(value)

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

    def _create_mip_bloom_texture(self, source_texture):
        """Create a mip chain texture for bloom processing."""

        device = get_shared().device

        # Get source dimensions
        source_size = source_texture.size
        bloom_texture_size = [source_size[0] // 2, source_size[1] // 2]
        max_mip_levels = int(
            np.floor(np.log2(max(bloom_texture_size[0], bloom_texture_size[1]))) + 1
        )

        mip_levels = min(self._max_mip_levels, max_mip_levels)

        # Create mip chain
        self._bloom_texture = device.create_texture(
            size=(bloom_texture_size[0], bloom_texture_size[1], 1),
            format=source_texture.format,
            usage=wgpu.TextureUsage.RENDER_ATTACHMENT
            | wgpu.TextureUsage.TEXTURE_BINDING,
            sample_count=1,  # todo: use multi-sampling
            mip_level_count=mip_levels,
            dimension=wgpu.TextureDimension.d2,
        )

    def _perform_downsampling(self, command_encoder, source_view):
        """Perform the downsampling phase."""
        # First downsample: from source to first mip
        self._downsample_pass._uniform_data["use_karis_average"] = (
            1 if self._use_karis_average else 0
        )

        # target_view = self._mip_textures[0].create_view()

        target_view = self._bloom_texture.create_view(
            base_mip_level=0,
            mip_level_count=1,
        )

        self._downsample_pass.render(
            command_encoder, colorTex=source_view, targetTex=target_view
        )

        # Subsequent downsamples: mip to mip
        for i in range(1, self._bloom_texture.mip_level_count):
            self._downsample_pass._uniform_data["use_karis_average"] = (
                0  # No Karis after first
            )

            source_view = target_view
            target_view = self._bloom_texture.create_view(
                base_mip_level=i,
                mip_level_count=1,
            )

            self._downsample_pass.render(
                command_encoder, colorTex=source_view, targetTex=target_view
            )

    def _perform_upsampling(self, command_encoder):
        """Perform the upsampling phase with accumulation."""
        self._upsample_pass._uniform_data["filter_radius"] = self._filter_radius

        # Work from smallest to largest, accumulating results
        target_view = self._bloom_texture.create_view(
            base_mip_level=self._bloom_texture.mip_level_count - 1,
            mip_level_count=1,
        )

        for i in range(self._bloom_texture.mip_level_count - 1, 0, -1):
            source_view = target_view
            target_view = self._bloom_texture.create_view(
                base_mip_level=i - 1,
                mip_level_count=1,
            )

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
        source_texture = color_tex.texture
        source_size = (source_texture.size[0], source_texture.size[1])

        if source_size != self._current_source_size or not self._bloom_texture:
            self._create_mip_bloom_texture(source_texture)
            self._current_source_size = source_size

        # Phase 1: Downsampling - create bloom mip chain
        self._perform_downsampling(command_encoder, color_tex)

        # Phase 2: Upsampling - accumulate bloom across mips
        self._perform_upsampling(command_encoder)

        # Phase 3: Final composition - blend original + bloom
        self._perform_final_composition(command_encoder, color_tex, target_tex)

    def _perform_final_composition(self, command_encoder, original_tex, target_tex):
        """Perform final composition: original + bloom."""
        # Update bloom strength uniform
        self._composite_pass._uniform_data["bloom_strength"] = float(
            self.bloom_strength
        )

        # Get bloom result from first mip texture
        bloom_view = self._bloom_texture.create_view(
            base_mip_level=0,
            mip_level_count=1,
        )

        # Render with both original and bloom textures
        self._composite_pass.render(
            command_encoder,
            originalTex=original_tex,
            bloomTex=bloom_view,
            targetTex=target_tex,
        )

    def __repr__(self):
        return f"<{self.__class__.__name__} strength={self.bloom_strength} mips={self.mip_levels} at {hex(id(self))}>"
