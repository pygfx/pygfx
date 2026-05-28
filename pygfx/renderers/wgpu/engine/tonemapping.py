from .effectpasses import EffectPass
from ....utils.enums import ToneMappingMode


class ToneMappingPass(EffectPass):
    """An effect pass that applies neutral tone mapping to the image.

    This tone mapping algorithm provides a neutral color response while
    compressing the dynamic range of the image to displayable values.
    """

    REQUIRES_HDR = True

    uniform_type = dict(
        EffectPass.uniform_type,
        exposure="f4",
    )

    wgsl = """
        {$ include 'pygfx.tone_mapping.wgsl' $}

        @fragment
        fn fs_main(varyings: Varyings) -> @location(0) vec4<f32> {
            let texCoord = varyings.texCoord;
            var color = textureSample(colorTex, texSampler, texCoord);

            // Apply tone mapping to RGB channels
            return vec4<f32>(toneMapping(color.rgb * u_effect.exposure), color.a);
        }
    """

    def __init__(self, exposure=1.0, mode=ToneMappingMode.neutral):
        """Initialize the neutral tone mapping pass.

        Parameters
        ----------
        exposure : float, optional
            The exposure value to apply before tone mapping. Default is 1.0.
        mode : ToneMappingMode, optional
            The tone mapping mode to use. Default is ToneMappingMode.neutral.
        """
        super().__init__()
        self.exposure = exposure
        self.mode = mode

    @property
    def mode(self) -> ToneMappingMode:
        """The tone mapping type."""
        return self._template_vars["tone_mapping_mode"]

    @mode.setter
    def mode(self, mode: ToneMappingMode):
        if mode not in ToneMappingMode:
            raise ValueError(f"Invalid tone mapping mode: {mode}")
        self._set_template_var(tone_mapping_mode=mode)

    @property
    def exposure(self):
        """The exposure value applied before tone mapping."""
        return float(self._uniform_data["exposure"])

    @exposure.setter
    def exposure(self, exposure):
        self._uniform_data["exposure"] = float(exposure)
