"""
Eye-Dome Lighting (EDL) post-processing pass for pygfx.

This pass is inspired by Potree's EDL implementation and
applies a depth-based shading in screen space to enhance the perception of
surfaces in dense point clouds or meshes.

References:
- Potree (EDL shader and technique): https://github.com/potree/potree
"""

from .effectpasses import EffectPass


class EDLPass(EffectPass):
    """Eye-Dome Lighting post-effect (Potree-like).

    Parameters
    ----------
    strength : float
        EDL strength. Typical range ~ [0.5, 10.0]. Default 1.0.
    radius : float
        Sampling radius in pixels. Typical range ~ [1.0, 3.0]. Default 1.5.
    num_samples : int
        Number of neighbor samples around the pixel. Typical values 4 or 8.
    depth_edge_threshold : float
        Threshold to gate the response for very small depth differences.
    """

    USES_DEPTH = True

    uniform_type = dict(
        EffectPass.uniform_type,
        strength="f4",
        radius="f4",
        depth_edge_threshold="f4",
    )

    wgsl = """
        const EDL_SCALE: f32 = 200.0; // scale to match Potree-like response levels

        @fragment
        fn fs_main(varyings: Varyings) -> @location(0) vec4<f32> {
            let texIndex = vec2i(varyings.position.xy);

            // Load center depth and color
            let depth_c = textureLoad(depthTex, texIndex, 0);
            let color_c = textureLoad(colorTex, texIndex, 0);

            // Early out if depth is invalid (background)
            if (depth_c >= 1.0) {
                return color_c;
            }

            // Neighbor sampling pattern (8 directions). We sample two rings: r and 2r (total 16 taps)
            // Currently shader is fixed at 8 taps; future: template var for 4 taps
            let dir = array<vec2<f32>, 8>(
                vec2<f32>(1.0, 0.0),
                vec2<f32>(0.70710678, 0.70710678),
                vec2<f32>(0.0, 1.0),
                vec2<f32>(-0.70710678, 0.70710678),
                vec2<f32>(-1.0, 0.0),
                vec2<f32>(-0.70710678, -0.70710678),
                vec2<f32>(0.0, -1.0),
                vec2<f32>(0.70710678, -0.70710678)
            );

            let dims = textureDimensions(depthTex);
            let max_i = vec2<i32>(i32(dims.x) - 1, i32(dims.y) - 1);

            var response: f32 = 0.0;
            // Ring 1 (radius r)
            for (var i: i32 = 0; i < 8; i = i + 1) {
                let off = vec2<i32>(
                    i32(round(dir[i].x * u_effect.radius)),
                    i32(round(dir[i].y * u_effect.radius)),
                );
                var ni = texIndex + off;
                ni = clamp(ni, vec2<i32>(0, 0), max_i);
                let depth_n = textureLoad(depthTex, ni, 0);
                let is_bg = depth_n >= 0.9999;
                if (!is_bg) {
                    // Positive when neighbor is further (darker)
                    let dd = depth_n - depth_c;
                    let contrib = max(0.0, dd - u_effect.depth_edge_threshold);
                    response = response + contrib;
                }
            }
            // Ring 2 (radius 2r), weighted lower
            for (var i: i32 = 0; i < 8; i = i + 1) {
                let off = vec2<i32>(
                    i32(round(dir[i].x * (u_effect.radius * 2.0))),
                    i32(round(dir[i].y * (u_effect.radius * 2.0))),
                );
                var ni = texIndex + off;
                ni = clamp(ni, vec2<i32>(0, 0), max_i);
                let depth_n = textureLoad(depthTex, ni, 0);
                let is_bg = depth_n >= 0.9999;
                if (!is_bg) {
                    let dd = depth_n - depth_c;
                    let contrib = max(0.0, dd - u_effect.depth_edge_threshold) * 0.5;
                    response = response + contrib;
                }
            }

            // Normalize and map response to a dimming factor
            // scale with EDL_SCALE for visible effect
            let shade = exp(-u_effect.strength * response * EDL_SCALE);
            let shaded_rgb = color_c.rgb * shade;
            return vec4<f32>(shaded_rgb, color_c.a);
        }
    """

    def __init__(self, *, strength=1.0, radius=1.5, depth_edge_threshold=0.0):
        super().__init__()
        self.strength = strength
        self.radius = radius
        self.depth_edge_threshold = depth_edge_threshold

    # Properties
    @property
    def strength(self):
        return float(self._uniform_data["strength"])

    @strength.setter
    def strength(self, value):
        self._uniform_data["strength"] = float(value)

    @property
    def radius(self):
        return float(self._uniform_data["radius"])

    @radius.setter
    def radius(self, value):
        self._uniform_data["radius"] = float(value)

    @property
    def depth_edge_threshold(self):
        return float(self._uniform_data["depth_edge_threshold"])

    @depth_edge_threshold.setter
    def depth_edge_threshold(self, value):
        self._uniform_data["depth_edge_threshold"] = float(value)
