// ddaa2.wgsl version 2.3
//
// Directional Diffusion Anti Aliasing (DDAA) version 2
//
// Home: https://github.com/almarklein/ppaa-experiments/blob/main/ddaa.md
// Source: https://github.com/almarklein/ppaa-experiments/blob/main/wgsl/ddaa2.wgsl
//
//
// Summary:
//
// Smooth along the edges based on Scharr kernel, and perform edge-search to
// better support horizontal/vertical edges. It this combines a strategy that is
// good at near-diagonal edges (diffusion) with a strategie that is good at
// near-horizontal / near-vertical edges (edge-search).
//
//
// Changelog:
//
// v0.0 (2013): Original: https://github.com/vispy/experimental/blob/master/fsaa/ddaa.glsl
// v1.0 (2025): Ported to wgsl and tweaked: https://github.com/almarklein/ppaa-experiments/blob/main/wgsl/ddaa1.wgsl
// v2.1 (2025): Added edge search (2025): https://github.com/almarklein/ppaa-experiments/blob/main/wgsl/ddaa2.wgsl
// v2.2 (2025): Made SAMPLES_PER_STEP configurable, and fixed a little sampling bug causing an asymetry.
// v2.3 (2025): Configure edge search with EDGE_STEP_LIST, optimized sample batching, and get one sample for free.


// ========== CONFIG ==========

// The edge search happens in steps, where in each step multiple samples are
// taken. Batching samples this way  helps performance because the texture
// queries can be performed in parallel, to a certain degree.
//
// We define the list that specifies the number of samples per step. The maximum
// distance that the algorithm can track along the egde is the sum of this list.
// If this list is zero, no edge-search is performed, effectively ddaa1.
//
// Note that for the first step, the steps for both directions are combined, and
// get one sample for each direction for free, so the actual number of samples
// for the first edge-search step is EDGE_STEP_LIST[0] * 2 - 2.
//
$$ if EDGE_STEP_LIST is not defined
$$ set EDGE_STEP_LIST = [3, 3, 3, 3, 3]
$$ endif
$$ set MAX_EDGE_STEP = EDGE_STEP_LIST | max
$$ set MAX_EDGE_ITERS = EDGE_STEP_LIST | length
// Note that the the int pixel offset in textureSampleLevel should be [-8..7],
// so the items in EDGE_STEP_LIST should be <= 14, and the first item <= 7.
$$ if EDGE_STEP_LIST[0] > 7
{{'woops_first_element_in_EDGE_STEP_LIST_must_be_no_larger_than_7'}}
$$endif
$$ if MAX_EDGE_STEP > 14
{{'woops_elements_in_EDGE_STEP_LIST_must_be_no_larger_than_14'}}
$$endif
//
// The templated EDGE_STEP_LIST = {{EDGE_STEP_LIST}}
const MAX_EDGE_ITERS = {{ MAX_EDGE_ITERS }};  // length(EDGE_STEP_LIST)

// The strength of the diffusion. A value of 3 seems to work well.
$$ if DDAA_STRENGTH is not defined
$$ set DDAA_STRENGTH = 3.0
$$ endif
const DDAA_STRENGTH : f32 = {{ DDAA_STRENGTH }};

// Trims the algorithm from processing darks.
// low: 0.0833, medium: 0.0625, high: 0.0312, ultra: 0.0156, extreme: 0.0078
$$ if EDGE_THRESHOLD_MIN is not defined
$$ set EDGE_THRESHOLD_MIN = 0.0625
$$ endif
const EDGE_THRESHOLD_MIN : f32 = {{ EDGE_THRESHOLD_MIN }};

// The minimum amount of local contrast required to apply algorithm.
// low: 0.250, medium: 0.166, high: 0.125, ultra: 0.063, extreme: 0.031
$$ if EDGE_THRESHOLD_MAX is not defined
$$ set EDGE_THRESHOLD_MAX = 0.166
$$ endif
const EDGE_THRESHOLD_MAX : f32 = {{ EDGE_THRESHOLD_MAX }};


// ========== Constants and helper functions ==========

const sqrt2  = sqrt(2.0);

fn rgb2luma(rgb: vec3f) -> f32 {
    return sqrt(dot(rgb, vec3f(0.299, 0.587, 0.114)));  // trick for perceived lightness, used in Bevy
    // return dot(rgb, vec3f(0.299, 0.587, 0.114));  // real luma
}


@fragment
fn fs_main(varyings: Varyings) -> @location(0) vec4<f32> {

    let tex: texture_2d<f32> = colorTex;
    let smp: sampler = texSampler;
    let texCoord: vec2f = varyings.texCoord;

    let resolution = vec2f(textureDimensions(tex));
    let pixelStep = 1.0 / resolution.xy;

    // Sample the center pixel
    let centerSample = textureSampleLevel(tex, smp, texCoord, 0.0);
    let lumaCenter = rgb2luma(centerSample.rgb);

    // Luma at the four direct neighbors of the current fragment.
    let lumaN = rgb2luma(textureSampleLevel(tex, smp, texCoord, 0.0, vec2i(0, 1)).rgb);
    let lumaE = rgb2luma(textureSampleLevel(tex, smp, texCoord, 0.0, vec2i(1, 0)).rgb);
    let lumaS = rgb2luma(textureSampleLevel(tex, smp, texCoord, 0.0, vec2i(0, -1)).rgb);
    let lumaW = rgb2luma(textureSampleLevel(tex, smp, texCoord, 0.0, vec2i(-1, 0)).rgb);

    // Query the 4 remaining corners lumas.
    let lumaNW = rgb2luma(textureSampleLevel(tex, smp, texCoord, 0.0, vec2i(-1, 1)).rgb);
    let lumaNE = rgb2luma(textureSampleLevel(tex, smp, texCoord, 0.0, vec2i(1, 1)).rgb);
    let lumaSW = rgb2luma(textureSampleLevel(tex, smp, texCoord, 0.0, vec2i(-1, -1)).rgb);
    let lumaSE = rgb2luma(textureSampleLevel(tex, smp, texCoord, 0.0, vec2i(1, -1)).rgb);

    // Compute the range
    let lumaMin = min(lumaCenter, min(min(lumaS, lumaN), min(lumaW, lumaE)));
    let lumaMax = max(lumaCenter, max(max(lumaS, lumaN), max(lumaW, lumaE)));
    let lumaRange = lumaMax - lumaMin;

    // If the luma variation is lower that a threshold (or if we are in a really dark area), we are not on an edge, don't perform any AA.
    if (lumaRange < max(EDGE_THRESHOLD_MIN, lumaMax * EDGE_THRESHOLD_MAX)) {
        return centerSample;
    }

    // Combine the four edges lumas (using intermediary variables for future computations with the same values).
    let lumaSUp = lumaS + lumaN;
    let lumaWRight = lumaW + lumaE;
    let lumaWCorners = lumaSW + lumaNW;

    // Same for corners
    let lumaSCorners = lumaSW + lumaSE;
    let lumaECorners = lumaSE + lumaNE;
    let lumaNCorners = lumaNE + lumaNW;

    // Calculate the image gradient using the Schar kernel, which is (relatively) rotationally invariant.
    const k1 = 162.0 / 256.0; // 61
    const k2 = 47.0 / 256.0; // 17
    let imDx = (lumaW * k1 + lumaSW * k2 + lumaNW * k2) - (lumaE * k1 + lumaSE * k2 + lumaNE * k2);
    let imDy = (lumaS * k1 + lumaSW * k2 + lumaSE * k2) - (lumaN * k1 + lumaNW * k2 + lumaNE * k2);

    // Get the edge vector (orthogonal to the gradient), and calculate strength and direction.
    let edgeVector = vec2f(-imDy, imDx);
    var diffuseStrength = sqrt(length(edgeVector)) * DDAA_STRENGTH;
    var diffuseDirection = normalize(edgeVector);
    if diffuseStrength < 1e-6 {
        diffuseDirection = vec2f(0.0, 0.0);
        diffuseStrength = 0.0;
    }
    diffuseStrength = min(1.0, diffuseStrength);

    // Is the local edge horizontal or vertical ?
    let edgeHorizontal = abs(-2.0 * lumaW + lumaWCorners) + abs(-2.0 * lumaCenter + lumaSUp) * 2.0 + abs(-2.0 * lumaE + lumaECorners);
    let edgeVertical = abs(-2.0 * lumaN + lumaNCorners) + abs(-2.0 * lumaCenter + lumaWRight) * 2.0 + abs(-2.0 * lumaS + lumaSCorners);
    let isHorizontal = (edgeHorizontal >= edgeVertical);
    //let isHorizontal = (abs(diffuseDirection.x) >= abs(diffuseDirection.y)); -> different, resulting in wrong ridge detection

    // Calculate gradient on both sides of the current pixel
    var luma1 = select(lumaW, lumaS, isHorizontal);
    var luma2 = select(lumaE, lumaN, isHorizontal);    // Compute gradients in this direction.
    let gradient1 = luma1 - lumaCenter;
    let gradient2 = luma2 - lumaCenter;

    // Maintain ridges and thin lines. This is inspired by AXAA's 2nd enhancement, except we also apply it to negative edges and do a smooth transition instead of a threshold.
    // Note that we can diminish quite hard, because the neighbouring pixels likely get diffused in the direction of the edge (this is one of our advantages over fxaa).
    if sign(gradient1) == sign(gradient2) {
        // This is a ridge or a valley, e.g. a thin line. We want to presereve these.
        let ridgeness = min(abs(gradient1), abs(gradient2));
        let diminish_factor = 1.0 - (min(1.0, 10 * ridgeness));
        diffuseStrength *= diminish_factor;
    }

    // For long edges, the diffusion has to be huge to remove the jaggies. What algorithms like FXAA do instead, is detect
    // the length of the edge segment (successive horizontal/vertical pixels), and the use that to calculate the subpixel
    // texture coordinate offset, perpendicular to the edge. So technically this is diffusion perpendicular to the edge,
    // but in a controlled manner to smooth the step/jaggy.
    var subpixelEdgeOffset = vec2f(0.0);
    if (MAX_EDGE_ITERS > 0) {

        // Choose the step size (one pixel) accordingly.
        var stepLength = select(pixelStep.x, pixelStep.y, isHorizontal);

        // Gradient in the corresponding direction, normalized.
        let gradientScaled = 0.25 * max(abs(gradient1), abs(gradient2));

        // Average luma in the current direction.
        var lumaLocalAverage = 0.0;
        let gradient2IsHigher = abs(gradient2) > abs(gradient1);
        if  gradient2IsHigher {
            lumaLocalAverage = 0.5 * (luma2 + lumaCenter);
        } else {
            stepLength = -stepLength;  // switch the direction
            lumaLocalAverage = 0.5 * (luma1 + lumaCenter);
        }

        // Shift UV in the correct direction by half a pixel (orthogonal to the edge)
        var currentUv = texCoord;
        if isHorizontal {
            currentUv.y += stepLength * 0.5;
        } else {
            currentUv.x += stepLength * 0.5;
        }

        // Initialize edge search params

        var distance1 = 999.0;
        var distance2 = 999.0;

        var lumaEnd1: f32;
        var lumaEnd2: f32;

        $$ set N_LUMA_VARS = EDGE_STEP_LIST | max
        $$ set N_LUMA_VARS = [N_LUMA_VARS, EDGE_STEP_LIST[0] * 2] | max
        $$ for si in range(0, N_LUMA_VARS)
        var lumaEnd_{{si}}: f32;
        $$ endfor
        //

        $$ set ns = namespace(stepOffset=0, edgeSteps=0)
        $$ set ns.edgeSteps = EDGE_STEP_LIST[0]

        // Step 0 ({{ ns.edgeSteps }} steps)

        // Read the lumas at both current extremities of the exploration segment, and compute the delta wrt to the local average luma.
        // We take {{ ns.edgeSteps }} steps in both directions, but we get one sample for free (for each direction), so we take {{ ns.edgeSteps * 2 - 2 }} samples.
        if isHorizontal {
            lumaEnd_0 = 0.5 * (lumaW + select(lumaSW, lumaNW, gradient2IsHigher)) - lumaLocalAverage;
            lumaEnd_{{ns.edgeSteps}} = 0.5 * (lumaE + select(lumaSE, lumaNE, gradient2IsHigher)) - lumaLocalAverage;
            $$ for si in range(1, ns.edgeSteps)
            lumaEnd_{{ si }} = rgb2luma(textureSampleLevel(tex, smp, currentUv, 0.0, -vec2i({{ si + 1 }}, 0)).rgb) - lumaLocalAverage;
            $$ endfor
            $$ for si in range(1+ns.edgeSteps, ns.edgeSteps*2)
            lumaEnd_{{ si }} = rgb2luma(textureSampleLevel(tex, smp, currentUv, 0.0, vec2i({{ si - ns.edgeSteps + 1 }}, 0)).rgb) - lumaLocalAverage;
            $$ endfor
        } else {
            lumaEnd_0 = 0.5 * (lumaS + select(lumaSW, lumaSE, gradient2IsHigher)) - lumaLocalAverage;
            lumaEnd_{{ns.edgeSteps}} = 0.5 * (lumaN + select(lumaNW, lumaNE, gradient2IsHigher)) - lumaLocalAverage;
            $$ for si in range(1, ns.edgeSteps)
            lumaEnd_{{ si }} = rgb2luma(textureSampleLevel(tex, smp, currentUv, 0.0, -vec2i(0, {{ si + 1 }})).rgb) - lumaLocalAverage;
            $$ endfor
            $$ for si in range(1+ns.edgeSteps, ns.edgeSteps*2)
            lumaEnd_{{ si }} = rgb2luma(textureSampleLevel(tex, smp, currentUv, 0.0, vec2i(0, {{ si - ns.edgeSteps + 1 }})).rgb) - lumaLocalAverage;
            $$ endfor
        }

        // Search for left endpoint in the current {{ ns.edgeSteps }} samples
        $$ for si in range(0, ns.edgeSteps) | reverse
        if (abs(lumaEnd_{{si}}) >= gradientScaled) { distance1 = {{si+1}}.0; lumaEnd1 = lumaEnd_{{si}}; }
        $$ endfor

        // Same for the right endpoint
        $$ for si in range(ns.edgeSteps, ns.edgeSteps*2) | reverse
        if (abs(lumaEnd_{{si}}) >= gradientScaled) { distance2 = {{si-ns.edgeSteps+1}}.0; lumaEnd2 = lumaEnd_{{si}}; }
        $$ endfor
        //

        // Now search for endpoints in a series of rounds, using a templated (i.e. unrolled) loop.
        // This is much faster (in WGSL) than a normal loop, probably due to optimization related to the texture lookups.
        // We use textureSampleLevel with the offset parameter to help the gpu compiler batch the samples.

        var max_distance = {{ns.edgeSteps}}.0;

        $$ for iter in range(1, MAX_EDGE_ITERS)
        $$ set ns.edgeSteps = EDGE_STEP_LIST[iter]
        $$ set ns.stepOffset = ns.stepOffset + EDGE_STEP_LIST[iter-1]

        // Step {{ iter }} ({{ ns.edgeSteps }} steps, offset {{ ns.stepOffset }})
        max_distance = {{ ns.stepOffset + ns.edgeSteps }}.0;
        if (distance1 > 900.0) {
            if isHorizontal {
                let currentUv1 = currentUv - vec2f({{ ns.stepOffset + ns.edgeSteps//2 + 1 }}.0, 0.0) * pixelStep;
                $$ for si in range(ns.edgeSteps)
                lumaEnd_{{si}} = rgb2luma(textureSampleLevel(tex, smp, currentUv1, 0.0, -vec2i( {{si-ns.edgeSteps//2}}, 0)).rgb) - lumaLocalAverage;
                $$ endfor
            } else {
                let currentUv1 = currentUv - vec2f(0.0, {{ ns.stepOffset + ns.edgeSteps//2 +1 }}.0) * pixelStep;
                $$ for si in range(ns.edgeSteps)
                lumaEnd_{{si}} = rgb2luma(textureSampleLevel(tex, smp, currentUv1, 0.0, -vec2i(0, {{si-ns.edgeSteps//2}} )).rgb) - lumaLocalAverage;
                $$ endfor
            }
            lumaEnd1 = lumaEnd_{{ns.edgeSteps-1}};
            $$ for si in range(ns.edgeSteps) | reverse
            if (abs(lumaEnd_{{si}}) >= gradientScaled) { distance1 = {{si + 1 + ns.stepOffset}}.0; lumaEnd1 = lumaEnd_{{si}}; }
            $$ endfor
        }
        if (distance2 > 900.0) {
            if isHorizontal {
                let currentUv2 = currentUv + vec2f({{ ns.stepOffset + ns.edgeSteps//2 + 1}}.0, 0.0) * pixelStep;
                $$ for si in range(ns.edgeSteps)
                lumaEnd_{{si}} = rgb2luma(textureSampleLevel(tex, smp, currentUv2, 0.0, vec2i( {{si-ns.edgeSteps//2}}, 0)).rgb) - lumaLocalAverage;
                $$ endfor
            } else {
                let currentUv2 = currentUv + vec2f(0.0, {{ ns.stepOffset + ns.edgeSteps//2 + 1 }}.0) * pixelStep;
                $$ for si in range(ns.edgeSteps)
                lumaEnd_{{si}} = rgb2luma(textureSampleLevel(tex, smp, currentUv2, 0.0, vec2i(0, {{si-ns.edgeSteps//2}} )).rgb) - lumaLocalAverage;
                $$ endfor
            }
            lumaEnd2 = lumaEnd_{{ns.edgeSteps-1}};
            $$ for si in range(ns.edgeSteps) | reverse
            if (abs(lumaEnd_{{si}}) >= gradientScaled) { distance2 = {{si + 1 + ns.stepOffset}}.0; lumaEnd2 = lumaEnd_{{si}}; }
            $$ endfor
        }

        $$endfor

        // Clip the distance (if we did not find the end, we assume it's one pixel further)
        distance1 = min(distance1, max_distance + 1.0);
        distance2 = min(distance2, max_distance + 1.0);

        // UV offset: read in the direction of the closest side of the edge.
        let pixelOffset = - min(distance1, distance2) / (distance1 + distance2) + 0.5;

        // If the luma at center is smaller than at its neighbor, the delta luma at each end should be positive (same variation).
        let isLumaCenterSmaller = lumaCenter < lumaLocalAverage;
        var correctVariation: bool;
        if (distance1 < distance2) {
            correctVariation = (lumaEnd1 < 0.0) != isLumaCenterSmaller;
        } else {
            correctVariation = (lumaEnd2 < 0.0) != isLumaCenterSmaller;
        }

        // Set subpixel texCoord offset
        if (!correctVariation) {
            subpixelEdgeOffset = vec2f(0.0);
        } else if isHorizontal {
            subpixelEdgeOffset = vec2f(0.0, pixelOffset * stepLength);
        } else {
            subpixelEdgeOffset = vec2f(pixelOffset * stepLength, 0.0);
        }
    }

    // We mix the effects of the edge-search with the directional diffusion.
    // Basically, we allow more diffusion if the edge-effect is smaller.
    let edgeStrength = (min(1.0, length(2.0 * subpixelEdgeOffset / pixelStep)));
    diffuseStrength = diffuseStrength * (1.0 - edgeStrength);

    // The step to take for the diffusion effect (blur in the direction of the
    // edge). Note that for diagonal-ish lines, the most blur is obtained when
    // stepping halfway to the next pixel, i.e. 0.707, because then the neighbour
    // pixels are taken into account more. Let's use no more than 0.6 because then
    // for horizontal lines, the max diffusion kernel is effectively [0.6, 2 * 0.5,
    // 0.6] which is still somewhat bell-shaped. Actually, if we use 0.5, we trade a
    // bit more smoothness for perceived sharpness. Make its 0.51 so it does not
    // look like some sort of offset.
    let max_step_size = 0.51;
    let diffuseStep = diffuseDirection * pixelStep * (max_step_size * diffuseStrength);

    // Compose the three texture coordinates, combining the ede-offset with the diffusion componennt, depending on the steepness of the edge.
    let texCoord1 = texCoord - diffuseStep + subpixelEdgeOffset;
    let texCoord2 = texCoord + diffuseStep + subpixelEdgeOffset;

    // Sample the final color
    var finalColor = vec3f(0.0);
    finalColor += 0.5 * textureSampleLevel(tex, smp, texCoord1, 0.0).rgb;
    finalColor += 0.5 * textureSampleLevel(tex, smp, texCoord2, 0.0).rgb;

    return vec4f(finalColor, centerSample.a);

}
