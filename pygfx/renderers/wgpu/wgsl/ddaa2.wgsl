// ddaa2.wgsl version 2.1
//
// Directional Diffusion Anti Aliasing (DDAA) version 2: smooth along the edges based on Scharr kernel, and perform edge-search to better support horizontal/vertical edges.
//
// v0: original (2013): https://github.com/vispy/experimental/blob/master/fsaa/ddaa.glsl
// v1: ported to wgsl and tweaked (2025): https://github.com/almarklein/ppaa-experiments/blob/main/wgsl/ddaa1.wgsl
// v2: added edge search (2025): https://github.com/almarklein/ppaa-experiments/blob/main/wgsl/ddaa2.wgsl

// ========== CONFIG ==========

// The number of iterations to walk along the edge.
// Setting this to zero disables the edge search, effectively ddaa1.
// The first iter takes 8 samples, 4 in each direction.
// Each next iters takes 8 samples in the direction for which the end has not yet been found.
// A value of 2 is probably good enough, and is quite performant.
// A value of 3 is nice, but beyond that it does not do much.
$$ if MAX_EDGE_ITERS is not defined
$$ set MAX_EDGE_ITERS = 2
$$ endif
const MAX_EDGE_ITERS : i32 = {{ MAX_EDGE_ITERS }};

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
        if abs(gradient1) >= abs(gradient2) {
            stepLength = -stepLength;  // switch the direction
            lumaLocalAverage = 0.5 * (luma1 + lumaCenter);
        } else {
            lumaLocalAverage = 0.5 * (luma2 + lumaCenter);
        }

        // Shift UV in the correct direction by half a pixel (orthogonal to the edge)
        var currentUv = texCoord;
        if isHorizontal {
            currentUv.y += stepLength * 0.5;
        } else {
            currentUv.x += stepLength * 0.5;
        }

        // We'll sample values from the texture in groups of 4

        var distance1 = 99.0;
        var distance2 = 99.0;

        var lumaEnd1: f32;
        var lumaEnd2: f32;

        var lumaEnd_1: f32;
        var lumaEnd_2: f32;
        var lumaEnd_3: f32;
        var lumaEnd_4: f32;
        var lumaEnd_5: f32;
        var lumaEnd_6: f32;
        var lumaEnd_7: f32;
        var lumaEnd_8: f32;

        // Read the lumas at both current extremities of the exploration segment, and compute the delta wrt to the local average luma.
        if isHorizontal {
            lumaEnd_4 = rgb2luma(textureSampleLevel(tex, smp, currentUv, 0.0, vec2i(-4, 0)).rgb) - lumaLocalAverage;
            lumaEnd_3 = rgb2luma(textureSampleLevel(tex, smp, currentUv, 0.0, vec2i(-3, 0)).rgb) - lumaLocalAverage;
            lumaEnd_2 = rgb2luma(textureSampleLevel(tex, smp, currentUv, 0.0, vec2i(-2, 0)).rgb) - lumaLocalAverage;
            lumaEnd_1 = rgb2luma(textureSampleLevel(tex, smp, currentUv, 0.0, vec2i(-1, 0)).rgb) - lumaLocalAverage;
            lumaEnd_5 = rgb2luma(textureSampleLevel(tex, smp, currentUv, 0.0, vec2i( 1, 0)).rgb) - lumaLocalAverage;
            lumaEnd_6 = rgb2luma(textureSampleLevel(tex, smp, currentUv, 0.0, vec2i( 2, 0)).rgb) - lumaLocalAverage;
            lumaEnd_7 = rgb2luma(textureSampleLevel(tex, smp, currentUv, 0.0, vec2i( 3, 0)).rgb) - lumaLocalAverage;
            lumaEnd_8 = rgb2luma(textureSampleLevel(tex, smp, currentUv, 0.0, vec2i( 4, 0)).rgb) - lumaLocalAverage;
        } else {
            lumaEnd_4 = rgb2luma(textureSampleLevel(tex, smp, currentUv, 0.0, vec2i(0, -4)).rgb) - lumaLocalAverage;
            lumaEnd_3 = rgb2luma(textureSampleLevel(tex, smp, currentUv, 0.0, vec2i(0, -3)).rgb) - lumaLocalAverage;
            lumaEnd_2 = rgb2luma(textureSampleLevel(tex, smp, currentUv, 0.0, vec2i(0, -2)).rgb) - lumaLocalAverage;
            lumaEnd_1 = rgb2luma(textureSampleLevel(tex, smp, currentUv, 0.0, vec2i(0, -1)).rgb) - lumaLocalAverage;
            lumaEnd_5 = rgb2luma(textureSampleLevel(tex, smp, currentUv, 0.0, vec2i(0,  1)).rgb) - lumaLocalAverage;
            lumaEnd_6 = rgb2luma(textureSampleLevel(tex, smp, currentUv, 0.0, vec2i(0,  2)).rgb) - lumaLocalAverage;
            lumaEnd_7 = rgb2luma(textureSampleLevel(tex, smp, currentUv, 0.0, vec2i(0,  3)).rgb) - lumaLocalAverage;
            lumaEnd_8 = rgb2luma(textureSampleLevel(tex, smp, currentUv, 0.0, vec2i(0,  4)).rgb) - lumaLocalAverage;
        }

        // The lumaEnd is at least that of the furthest pixel
        lumaEnd1 = lumaEnd_4;
        lumaEnd2 = lumaEnd_8;

        // Search for left endpoint in the current 4 samples
        if (abs(lumaEnd_4) >= gradientScaled) { distance1 = 4.0; lumaEnd1 = lumaEnd_4; }
        if (abs(lumaEnd_3) >= gradientScaled) { distance1 = 3.0; lumaEnd1 = lumaEnd_3; }
        if (abs(lumaEnd_2) >= gradientScaled) { distance1 = 2.0; lumaEnd1 = lumaEnd_2; }
        if (abs(lumaEnd_1) >= gradientScaled) { distance1 = 1.0; lumaEnd1 = lumaEnd_1; }
        // Same for the right endpoint
        if (abs(lumaEnd_8) >= gradientScaled) { distance2 = 4.0; lumaEnd2 = lumaEnd_8; }
        if (abs(lumaEnd_7) >= gradientScaled) { distance2 = 3.0; lumaEnd2 = lumaEnd_7; }
        if (abs(lumaEnd_6) >= gradientScaled) { distance2 = 2.0; lumaEnd2 = lumaEnd_6; }
        if (abs(lumaEnd_5) >= gradientScaled) { distance2 = 1.0; lumaEnd2 = lumaEnd_5; }

        // Now search for endpoints in a series of rounds, using a templated (i.e. unrolled) loop.
        // This is much faster (in WGSL) than a normal loop, probably due to optimization related to the texture lookups.
        // I found it also helps performance to use the same uv coordinate and use the offset parameter.

        $$ set ns = namespace(stepOffset=0)
        var max_distance = 4.0;
        $$ for iter in range(1, MAX_EDGE_ITERS)
            $$ set ns.stepOffset = 4 + (iter - 1) * 8
            // Iteration {{ iter }}
            max_distance = {{ 8.0 + ns.stepOffset }};
            if (distance1 > 90.0) {
                if isHorizontal {
                    let currentUv1 = currentUv + vec2f(-{{ ns.stepOffset + 1}}, 0.0) * pixelStep;
                    lumaEnd_1 = rgb2luma(textureSampleLevel(tex, smp, currentUv1, 0.0, vec2i( 0, 0)).rgb) - lumaLocalAverage;
                    lumaEnd_2 = rgb2luma(textureSampleLevel(tex, smp, currentUv1, 0.0, vec2i(-1, 0)).rgb) - lumaLocalAverage;
                    lumaEnd_3 = rgb2luma(textureSampleLevel(tex, smp, currentUv1, 0.0, vec2i(-2, 0)).rgb) - lumaLocalAverage;
                    lumaEnd_4 = rgb2luma(textureSampleLevel(tex, smp, currentUv1, 0.0, vec2i(-3, 0)).rgb) - lumaLocalAverage;
                    lumaEnd_5 = rgb2luma(textureSampleLevel(tex, smp, currentUv1, 0.0, vec2i(-4, 0)).rgb) - lumaLocalAverage;
                    lumaEnd_6 = rgb2luma(textureSampleLevel(tex, smp, currentUv1, 0.0, vec2i(-5, 0)).rgb) - lumaLocalAverage;
                    lumaEnd_7 = rgb2luma(textureSampleLevel(tex, smp, currentUv1, 0.0, vec2i(-6, 0)).rgb) - lumaLocalAverage;
                    lumaEnd_8 = rgb2luma(textureSampleLevel(tex, smp, currentUv1, 0.0, vec2i(-7, 0)).rgb) - lumaLocalAverage;
                } else {
                    let currentUv1 = currentUv + vec2f(0.0, -{{ ns.stepOffset + 1 }}) * pixelStep;
                    lumaEnd_1 = rgb2luma(textureSampleLevel(tex, smp, currentUv1, 0.0, vec2i(0, 0)).rgb) - lumaLocalAverage;
                    lumaEnd_2 = rgb2luma(textureSampleLevel(tex, smp, currentUv1, 0.0, vec2i(0, -1)).rgb) - lumaLocalAverage;
                    lumaEnd_3 = rgb2luma(textureSampleLevel(tex, smp, currentUv1, 0.0, vec2i(0, -2)).rgb) - lumaLocalAverage;
                    lumaEnd_4 = rgb2luma(textureSampleLevel(tex, smp, currentUv1, 0.0, vec2i(0, -3)).rgb) - lumaLocalAverage;
                    lumaEnd_5 = rgb2luma(textureSampleLevel(tex, smp, currentUv1, 0.0, vec2i(0, -4)).rgb) - lumaLocalAverage;
                    lumaEnd_6 = rgb2luma(textureSampleLevel(tex, smp, currentUv1, 0.0, vec2i(0, -5)).rgb) - lumaLocalAverage;
                    lumaEnd_7 = rgb2luma(textureSampleLevel(tex, smp, currentUv1, 0.0, vec2i(0, -6)).rgb) - lumaLocalAverage;
                    lumaEnd_8 = rgb2luma(textureSampleLevel(tex, smp, currentUv1, 0.0, vec2i(0, -7)).rgb) - lumaLocalAverage;
                }
                lumaEnd1 = lumaEnd_8;
                if (abs(lumaEnd_8) >= gradientScaled) { distance1 = {{ 8.0 + ns.stepOffset }}; lumaEnd1 = lumaEnd_8; }
                if (abs(lumaEnd_7) >= gradientScaled) { distance1 = {{ 7.0 + ns.stepOffset }}; lumaEnd1 = lumaEnd_7; }
                if (abs(lumaEnd_6) >= gradientScaled) { distance1 = {{ 6.0 + ns.stepOffset }}; lumaEnd1 = lumaEnd_6; }
                if (abs(lumaEnd_5) >= gradientScaled) { distance1 = {{ 5.0 + ns.stepOffset }}; lumaEnd1 = lumaEnd_5; }
                if (abs(lumaEnd_4) >= gradientScaled) { distance1 = {{ 4.0 + ns.stepOffset }}; lumaEnd1 = lumaEnd_4; }
                if (abs(lumaEnd_3) >= gradientScaled) { distance1 = {{ 3.0 + ns.stepOffset }}; lumaEnd1 = lumaEnd_3; }
                if (abs(lumaEnd_2) >= gradientScaled) { distance1 = {{ 2.0 + ns.stepOffset }}; lumaEnd1 = lumaEnd_2; }
                if (abs(lumaEnd_1) >= gradientScaled) { distance1 = {{ 1.0 + ns.stepOffset }}; lumaEnd1 = lumaEnd_1; }
            }
            if (distance2 > 90.0) {
                if isHorizontal {
                    let currentUv2 = currentUv + vec2f({{ ns.stepOffset + 1}}, 0.0) * pixelStep;
                    lumaEnd_1 = rgb2luma(textureSampleLevel(tex, smp, currentUv2, 0.0, vec2i(1, 0)).rgb) - lumaLocalAverage;
                    lumaEnd_2 = rgb2luma(textureSampleLevel(tex, smp, currentUv2, 0.0, vec2i(2, 0)).rgb) - lumaLocalAverage;
                    lumaEnd_3 = rgb2luma(textureSampleLevel(tex, smp, currentUv2, 0.0, vec2i(3, 0)).rgb) - lumaLocalAverage;
                    lumaEnd_4 = rgb2luma(textureSampleLevel(tex, smp, currentUv2, 0.0, vec2i(4, 0)).rgb) - lumaLocalAverage;
                    lumaEnd_5 = rgb2luma(textureSampleLevel(tex, smp, currentUv2, 0.0, vec2i(5, 0)).rgb) - lumaLocalAverage;
                    lumaEnd_6 = rgb2luma(textureSampleLevel(tex, smp, currentUv2, 0.0, vec2i(6, 0)).rgb) - lumaLocalAverage;
                    lumaEnd_7 = rgb2luma(textureSampleLevel(tex, smp, currentUv2, 0.0, vec2i(7, 0)).rgb) - lumaLocalAverage;
                    lumaEnd_8 = rgb2luma(textureSampleLevel(tex, smp, currentUv2, 0.0, vec2i(8, 0)).rgb) - lumaLocalAverage;
                } else {
                    let currentUv2 = currentUv + vec2f(0.0, {{ ns.stepOffset + 1 }}) * pixelStep;
                    lumaEnd_1 = rgb2luma(textureSampleLevel(tex, smp, currentUv2, 0.0, vec2i(0, 1)).rgb) - lumaLocalAverage;
                    lumaEnd_2 = rgb2luma(textureSampleLevel(tex, smp, currentUv2, 0.0, vec2i(0, 2)).rgb) - lumaLocalAverage;
                    lumaEnd_3 = rgb2luma(textureSampleLevel(tex, smp, currentUv2, 0.0, vec2i(0, 3)).rgb) - lumaLocalAverage;
                    lumaEnd_4 = rgb2luma(textureSampleLevel(tex, smp, currentUv2, 0.0, vec2i(0, 4)).rgb) - lumaLocalAverage;
                    lumaEnd_5 = rgb2luma(textureSampleLevel(tex, smp, currentUv2, 0.0, vec2i(0, 5)).rgb) - lumaLocalAverage;
                    lumaEnd_6 = rgb2luma(textureSampleLevel(tex, smp, currentUv2, 0.0, vec2i(0, 6)).rgb) - lumaLocalAverage;
                    lumaEnd_7 = rgb2luma(textureSampleLevel(tex, smp, currentUv2, 0.0, vec2i(0, 7)).rgb) - lumaLocalAverage;
                    lumaEnd_8 = rgb2luma(textureSampleLevel(tex, smp, currentUv2, 0.0, vec2i(0, 8)).rgb) - lumaLocalAverage;
                }
                lumaEnd2 = lumaEnd_8;
                if (abs(lumaEnd_8) >= gradientScaled) { distance2 = {{ 8.0 + ns.stepOffset }}; lumaEnd2 = lumaEnd_8; }
                if (abs(lumaEnd_7) >= gradientScaled) { distance2 = {{ 7.0 + ns.stepOffset }}; lumaEnd2 = lumaEnd_7; }
                if (abs(lumaEnd_6) >= gradientScaled) { distance2 = {{ 6.0 + ns.stepOffset }}; lumaEnd2 = lumaEnd_6; }
                if (abs(lumaEnd_5) >= gradientScaled) { distance2 = {{ 5.0 + ns.stepOffset }}; lumaEnd2 = lumaEnd_5; }
                if (abs(lumaEnd_4) >= gradientScaled) { distance2 = {{ 4.0 + ns.stepOffset }}; lumaEnd2 = lumaEnd_4; }
                if (abs(lumaEnd_3) >= gradientScaled) { distance2 = {{ 3.0 + ns.stepOffset }}; lumaEnd2 = lumaEnd_3; }
                if (abs(lumaEnd_2) >= gradientScaled) { distance2 = {{ 2.0 + ns.stepOffset }}; lumaEnd2 = lumaEnd_2; }
                if (abs(lumaEnd_1) >= gradientScaled) { distance2 = {{ 1.0 + ns.stepOffset }}; lumaEnd2 = lumaEnd_1; }
            }
        $$endfor

        // Clip the distance
        distance1 = min(distance1, max_distance);
        distance2 = min(distance2, max_distance);

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

        // For debugging
        // subpixelEdgeOffset = vec2f(distance1, distance2);
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

    // For debugging
    // if (subpixelEdgeOffset.x == 0.0 && subpixelEdgeOffset.y == 0.0) {
    //     finalColor = vec3f(0.0, 0.0, 0.0);
    // } else if (subpixelEdgeOffset.x <= 12.0 && subpixelEdgeOffset.y <= 12.0) {
    //     finalColor = vec3f(0.0, 0.0, 1.0);
    // } else {
    //     finalColor = vec3f(1.0, 0.0, 0.0);
    // }

    return vec4f(finalColor, centerSample.a);

}
