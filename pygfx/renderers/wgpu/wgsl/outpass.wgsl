
// Interpolating filter

fn filterweightBox(t: vec2f) -> f32 {
    // The box filter results in nearest-neighbour interpolation when upsampling,
    // and simple averaging when downsampling.
    return f32(abs(t.x) < 0.5) * f32(abs(t.y) < 0.5);
}

fn filterweightDisk(t: vec2f) -> f32 {
    // The disk filter results in round dots when upsampling, and simple averaging
    // when downsampling. The comparison with 0.4 is deliberate to produce separate
    // dots; its's a nice way to visualize individual pixels.
    return f32(length(t) < 0.4);
}

fn filterweightPyramid(t: vec2f) -> f32 {
    // The pyramid filter results in linear interpolation when upsampling and
    // downsampling. The result looks quite adequate on first sight, but the cubic
    // filters produce a sharper image due to their better frequency response.
    return max(0.0, f32(1.0 - abs(t.x))) * max(0.0, f32(1.0 - abs(t.y)));
}

fn filterweightCone(t: vec2f) -> f32 {
    // Not sure what to make of this: the disk-equivalent of the triangle filter.
    return max(0.0, f32(1.0 - length(t)));;
}

fn filterweightGaussian(t: vec2f) -> f32 {
    // The Gaussian filter applies diffusion to all pixels touched by the kernel,
    // leading to a smooth (i.e. blury) result. We multiply the t with 2, to decrease
    // the effective width of the Gaussian kernel. If we would not do this, the blur
    // would be large, while the result would be pixelated because the kernel
    // support would be too small to incorporate the tail of the kernel (i.e. we'd
    // have to sample much more pixels).
    let t2 = length(t) * 2.0;
    return exp(-0.5 * t2 * t2);
}

fn cubicWeights(t1: f32, B: f32, C: f32) -> f32 {
    // Generic parametrized Cubic kernel.
    let t = abs(t1);
    var w = 0.0f;
    let t2 = t * t;
    let t3 = t * t * t;
    if t < 1.0 {
        w = (12.0 - 9.0 * B - 6.0 * C) * t3 + (-18.0 + 12.0 * B + 6.0 * C) * t2 + (6.0 - 2.0 * B);
    } else if t <= 2.0 {
        w = (-B - 6.0 * C) * t3 + (6.0 * B + 30.0 * C) * t2 + (-12.0 * B - 48.0 * C) * t + (8.0 * B + 24.0 * C);
    }
    return w / 6.0;
}

fn filterweightBspline(t: vec2f) -> f32 {
    // The B-Spline is a non-interpolating cubic filter. It's sometimes useful
    // because it's C2 continuous, but quite useless in the current context t.b.h.
    return cubicWeights(length(t), 1.0, 0.0);
}

fn filterweightCatmullrom(t: vec2f) -> f32 {
    // The Catmull-Rom cubic spline is well know for its pleasing interpolating properties.
    return cubicWeights(length(t), 0.0, 0.5);
}

fn filterweightMitchell(t: vec2f) -> f32 {
    // The Mitchell cubic spline is designed to offer a good balance between
    // frequency response, blurring, and artifacts, in the context of image
    // interpolation and reconstruction. This is our best filter.
    return cubicWeights(length(t), 1 / 3.0, 1 / 3.0);
    // Note: unlike for Gaussian kernels, the below does *not* produce the same result. The diagonals won't have the negative lobes.
    // return cubicWeights(t.x, 1 / 3.0, 1 / 3.0) *  cubicWeights(t.y, 1 / 3.0, 1 / 3.0);
    // Note: writing out the formula for this specific B and C does not seem to help performance.
}


@fragment
fn fs_main(varyings: Varyings) -> @location(0) vec4<f32> {

    let texCoord = varyings.texCoord;
    let resolution = vec2<f32>(textureDimensions(colorTex).xy);

    // Get the coord expressed in float pixels (for the source texture). The pixel centers are at 0.5, 1.5, 2.5, etc.
    let fpos1: vec2f = texCoord * resolution;
    // Get the integer pixel index into the source texture (floor, not round!)
    let ipos: vec2i = vec2i(fpos1);
    // Project the rounded pixel location back to float, representing the center of that pixel
    let fpos2 = vec2f(ipos) + 0.5;
    // Get the offset for the current sample
    let tpos = fpos1 - fpos2;
    // The texcoord, snapped to the whole pixel in the source texture
    let texCoordSnapped = fpos2 / resolution;

    //  0.   1.   2.   3.   4.   position
    //   ____ ____ ____ ____
    //  |    |    | x  |    |
    //  |____|____|____|____|
    //     0    1    2    3      pixel index
    //
    //  Imagine the sample at x:
    //
    //  fpos1 = 2.4
    //  ipos  = 2
    //  fpos2 = 2.5
    //  tpos  = -0.1

    // To determine the size of the patch to sample, i.e. the support for the
    // kernel, we need the scale factor between the source and target texture. The
    // scaleFactor is defined such that if its < 1, the source is smaller, i.e.
    // we're upsampling. Ideally the kernelSupportFactor would be int((scaleFactor *
    // 1.99)) so that a cubic spline can be fully sampled, but that would result in
    // a lot of samples to be made (100 samples for fsaax2 (scaleFactor 2). With the
    // below it'd be 36. It does mean that the tails of the filter are not used, but
    // since that more or less means more smoothing, this is allright, because we're
    // already downsampling; it's a good compromise. What's important is that for
    // scaleFactor of 1 and lower, the kernel support is [-1 0 1 2].
    $$ set kernelSupportFactor = (scaleFactor * 0.5) | int
    $$ set delta1 = -1 - kernelSupportFactor
    $$ set delta2 = 3 + kernelSupportFactor

    // Generally speaking, even with a pixel ratio of 1, the input and output grid may not be aligned.
    // But in our case (we assume) they are, so this basically becomes 1-pixel copy-pass.
    $$ set delta1 = 0 if scaleFactor == 1.0 else delta1
    $$ set delta2 = 1 if scaleFactor == 1.0 else delta2

    // The sigma (scale) of the filter scales with the scaleFactor, because it
    // defines the cut-off frequency of the filter. But when we up-sample, we don't
    // need a filter, and we go in pure interpolation mode, and the filter must
    // match the resolution (== sample rate) of the source image, i.e. one.
    const sigma = max({{ scaleFactor }}, 1.0);

    // Prepare output
    var color = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    var weight = 0.0;

    // Templated loop (is more performant than a loop in wgsl)
    var c: vec4f;
    var t: vec2f;
    var w: f32;
    $$ for dy in range(delta1, delta2)
    $$ for dx in range(delta1, delta2)
        t = vec2f({{ dx }}, {{ dy }}) - tpos;
        w = filterweight{{ filter.lower().capitalize() }}(t / sigma);
        c = textureSampleLevel(colorTex, texSampler, texCoordSnapped, 0.0, vec2i({{ dx }}, {{ dy }}));
        color += w * c;
        weight += w;
        $$ if dx == 0 and dy == 0
        let nearest_color = c;
        $$ endif
    $$ endfor
    $$ endfor

    if weight == 0.0 { weight = 1.0; }
    // if weight == 0.0 { color = nearest_color;  weight = 1.0; }
    color *= (1.0 / weight);

    // The blend factors are simply ONE and ZERO, so the values as we return them here
    // are how they end up in the target texture.
    // We assume pre-multiply alpha for now.
    // We should at some point look into this, if we want to support transparent windows,
    // and change the code here based on the ``alpha_mode`` of the ``GPUCanvasContext``.
    // Note tha alpha is multiplied with itself, which is probbaly wrong.
    let a = color.a;
    return vec4f(color.rgb * a, a * a);

    // Note that the final opacity is not necessarily one. This means that
    // the framebuffer can be blended with the background, or one can render
    // images that can be better combined with other content in a document.
    // It also means that most examples benefit from a gfx.Background.
}
