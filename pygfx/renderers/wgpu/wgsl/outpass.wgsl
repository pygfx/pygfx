
// Interpolating / reconstruction filter
// Inspired by https://therealmjp.github.io/posts/msaa-resolve-filters/

fn filterweightBox(t: vec2f) -> f32 {
    // The box filter results in nearest-neighbour interpolation when upsampling,
    // and simple averaging when downsampling.
    return f32(abs(t.x) <= 0.5) * f32(abs(t.y) <= 0.5);
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

fn filterweightGaussian(t: vec2f) -> f32 {
    // The Gaussian filter applies diffusion to all pixels touched by the kernel,
    // leading to a smooth (i.e. blury) result. We multiply the t with 2, to decrease
    // the effective width of the Gaussian kernel. If we would not do this, the blur
    // would be large, while the result would be pixelated because the kernel
    // support would be too small to incorporate the tail of the kernel (i.e. we'd
    // have to sample much more pixels).
    let t2 = t * 2.0;
    let t3 = length(t2);
    return exp(-0.5 * t3 * t3);
    // return exp(-0.5 * t2.x * t2.x) * exp(-0.5 * t2.y * t2.y);  -> the same except for float inaccuracies because Gaussian is seperable
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
    return cubicWeights(t.x, 1.0, 0.0) * cubicWeights(t.y, 1.0, 0.0);
}

fn filterweightCatmullrom(t: vec2f) -> f32 {
    // The Catmull-Rom cubic spline is well know for its pleasing interpolating properties.
    return cubicWeights(t.x, 0.0, 0.5) * cubicWeights(t.y, 0.0, 0.5);
}

fn filterweightMitchell(t: vec2f) -> f32 {
    // The Mitchell cubic spline is designed to offer a good balance between
    // frequency response, blurring, and artifacts, in the context of image
    // interpolation and reconstruction. This is our best filter.
    const b = 1 / 3.0;
    return cubicWeights(t.x, b, b) * cubicWeights(t.y, b, b);
    // Note: writing out the formula for this specific B and C does not seem to help performance.
}

fn filterweightCubic(t: vec2f) -> f32 {
    // Alias for Mitchell
    const b = 1 / 3.0;
    return cubicWeights(t.x, b, b) * cubicWeights(t.y, b, b);
}


@fragment
fn fs_main(varyings: Varyings) -> @location(0) vec4<f32> {

    let resolution = vec2<f32>(textureDimensions(colorTex).xy);
    let texCoordOrig = varyings.texCoord;

    // Calculate positions, we distinguish between original position, nearest pixel, reference pixel.

    // Get the coord expressed in float pixels (for the source texture). The pixel centers are at 0.5, 1.5, 2.5, etc.
    let fPosOrig: vec2f = texCoordOrig * resolution;
    // Get the nearest integer pixel index into the source texture (floor, not round!)
    let iPosNear: vec2i = vec2i(fPosOrig);
    // Select the reference pixel index appropriate for a cubic kernel.
    let iPosLeft: vec2i = vec2i(round(fPosOrig)) - 1;
    // Project the rounded pixel location back to float, representing the center of that pixel
    let fPosNear = vec2f(iPosNear) + 0.5;
    let fPosLeft = vec2f(iPosLeft) + 0.5;
    // Translate to texture coords
    let texCoordNear = fPosNear / resolution;
    let texCoordLeft = fPosLeft / resolution;

    //  0.   1.   2.   3.   4.   position
    //   ____ ____ ____ ____
    //  |    |    | x  |    |
    //  |____|____|____|____|
    //     0    1    2    3      pixel index
    //
    //  Imagine the sample at x:
    //
    //  fPosOrig = 2.4
    //  iPosNear = 2
    //  iPosLeft = 1
    //  fPosLeft = 1.5

    // Generally speaking, even with a pixel ratio of 1, the input and output grid may not be aligned.
    // But in our case (we assume) they are, so this basically becomes a 1-pixel copy-pass.
    // We still use 'linear' and not 'nearest' because if the above assumption is not met, it's likely
    // easier to spot due to the blurring.
    $$ if scaleFactor == 1.0 and filter != 'nearest'
    $$     set filter = 'linear'
    $$ endif

    // To determine the size of the patch to sample, i.e. the support for the cubic
    // kernel, we need the scale factor between the source and target texture. The
    // scaleFactor is defined such that if its < 1, the source is smaller, i.e.
    // we're upsampling. Ideally the kernelSupport would be int((scaleFactor *
    // 1.99)) so that a cubic spline can be fully sampled, but that would result in
    // a lot of samples to be made (100 samples for fsaax2 (scaleFactor 2). With the
    // below it'd be 36. It does mean that the tails of the filter are not used, but
    // since that more or less means more smoothing, this is allright, because we're
    // already downsampling; it's a good compromise. What's important is that for
    // scaleFactor of 1 and lower, the kernel support is [-1 0 1 2].

    $$  if filter in ["nearest", "linear"]
    $$      set refPos = "Near" if filter == "nearest" else "Orig"
    $$      set delta1 = 0
    $$      set delta2 = 0
    $$  else
    $$      if filter in ["box", "disk", "pyramid"]
    $$          set kernelSupport = [0.999, scaleFactor * 0.999] | max
    $$      else
    $$          set kernelSupport = [1.999, scaleFactor * 1.999] | max
    $$      endif
    $$      set kernelSupportInt = kernelSupport | int
    $$      if kernelSupport % 1 <= 0.5
    {#          With this support, we can use a kernel centered around the nearest pixel. #}
    $$          set refPos = "Near"
    $$          set delta1 = - kernelSupportInt
    $$          set delta2 =   kernelSupportInt + 1
    $$      else
    {#          This is the generic case, where the cubic kernel is centered around the two nearest pixels. #}
    $$          set refPos = "Left"
    $$          set delta1 = - kernelSupportInt
    $$          set delta2 =   kernelSupportInt + 2
    $$      endif
    $$  endif

    {# Optimalization for scale factor being a whole uneven number #}
    $$  if scaleFactor > 1 and scaleFactor % 1 == 0 and scaleFactor % 2 != 0
    $$     set delta2 = delta2 - 1
    $$     set refPos = "Near"
    $$  endif

    // The sigma (scale) of the filter scales with the scaleFactor, because it
    // defines the cut-off frequency of the filter. But when we up-sample, we don't
    // need a filter, and we go in pure interpolation mode, and the filter must
    // match the resolution (== sample rate) of the source image, i.e. one.
    const sigma = max({{ scaleFactor }}, 1.0);

    // Prepare output
    var color = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    var weight = 0.0;

    $$ if delta1 == 0 and delta2 <= 1

        // Sample color directly from the texture
        color = textureSampleLevel(colorTex, texSampler, texCoord{{ refPos }}, 0.0, vec2i(0, 0));

    $$ elif false and filter == 'cubic' and scaleFactor == 2.0
        // Optimization: https://bartwronski.com/2022/03/07/fast-gpu-friendly-antialiasing-downsampling-filter/
        let invPixelSize = 1.0 / resolution;
        let uv = texCoordOrig;
        color += 0.37487566 * textureSampleLevel(colorTex, texSampler, uv + vec2f(-0.75777,-0.75777)*invPixelSize, 0.0);
        color += 0.37487566 * textureSampleLevel(colorTex, texSampler, uv + vec2f(0.75777,-0.75777)*invPixelSize, 0.0);
        color += 0.37487566 * textureSampleLevel(colorTex, texSampler, uv + vec2f(0.75777,0.75777)*invPixelSize, 0.0);
        color += 0.37487566 * textureSampleLevel(colorTex, texSampler, uv + vec2f(-0.75777,0.75777)*invPixelSize, 0.0);

        color += -0.12487566 * textureSampleLevel(colorTex, texSampler, uv + vec2f(-2.907,0.0)*invPixelSize, 0.0);
        color += -0.12487566 * textureSampleLevel(colorTex, texSampler, uv + vec2f(2.907,0.0)*invPixelSize, 0.0);
        color += -0.12487566 * textureSampleLevel(colorTex, texSampler, uv + vec2f(0.0,-2.907)*invPixelSize, 0.0);
        color += -0.12487566 * textureSampleLevel(colorTex, texSampler, uv + vec2f(0.0,2.907)*invPixelSize, 0.0);
    $$ else

        // Templated loop (is more performant than a loop in wgsl)
        // for dy in range({{ delta1 }}, {{ delta2 }})
        // for dx in range({{ delta1 }}, {{ delta2 }})
        var c: vec4f;
        var t: vec2f;
        var w: f32;
        $$ for dy in range(delta1, delta2)
        $$ for dx in range(delta1, delta2)
            t = fPos{{ refPos }} - fPosOrig + vec2f({{ dx }}, {{ dy }});
            w = filterweight{{ filter.lower().capitalize() }}(t / sigma);
            c = textureSampleLevel(colorTex, texSampler, texCoord{{ refPos }}, 0.0, vec2i({{ dx }}, {{ dy }}));
            color += w * c;
            weight += w;
        $$ endfor
        $$ endfor
        if weight == 0.0 { weight = 1.0; }
        color /= weight;

    $$ endif


    // Apply gamma
    let gamma3 = vec3<f32>(u_effect.gamma);
    let rgb = pow(color.rgb, gamma3);
    let a = color.a;

    // The blend factors are simply ONE and ZERO, so the values as we return them here
    // are how they end up in the target texture.
    // We assume pre-multiply alpha for now.
    // We should at some point look into this, if we want to support transparent windows,
    // and change the code here based on the ``alpha_mode`` of the ``GPUCanvasContext``.
    // Note tha alpha is multiplied with itself, which is probbaly wrong.

    return vec4f(rgb * a, a * a);

    // Note that the final opacity is not necessarily one. This means that
    // the framebuffer can be blended with the background, or one can render
    // images that can be better combined with other content in a document.
    // It also means that most examples benefit from a gfx.Background.
}
