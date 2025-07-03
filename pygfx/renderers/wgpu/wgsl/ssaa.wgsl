// ssaa.wgsl  version 1.1
//
// Source: https://github.com/almarklein/ppaa-experiments/blob/main/wgsl/ssaa.wgsl
// Used in https://github.com/pygfx/pygfx/, this code is somewhat opinionated towards Pygfx.
//
// An interpolation / reconstruction filter that has two purposes:
// * downsampling: render at a higher resolution, then downsample to reduce aliasing. This is SSAA.
// * upsampling: render at a lower resolution, then upsample to screen resolution. For performance.
//
// In both cases you want appropriate filtering. This module supports different filtering methods.
// In general the 'mitchell' filter is recommended.
//
// Inspired by https://therealmjp.github.io/posts/msaa-resolve-filters/
// and         https://bartwronski.com/2022/03/07/fast-gpu-friendly-antialiasing-downsampling-filter/


fn filterweightBox(t: vec2f) -> f32 {
    // The box filter results in nearest-neighbour interpolation when upsampling,
    // and simple averaging when downsampling.
    return f32(-0.5 < t.x && t.x <= 0.5) * f32(-0.5 < t.y && t.y <= 0.5);
}

fn filterweightDisk(t: vec2f) -> f32 {
    // The disk filter results in round dots when upsampling, and simple averaging
    // when downsampling. The comparison with 0.4 is deliberate to produce separate
    // dots; its's a nice way to visualize individual pixels.
    return f32(length(t) < 0.4);
}

fn filterweightTent(t: vec2f) -> f32 {
    // The tent/pyramid filter results in linear interpolation when upsampling and
    // downsampling. The result looks quite adequate on first sight, but the cubic
    // filters produce a sharper image due to their better frequency response.
    return max(0.0, f32(1.0 - abs(t.x))) * max(0.0, f32(1.0 - abs(t.y)));
}

fn cubicWeights(t1: f32, B: f32, C: f32) -> f32 {
    // Generic parametrized Cubic kernel.
    let t = abs(t1);
    var w = 0.0;
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
    const b = 1.0;
    const c = 0.0;
    return cubicWeights(t.x, b, c) * cubicWeights(t.y, b, c);
}

fn filterweightMitchell(t: vec2f) -> f32 {
    // The Mitchell cubic spline is designed to offer a good balance between
    // frequency response, blurring, and artifacts, in the context of image
    // interpolation and reconstruction.
    // https://en.wikipedia.org/wiki/Mitchell%E2%80%93Netravali_filters
    const b = 1.0 / 3.0;
    const c = 1.0 / 3.0;
    return cubicWeights(t.x, b, c) * cubicWeights(t.y, b, c);
    // Note: writing out the formula for this specific B and C does not seem to help performance.
}

fn filterweightCatmull(t: vec2f) -> f32 {
    // The Catmull-Rom cubic spline is well know for its pleasing interpolating properties.
    const b = 0.0; // b == 0 means Cardinal spline
    const c = 0.5;
    return cubicWeights(t.x, b, c) * cubicWeights(t.y, b, c);
}

fn filterweightCubictent(t: vec2f) -> f32 {
    // Cardinal spline with tension zero is effectively a tent/pyramid filter
    const b = 0.0;
    const c = 0.0;
    return cubicWeights(t.x, b, c) * cubicWeights(t.y, b, c);
}


@fragment
fn fs_main(varyings: Varyings) -> @location(0) vec4<f32> {

    let resolution = vec2<f32>(textureDimensions(colorTex).xy);
    let invPixelSize = 1.0 / resolution;
    let texCoordOrig = varyings.texCoord;

    // Calculate positions, we distinguish between original position, nearest pixel, reference pixel.

    // Get the coord expressed in float pixels (for the source texture). The pixel centers are at 0.5, 1.5, 2.5, etc.
    let fPosOrig: vec2f = texCoordOrig * resolution;
    // Get the nearest integer pixel index into the source texture (floor, not round!), for an odd kernel.
    let iPosNear: vec2i = vec2i(fPosOrig);
    // Select the reference pixel index representing the left pixel of an even kernel.
    let iPosLeft: vec2i = vec2i(round(fPosOrig)) - 1;
    // Project the rounded pixel location back to float, representing the center of that pixel
    let fPosNear = vec2f(iPosNear) + 0.5;
    let fPosLeft = vec2f(iPosLeft) + 0.5;
    // Translate to texture coords
    let texCoordNear = fPosNear * invPixelSize;
    let texCoordLeft = fPosLeft * invPixelSize;

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

    $$ set originalFilter = filter

    {# Generally speaking, even with a pixel ratio of 1, the input and output grid may not be aligned. #}
    {# But in our case (we assume) they are, so this basically becomes a 1-pixel copy-pass. #}
    {# We still use 'linear' and not 'nearest' because if the above assumption is not met, it's likely easier to spot due to the blurring. #}
    $$ if scaleFactor == 1.0 and filter != 'nearest'
    $$     set filter = 'linear'
    $$ endif

    // To determine the size of the filter kernel, i.e. the support for the cubic
    // kernel, we need the scale factor between the source and target texture. The
    // scaleFactor is defined such that if its < 1, the source is smaller, i.e.
    // we're upsampling. The kernelSupport factor is a float that represents the
    // distance in source pixels from the current sample point, at which the kernel
    // is still nonzero. The simple filters have a kernelSupport that is equal to
    // the scale factor. For cubic kernels it's twice the scale factor. Since the
    // filter is zero AT the max distance we multiply with 0.999 and 1.999,
    // respectively.
    //
    // Next, we can decide whether to use an odd or even kernel. If the decimal part
    // of the kernelSupport is smaler than 0.5, it cannot reach the next pixel from
    // the edge of the current pixel. In that case we can use the nearest pixel as
    // the reference, and thus use an odd kernel. If the decimal part is larger than
    // 0.5, it cannot reach the next pixel from the *center* of the reference pixel,
    // in which case it's necessary to use an even kernel. By selecting odd/even
    // kernels this way, we obtain perfect interpolation at max performance.

    $$  if filter in ["nearest", "linear"]
    $$      set refPos = "Near" if filter == "nearest" else "Orig"
    $$      set kernelSupport = 0
    $$      set delta1 = 0
    $$      set delta2 = 0
    $$  else
    $$      if filter in ["box", "disk", "tent"]
    $$          set kernelSupport = [0.999, scaleFactor * 0.999] | max
    $$      elif filter in ["bspline", "mitchell", "catmull"]
    $$          set kernelSupport = [1.999, scaleFactor * 1.999] | max
    $$      else
    $$          set kernelSupport = fail_because_invalid_filter
    $$      endif
    {#      The extraKernelSupport is for testing #}
    $$      if extraKernelSupport
    $$          set kernelSupport = kernelSupport + extraKernelSupport
    $$      endif
    $$      set kernelSupportInt = kernelSupport | int
    $$      if kernelSupport % 1 <= 0.5
    {#          With this support, we can use an odd kernel, centered around the nearest pixel. #}
    $$          set refPos = "Near"
    $$          set delta1 = - kernelSupportInt
    $$          set delta2 =   kernelSupportInt + 1
    $$      else
    {#          Otherwitse use an even kernel, centered around the two nearest pixels. #}
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

    {# Show info in the generated wgsl #}
    // Templating info:
    // Original filter '{{ originalFilter }}'
    // Used filter '{{ filter }}'
    // scaleFactor: {{ scaleFactor }}
    // kernelSupport: {{ kernelSupport }}
    // delta1: {{ delta1 }}
    // delta2: {{ delta2 }}

    // The sigma (scale) of the filter scales with the scaleFactor, because it
    // defines the cut-off frequency of the filter. But when we up-sample, we don't
    // need a filter, and we go in pure interpolation mode, and the filter must
    // match the resolution (== sample rate) of the source image, i.e. one.
    const sigma = max(f32({{ scaleFactor }}), 1.0);

    // Prepare output
    var color = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    var weight = 0.0;

    $$ if delta1 == 0 and delta2 == 0
        // Sample color directly from the texture
        color = textureSampleLevel(colorTex, texSampler, texCoord{{ refPos }}, 0.0);

    $$ elif false and optScale2 and scaleFactor == 2 and filter == 'tent'
        // Optimization: with scaleFactor 2, we can pre-calculate kernel weights *and* use bilinear sampling trickery!
        // Created with https://gist.github.com/almarklein/a6113c202ec87987df1c954bd947e757
        // For tent we just need 4 lookups.
        color +=  0.249998 * textureSampleLevel(colorTex, texSampler, texCoordOrig + vec2f(-0.656246, -0.656246) * invPixelSize, 0.0);
        color +=  0.249998 * textureSampleLevel(colorTex, texSampler, texCoordOrig + vec2f(-0.656246,  0.656246) * invPixelSize, 0.0);
        color +=  0.249998 * textureSampleLevel(colorTex, texSampler, texCoordOrig + vec2f( 0.656246, -0.656246) * invPixelSize, 0.0);
        color +=  0.249998 * textureSampleLevel(colorTex, texSampler, texCoordOrig + vec2f( 0.656246,  0.656246) * invPixelSize, 0.0);

     $$ elif optScale2 and scaleFactor == 2 and filter == 'bspline'
        // Optimization: with scaleFactor 2, we can pre-calculate kernel weights *and* use bilinear sampling trickery!
        // Created with https://gist.github.com/almarklein/a6113c202ec87987df1c954bd947e757
        // For Bspline we use all 16 lookups.
        color +=  0.001329 * textureSampleLevel(colorTex, texSampler, texCoordOrig + vec2f(-3.000000, -3.000000) * invPixelSize, 0.0);
        color +=  0.016961 * textureSampleLevel(colorTex, texSampler, texCoordOrig + vec2f(-2.538413, -0.840665) * invPixelSize, 0.0);
        color +=  0.016961 * textureSampleLevel(colorTex, texSampler, texCoordOrig + vec2f(-2.538413,  0.840665) * invPixelSize, 0.0);
        color +=  0.001329 * textureSampleLevel(colorTex, texSampler, texCoordOrig + vec2f(-3.000000,  3.000000) * invPixelSize, 0.0);
        color +=  0.016961 * textureSampleLevel(colorTex, texSampler, texCoordOrig + vec2f(-0.840665, -2.538413) * invPixelSize, 0.0);
        color +=  0.214872 * textureSampleLevel(colorTex, texSampler, texCoordOrig + vec2f(-0.839843, -0.839843) * invPixelSize, 0.0);
        color +=  0.214872 * textureSampleLevel(colorTex, texSampler, texCoordOrig + vec2f(-0.839843,  0.839843) * invPixelSize, 0.0);
        color +=  0.016961 * textureSampleLevel(colorTex, texSampler, texCoordOrig + vec2f(-0.840665,  2.538413) * invPixelSize, 0.0);
        color +=  0.016961 * textureSampleLevel(colorTex, texSampler, texCoordOrig + vec2f( 0.840665, -2.538413) * invPixelSize, 0.0);
        color +=  0.214872 * textureSampleLevel(colorTex, texSampler, texCoordOrig + vec2f( 0.839843, -0.839843) * invPixelSize, 0.0);
        color +=  0.214872 * textureSampleLevel(colorTex, texSampler, texCoordOrig + vec2f( 0.839843,  0.839843) * invPixelSize, 0.0);
        color +=  0.016961 * textureSampleLevel(colorTex, texSampler, texCoordOrig + vec2f( 0.840665,  2.538413) * invPixelSize, 0.0);
        color +=  0.001329 * textureSampleLevel(colorTex, texSampler, texCoordOrig + vec2f( 3.000000, -3.000000) * invPixelSize, 0.0);
        color +=  0.016961 * textureSampleLevel(colorTex, texSampler, texCoordOrig + vec2f( 2.538413, -0.840665) * invPixelSize, 0.0);
        color +=  0.016961 * textureSampleLevel(colorTex, texSampler, texCoordOrig + vec2f( 2.538413,  0.840665) * invPixelSize, 0.0);
        color +=  0.001329 * textureSampleLevel(colorTex, texSampler, texCoordOrig + vec2f( 3.000000,  3.000000) * invPixelSize, 0.0);

    $$ elif optScale2 and scaleFactor == 2 and filter == 'mitchell'
        // Optimization: with scaleFactor, we can pre-calculate kernel weights *and* use bilinear sampling trickery!
        // Created with https://gist.github.com/almarklein/a6113c202ec87987df1c954bd947e757
        // For Mitchell uses 12 lookups, which is more performant than 16 lookups while still producing an error < 0.001.
        color += -0.009934 * textureSampleLevel(colorTex, texSampler, texCoordOrig + vec2f(-2.886093, -0.746066) * invPixelSize, 0.0);
        color += -0.009934 * textureSampleLevel(colorTex, texSampler, texCoordOrig + vec2f(-2.886093,  0.746066) * invPixelSize, 0.0);
        color += -0.009934 * textureSampleLevel(colorTex, texSampler, texCoordOrig + vec2f(-0.746066, -2.886093) * invPixelSize, 0.0);
        color +=  0.269867 * textureSampleLevel(colorTex, texSampler, texCoordOrig + vec2f(-0.746677, -0.746677) * invPixelSize, 0.0);
        color +=  0.269867 * textureSampleLevel(colorTex, texSampler, texCoordOrig + vec2f(-0.746677,  0.746677) * invPixelSize, 0.0);
        color += -0.009934 * textureSampleLevel(colorTex, texSampler, texCoordOrig + vec2f(-0.746066,  2.886093) * invPixelSize, 0.0);
        color += -0.009934 * textureSampleLevel(colorTex, texSampler, texCoordOrig + vec2f( 0.746066, -2.886093) * invPixelSize, 0.0);
        color +=  0.269867 * textureSampleLevel(colorTex, texSampler, texCoordOrig + vec2f( 0.746677, -0.746677) * invPixelSize, 0.0);
        color +=  0.269867 * textureSampleLevel(colorTex, texSampler, texCoordOrig + vec2f( 0.746677,  0.746677) * invPixelSize, 0.0);
        color += -0.009934 * textureSampleLevel(colorTex, texSampler, texCoordOrig + vec2f( 0.746066,  2.886093) * invPixelSize, 0.0);
        color += -0.009934 * textureSampleLevel(colorTex, texSampler, texCoordOrig + vec2f( 2.886093, -0.746066) * invPixelSize, 0.0);
        color += -0.009934 * textureSampleLevel(colorTex, texSampler, texCoordOrig + vec2f( 2.886093,  0.746066) * invPixelSize, 0.0);

     $$ elif optScale2 and scaleFactor == 2 and filter == 'catmull'
        // Optimization: with scaleFactor, we can pre-calculate kernel weights *and* use bilinear sampling trickery!
        // Created with https://gist.github.com/almarklein/a6113c202ec87987df1c954bd947e757
        // For Bspline we use all 16 lookups.
        color +=  0.002197 * textureSampleLevel(colorTex, texSampler, texCoordOrig + vec2f(-3.000000, -3.000000) * invPixelSize, 0.0);
        color += -0.025636 * textureSampleLevel(colorTex, texSampler, texCoordOrig + vec2f(-2.750033, -0.707216) * invPixelSize, 0.0);
        color += -0.025636 * textureSampleLevel(colorTex, texSampler, texCoordOrig + vec2f(-2.750033,  0.707216) * invPixelSize, 0.0);
        color +=  0.002197 * textureSampleLevel(colorTex, texSampler, texCoordOrig + vec2f(-3.000000,  3.000000) * invPixelSize, 0.0);
        color += -0.025636 * textureSampleLevel(colorTex, texSampler, texCoordOrig + vec2f(-0.707216, -2.750033) * invPixelSize, 0.0);
        color +=  0.299072 * textureSampleLevel(colorTex, texSampler, texCoordOrig + vec2f(-0.707143, -0.707143) * invPixelSize, 0.0);
        color +=  0.299072 * textureSampleLevel(colorTex, texSampler, texCoordOrig + vec2f(-0.707143,  0.707143) * invPixelSize, 0.0);
        color += -0.025636 * textureSampleLevel(colorTex, texSampler, texCoordOrig + vec2f(-0.707216,  2.750033) * invPixelSize, 0.0);
        color += -0.025636 * textureSampleLevel(colorTex, texSampler, texCoordOrig + vec2f( 0.707216, -2.750033) * invPixelSize, 0.0);
        color +=  0.299072 * textureSampleLevel(colorTex, texSampler, texCoordOrig + vec2f( 0.707143, -0.707143) * invPixelSize, 0.0);
        color +=  0.299072 * textureSampleLevel(colorTex, texSampler, texCoordOrig + vec2f( 0.707143,  0.707143) * invPixelSize, 0.0);
        color += -0.025636 * textureSampleLevel(colorTex, texSampler, texCoordOrig + vec2f( 0.707216,  2.750033) * invPixelSize, 0.0);
        color +=  0.002197 * textureSampleLevel(colorTex, texSampler, texCoordOrig + vec2f( 3.000000, -3.000000) * invPixelSize, 0.0);
        color += -0.025636 * textureSampleLevel(colorTex, texSampler, texCoordOrig + vec2f( 2.750033, -0.707216) * invPixelSize, 0.0);
        color += -0.025636 * textureSampleLevel(colorTex, texSampler, texCoordOrig + vec2f( 2.750033,  0.707216) * invPixelSize, 0.0);
        color +=  0.002197 * textureSampleLevel(colorTex, texSampler, texCoordOrig + vec2f( 3.000000,  3.000000) * invPixelSize, 0.0);

    $$ else

        // Templated loop (is more performant than a loop in wgsl)
        var c: vec4f;
        var t: vec2f;
        var w: f32;
        $$ for dy in range(delta1, delta2)
        $$ for dx in range(delta1, delta2)
        $$ set iscorner = dx in [delta1, delta2 - 1] and dy in [delta1, delta2 - 1]
        $$ if (not optCorners) or (not iscorner) or (delta2 - delta1 <= 6)
            t = fPos{{ refPos }} - fPosOrig + vec2f({{ dx }}, {{ dy }});
            w = filterweight{{ filter.lower().capitalize() }}(t / sigma);
            c = textureSampleLevel(colorTex, texSampler, texCoord{{ refPos }}, 0.0, vec2i({{ dx }}, {{ dy }}));
            color += w * c;
            weight += w;
        $$ endif
        $$ endfor
        $$ endfor
        if weight == 0.0 { weight = 1.0; }
        color /= weight;

    $$ endif


    // Apply gamma
    $$ if gamma is not defined
    $$ set gamma = 1.0
    $$ endif
    let gamma3 = vec3<f32>({{ gamma }});
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
