// Common functionality for image

{$ include 'pygfx.image_sample.wgsl' $}


fn bilinear_weights(t: vec2f) -> f32 {
    return max(0.0, f32(1.0 - abs(t.x))) * max(0.0, f32(1.0 - abs(t.y)));
}


fn bicubic_weights(t: vec2f) -> f32 {
    // The Mitchell cubic spline is designed to offer a good balance between frequency response,
    // blurring, and artifacts, in the context of image interpolation and reconstruction.
    const b = 1.0 / 3.0;
    const c = 1.0 / 3.0;
    return cubic_weights(t.x, b, c) * cubic_weights(t.y, b, c);
}


// See https://ffmpeg.org/doxygen/7.0/pixfmt_8h_source.html#l00609
// for some helpful definitions of color spaces and color ranges
fn yuv_limited_to_rgb(y: f32, u: f32, v: f32) -> vec4<f32> {
    // This formula is correct for the "limited range" YUV
    let c: f32 = y - 0.0625;        // Offset Y by 16/255
    let d: f32 = u - 0.5;           // Offset U by 128/255
    let e: f32 = v - 0.5;           // Offset V by 128/255

    let r: f32 = 1.1643 * c + 1.5958 * e;
    let g: f32 = 1.1643 * c - 0.3917 * d - 0.8129 * e;
    let b: f32 = 1.1643 * c + 2.0170 * d;

    return vec4<f32>(r, g, b, 1.);
}

fn yuv_full_to_rgb(y: f32, u: f32, v: f32) -> vec4<f32> {
    // this formula is correct for the "full range" YUV
    let d: f32 = u - 0.5;           // Offset U by 128/255
    let e: f32 = v - 0.5;           // Offset V by 128/255

    let r = y + 1.402 * e;
    let g = y - 0.344136 * d - 0.714136 * e;
    let b = y + 1.772 * d;

    return vec4<f32>(r, g, b, 1.);
}

fn sample_im(texcoord: vec2<f32>, sizef: vec2<f32>) -> vec4<f32> {
    $$ if colorspace.startswith('yuv')
        // YUV colorspace

        $$ if interpolation != 'via-sampler'
            invalid texture sampling  // yuv textures are only supported when sampling via a sampler
        $$ elif colorspace == 'yuv420p'
            $$ if three_grid_yuv
            let y = textureSample(t_img, s_img, texcoord.xy).x;
            let u = textureSample(t_u_img, s_img, texcoord.xy).x;
            let v = textureSample(t_v_img, s_img, texcoord.xy).x;
            $$ else
            // In this implementation we share a single 2D texture between U and V
            // We must therefore take care to not sample at the edge where
            // the texture will be poorly interpolated. See
            // https://github.com/pygfx/pygfx/pull/873#issuecomment-2516613301
            let txy = clamp(texcoord.xy / 2.0, 0.5 / sizef, 0.5 - 0.5 / sizef);
            let y = textureSample(t_img, s_img, texcoord.xy, 0).x;
            let u = textureSample(t_img, s_img, txy, 1).x;
            let v = textureSample(t_img, s_img, txy + vec2<f32>(0.5, 0.0), 1).x;
            $$ endif

            $$ if colorrange == "limited"
            return yuv_limited_to_rgb(y, u, v);
            $$ else
            return yuv_full_to_rgb(y, u, v);
            $$ endif
        $$ elif colorspace == "yuv444p"
            $$ if three_grid_yuv
            let y = textureSample(t_img, s_img, texcoord.xy).x;
            let u = textureSample(t_u_img, s_img, texcoord.xy).x;
            let v = textureSample(t_v_img, s_img, texcoord.xy).x;
            $$ else
            let y = textureSample(t_img, s_img, texcoord.xy, 0).x;
            let u = textureSample(t_img, s_img, texcoord.xy, 1).x;
            let v = textureSample(t_img, s_img, texcoord.xy, 2).x;
            $$ endif
            $$ if colorrange == "limited"
            return yuv_limited_to_rgb(y, u, v);
            $$ else
            return yuv_full_to_rgb(y, u, v);
            $$ endif
        $$ else
            unexpected colorspace '{{ colorspace }}'
        $$ endif

    $$ elif interpolation == 'via-sampler'
        // Using a sampler with either linear or nearest interpolation.
        // This path means that interpolation can be changed by only swapping the sampler.
        return textureSample(t_img, s_img, texcoord.xy);

    $$ else
        // Hard-coded interpolation using textureLoad

        let posf = texcoord.xy * sizef.xy - 0.5;    // offset 0.5 to align with center of pixels
        let posi = vec2i(posf);                     // the pixel directly 'left' of the coord
        let min_coord = vec2i(0);
        let max_coord = vec2i(sizef) - 1;
        var value = vec4f(0.0);
        var weight = 0.0;
        var w: f32;
        var p: vec2i;

        $$ if interpolation == 'cubic'
            $$ for dy in [-1, 0, 1, 2]
            $$ for dx in [-1, 0, 1, 2]
                p = posi + vec2i({{dx}}, {{dy}});
                w = bicubic_weights(posf - vec2f(p));
                weight += w;
                value += w * vec4f(textureLoad(t_img, clamp(p, min_coord, max_coord), 0));
            $$ endfor
            $$ endfor
        $$ elif interpolation == 'linear'
            $$ for dy in [0, 1]
            $$ for dx in [0, 1]
                p = posi + vec2i({{dx}}, {{dy}});
                w = bilinear_weights(posf - vec2f(p));
                weight += w;
                value += w * vec4f(textureLoad(t_img, clamp(p, min_coord, max_coord), 0));
            $$ endfor
            $$ endfor
        $$ else
            p = vec2<i32>(texcoord.xy * sizef.xy);
            weight = 1.0;
            value = vec4<f32>(textureLoad(t_img, p, 0));
        $$ endif

        return value / weight;

    $$ endif
}


struct ImGeometry {
    indices: array<i32,6>,
    positions: array<vec3<f32>,4>,
    texcoords: array<vec2<f32>,4>,
};

fn get_im_geometry() -> ImGeometry {
    let size = textureDimensions(t_img);
    var geo: ImGeometry;

    geo.indices = array<i32,6>(0, 1, 2,   3, 2, 1);

    let pos1 = vec2<f32>(-0.5);
    let pos2 = vec2<f32>(size.xy) + pos1;
    geo.positions = array<vec3<f32>,4>(
        vec3<f32>(pos2.x, pos1.y, 0.0),
        vec3<f32>(pos2.x, pos2.y, 0.0),
        vec3<f32>(pos1.x, pos1.y, 0.0),
        vec3<f32>(pos1.x, pos2.y, 0.0),
    );

    geo.texcoords = array<vec2<f32>,4>(
        vec2<f32>(1.0, 0.0),
        vec2<f32>(1.0, 1.0),
        vec2<f32>(0.0, 0.0),
        vec2<f32>(0.0, 1.0),
    );

    return geo;
}
