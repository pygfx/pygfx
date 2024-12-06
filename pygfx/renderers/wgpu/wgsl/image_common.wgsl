// Common functionality for image

{$ include 'pygfx.image_sample.wgsl' $}

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
    $$ if img_format == 'f32'
        $$ if colorspace == 'yuv420p'
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
            return textureSample(t_img, s_img, texcoord.xy);
        $$ endif
    $$ else
        let texcoords_u = vec2<i32>(texcoord.xy * sizef.xy);
        return vec4<f32>(textureLoad(t_img, texcoords_u, 0));
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
