// Common functionality for volumes

{$ include 'pygfx.image_sample.wgsl' $}


fn trilinear_weights(t: vec3f) -> f32 {
    return max(0.0, f32(1.0 - abs(t.x))) * max(0.0, f32(1.0 - abs(t.y))) * max(0.0, f32(1.0 - abs(t.z)));
}


fn tricubic_weights(t: vec3f) -> f32 {
    // Pretty heavy. Ok for slices, but not recommended for raycasting.
    const b = 1.0 / 3.0;
    const c = 1.0 / 3.0;
    return cubic_weights(t.x, b, c) * cubic_weights(t.y, b, c) * cubic_weights(t.z, b, c);
}


fn sample_vol(texcoord: vec3<f32>, sizef: vec3<f32>) -> vec4<f32> {

    $$ if interpolation == 'via-sampler'
        // Using a sampler with either linear or nearest interpolation.
        // This path means that interpolation can be changed by only swapping the sampler.
        return textureSample(t_img, s_vol, texcoord.xyz);

    $$ else
        // Hard-coded interpolation using textureLoad

        let posf = texcoord.xyz * sizef.xyz - 0.5;    // offset 0.5 to align with center of pixels
        let posi = vec3i(posf);                      // the pixel directly 'left' of the coord
        let min_coord = vec3i(0);
        let max_coord = vec3i(sizef) - 1;
        var value = vec4f(0.0);
        var weight = 0.0;
        var w: f32;
        var p: vec3i;

        $$ if interpolation == 'cubic'
            $$ for dz in [-1, 0, 1, 2]
            $$ for dy in [-1, 0, 1, 2]
            $$ for dx in [-1, 0, 1, 2]
                p = posi + vec3i({{dx}}, {{dy}}, {{dz}});
                w = tricubic_weights(posf - vec3f(p));
                weight += w;
                value += w * vec4f(textureLoad(t_img, clamp(p, min_coord, max_coord), 0));
            $$ endfor
            $$ endfor
            $$ endfor
        $$ elif interpolation == 'linear'
            $$ for dz in [0, 1]
            $$ for dy in [0, 1]
            $$ for dx in [0, 1]
                p = posi + vec3i({{dx}}, {{dy}}, {{dz}});
                w = trilinear_weights(posf - vec3f(p));
                weight += w;
                value += w * vec4f(textureLoad(t_img, clamp(p, min_coord, max_coord), 0));
            $$ endfor
            $$ endfor
            $$ endfor
        $$ else
            p = vec3<i32>(texcoord.xyz * sizef.xyz);
            weight = 1.0;
            value = vec4<f32>(textureLoad(t_img, p, 0));
        $$ endif

        return value / weight;

    $$ endif
}

struct VolGeometry {
    size: vec3u,
    indices: array<i32,36>,
    positions: array<vec3<f32>,8>,
    texcoords: array<vec3<f32>,8>,
};

fn get_vol_geometry() -> VolGeometry {
    let size = textureDimensions(t_img);
    var geo: VolGeometry;

    geo.size = vec3u(size.xyz);

    geo.indices = array<i32,36>(
        0, 1, 2,   3, 2, 1,   4, 5, 6,   7, 6, 5,   6, 7, 3,   2, 3, 7,
        1, 0, 4,   5, 4, 0,   5, 0, 7,   2, 7, 0,   1, 4, 3,   6, 3, 4,
    );

    let pos1 = vec3<f32>(-0.5);
    let pos2 = vec3<f32>(size) + pos1;

    geo.positions = array<vec3<f32>,8>(
        vec3<f32>(pos2.x, pos1.y, pos2.z),
        vec3<f32>(pos2.x, pos1.y, pos1.z),
        vec3<f32>(pos2.x, pos2.y, pos2.z),
        vec3<f32>(pos2.x, pos2.y, pos1.z),
        vec3<f32>(pos1.x, pos1.y, pos1.z),
        vec3<f32>(pos1.x, pos1.y, pos2.z),
        vec3<f32>(pos1.x, pos2.y, pos1.z),
        vec3<f32>(pos1.x, pos2.y, pos2.z),
    );

    geo.texcoords = array<vec3<f32>,8>(
        vec3<f32>(1.0, 0.0, 1.0),
        vec3<f32>(1.0, 0.0, 0.0),
        vec3<f32>(1.0, 1.0, 1.0),
        vec3<f32>(1.0, 1.0, 0.0),
        vec3<f32>(0.0, 0.0, 0.0),
        vec3<f32>(0.0, 0.0, 1.0),
        vec3<f32>(0.0, 1.0, 0.0),
        vec3<f32>(0.0, 1.0, 1.0),
    );

    return geo;
}
