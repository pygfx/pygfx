// Common functionality for volumes

{$ include 'pygfx.image_sample.wgsl' $}


fn sample_vol(texcoord: vec3<f32>, sizef: vec3<f32>) -> vec4<f32> {
    $$ if img_format == 'f32'
        return textureSample(t_img, s_img, texcoord.xyz);
    $$ else
        let texcoords_u = vec3<i32>(texcoord.xyz * sizef);
        return vec4<f32>(textureLoad(t_img, texcoords_u, 0));
    $$ endif
}

struct VolGeometry {
    indices: array<i32,36>,
    positions: array<vec3<f32>,8>,
    texcoords: array<vec3<f32>,8>,
};

fn get_vol_geometry() -> VolGeometry {
    let size = textureDimensions(t_img);
    var geo: VolGeometry;

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
