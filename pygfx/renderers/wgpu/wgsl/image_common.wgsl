// Common functionality for image

{$ include 'pygfx.image_sample.wgsl' $}


fn sample_im(texcoord: vec2<f32>, sizef: vec2<f32>) -> vec4<f32> {
    $$ if img_format == 'f32'
        return textureSample(t_img, s_img, texcoord.xy);
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

