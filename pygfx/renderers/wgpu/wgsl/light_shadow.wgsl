// Shadow support

fn get_shadow(t_shadow: texture_depth_2d_array, u_shadow_sampler: sampler_comparison, layer_index: i32, shadow_coords: vec4<f32>, bias: f32) -> f32 {
    if (shadow_coords.w <= 0.0) {
        return 1.0;
    }
    // compensate for the Y-flip difference between the NDC and texture coordinates
    let proj_coords = shadow_coords.xyz / shadow_coords.w;
    let flip_correction = vec2<f32>(0.5, -0.5);
    let light_local = proj_coords.xy * flip_correction + vec2<f32>(0.5, 0.5);
    if (light_local.x < 0.0 || light_local.x > 1.0 || light_local.y < 0.0 || light_local.y > 1.0) {
        return 1.0;
    }
    var depth:f32 = proj_coords.z - bias;
    depth = saturate(depth);
    var shadow: f32 = 0.0;
    shadow += textureSampleCompareLevel(t_shadow, u_shadow_sampler, light_local, layer_index, depth, vec2<i32>(0, 0));
    shadow += textureSampleCompareLevel(t_shadow, u_shadow_sampler, light_local, layer_index, depth, vec2<i32>(1, 0));
    shadow += textureSampleCompareLevel(t_shadow, u_shadow_sampler, light_local, layer_index, depth, vec2<i32>(0, 1));
    shadow += textureSampleCompareLevel(t_shadow, u_shadow_sampler, light_local, layer_index, depth, vec2<i32>(1, 1));
    shadow += textureSampleCompareLevel(t_shadow, u_shadow_sampler, light_local, layer_index, depth, vec2<i32>(-1, 0));
    shadow += textureSampleCompareLevel(t_shadow, u_shadow_sampler, light_local, layer_index, depth, vec2<i32>(0, -1));
    shadow += textureSampleCompareLevel(t_shadow, u_shadow_sampler, light_local, layer_index, depth, vec2<i32>(-1, -1));
    shadow += textureSampleCompareLevel(t_shadow, u_shadow_sampler, light_local, layer_index, depth, vec2<i32>(-1, 1));
    shadow += textureSampleCompareLevel(t_shadow, u_shadow_sampler, light_local, layer_index, depth, vec2<i32>(1, -1));
    shadow += textureSampleCompareLevel(t_shadow, u_shadow_sampler, light_local, layer_index, depth, vec2<i32>(2, 0));
    shadow += textureSampleCompareLevel(t_shadow, u_shadow_sampler, light_local, layer_index, depth, vec2<i32>(0, 2));
    shadow += textureSampleCompareLevel(t_shadow, u_shadow_sampler, light_local, layer_index, depth, vec2<i32>(2, 2));
    shadow += textureSampleCompareLevel(t_shadow, u_shadow_sampler, light_local, layer_index, depth, vec2<i32>(-2, 0));
    shadow += textureSampleCompareLevel(t_shadow, u_shadow_sampler, light_local, layer_index, depth, vec2<i32>(0, -2));
    shadow += textureSampleCompareLevel(t_shadow, u_shadow_sampler, light_local, layer_index, depth, vec2<i32>(-2, -2));
    shadow += textureSampleCompareLevel(t_shadow, u_shadow_sampler, light_local, layer_index, depth, vec2<i32>(-2, 2));
    shadow += textureSampleCompareLevel(t_shadow, u_shadow_sampler, light_local, layer_index, depth, vec2<i32>(2, -2));
    shadow /= 17.0;
    return shadow;
}

fn uv_to_direction( face:i32,  uv:vec2<f32>)->vec3<f32> {
    var u = 2.0 * uv.x - 1.0;
    var v = -2.0 * uv.y + 1.0;
    switch face{
        case 0: { return vec3<f32>(1.0, v, -u); }
        case 1: { return vec3<f32>(-1.0, v, u); }
        case 2: { return vec3<f32>(u, 1.0, -v); }
        case 3: { return vec3<f32>(u, -1.0, v); }
        case 4: { return vec3<f32>(u, v, 1.0); }
        case 5: { return vec3<f32>(-u, v, -1.0); }
        default: { return vec3<f32>(1.0, 1.0, 1.0); }
    }
}

fn get_cube_shadow(t_shadow: texture_depth_cube_array, u_shadow_sampler: sampler_comparison, layer_index: i32,
    light_view_proj: array<mat4x4<f32>,6>, world_pos: vec3<f32>, light_direction: vec3<f32>,  bias: f32) -> f32 {
    //var direction = world_pos - light_pos;
    var direction = -light_direction;
    let scale = 1.0 / max(max(abs(direction.x), abs(direction.y)), abs(direction.z));
    direction = direction * scale;
    var faceIndex = 0;
    var view_proj: mat4x4<f32>;
    if (abs(direction.x - 1.0) < EPSILON) {
        faceIndex = 0;
        view_proj = light_view_proj[0];
    } else if (abs(direction.x + 1.0) < EPSILON) {
        faceIndex = 1;
        view_proj = light_view_proj[1];
    }  else if (abs(direction.y - 1.0) < EPSILON) {
        faceIndex = 2;
        view_proj = light_view_proj[2];
    } else if (abs(direction.y + 1.0) < EPSILON) {
        faceIndex = 3;
        view_proj = light_view_proj[3];
    } else if (abs(direction.z - 1.0) < EPSILON) {
        faceIndex = 4;
        view_proj = light_view_proj[4];
    } else if (abs(direction.z + 1.0) < EPSILON) {
        faceIndex = 5;
        view_proj = light_view_proj[5];
    }
    var shadow_coords = view_proj * vec4<f32>(world_pos, 1.0);
    if (shadow_coords.w <= 0.0) {
        return 1.0;
    }
    let proj_coords = shadow_coords.xyz / shadow_coords.w;
    let flip_correction = vec2<f32>(0.5, -0.5);
    let light_local = proj_coords.xy * flip_correction + vec2<f32>(0.5, 0.5);
    var depth:f32 = proj_coords.z - bias; // bias?
    depth = saturate(depth);
    var dir = uv_to_direction(faceIndex, light_local);
    var shadow = textureSampleCompareLevel(t_shadow, u_shadow_sampler, dir, layer_index, depth);
    return shadow;
}