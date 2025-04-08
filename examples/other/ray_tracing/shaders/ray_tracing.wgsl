@group(0) @binding(0)
var<storage> spheres: array<Triangle>;

@group(0) @binding(1)
var<storage> materials: array<Material>;

@group(0) @binding(2)
var<uniform> common_uniforms: CommonUniforms;

@group(0) @binding(3)
var<uniform> camera_uniforms: CameraUniforms;

@group(0) @binding(4)
var output_texture: texture_storage_2d<rgba32float, read_write>;

override WORKGROUP_SIZE_X: u32;
override WORKGROUP_SIZE_Y: u32;
override OBJECTS_COUNT_IN_SCENE: u32;
override MAX_BOUNCES: u32;


@compute
@workgroup_size(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {

    let t_width = common_uniforms.width;
    let t_height = common_uniforms.height;

    if(global_id.x >= t_width || global_id.y >= t_height) {
        return;
    }

    let frame_counter = common_uniforms.frame_counter;
    let pixel_idx = global_id.x + global_id.y * t_width;

    init_rng(pixel_idx, frame_counter);

    // var rng_state = rng_state_buffer[pixel_idx];
    var cam = init_camera();
    var ray = get_camera_ray(cam, global_id.xy);

    var color = ray_color(ray);

    color = clamp(color, vec3f(0.0), vec3f(15.0));

    if frame_counter > 1u {
        let prev_color = textureLoad(output_texture, global_id.xy);
        color = mix(prev_color.rgb, color, 1.0 / f32(frame_counter));
    }

    textureStore(output_texture, global_id.xy, vec4<f32>(color, 1.0));
}