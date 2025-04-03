@group(0) @binding(0)
var<storage> spheres: array<Sphere>;
@group(0) @binding(1)
var output_texture: texture_storage_2d<rgba8unorm, read_write>;
// var<storage, read_write> frame_buffer: array<vec4f>;
@group(0) @binding(2)
var<storage, read_write> rng_state_buffer: array<u32>;

@group(0) @binding(3)
var<uniform> common_uniforms: CommonUniforms;

override WORKGROUP_SIZE_X: u32;
override WORKGROUP_SIZE_Y: u32;
override OBJECTS_COUNT_IN_SCENE: u32;


@compute
@workgroup_size(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {

    // let t_size = textureDimensions(output_texture);

    let t_size = common_uniforms.viewport_size;

    if(global_id.x >= t_size.x || global_id.y >= t_size.y) {
        return;
    }

    let pos = vec2<f32>(global_id.xy);

    let idx = global_id.x + global_id.y * t_size.x;



    var rng_state = rng_state_buffer[idx];

    var cam: Camera;
    cam.image_width = f32(t_size.x);
    cam.aspect_ratio = 16.0 / 9.0;

    initCamera(&cam);

    var ray = getCameraRay(&cam, pos, &rng_state);

    var color = ray_color(ray, &rng_state);
    // var rec: HitRecord;
    // var hit_anything = false;
    // var ray_interval = Interval(0.0, INF);

    // for (var i = 0u; i < OBJECTS_COUNT_IN_SCENE; i = i + 1u) {
    //     var sphere: Sphere = spheres[i];
    //     if hit(&ray, &sphere, ray_interval, &rec) {
    //         hit_anything = true;
    //         ray_interval.max = rec.t;
    //     }
    // }

    // var color: vec3<f32>;

    // if hit_anything {
    //     color = 0.5 * (rec.normal + vec3<f32>(1.0, 1.0, 1.0));
    // }else {
    //     let unit_direction = normalize(ray.direction);
    //     let a = 0.5 * (unit_direction.y + 1.0);
    //     color = mix(vec3<f32>(0.5, 0.7, 1.0), vec3<f32>(1.0, 1.0, 1.0), a);
    //     // color = vec3f(0.0, 0.0, 1.0);
    // }

    let frame_counter = common_uniforms.frame_counter;

    if frame_counter > 0u {
        let prev_color = textureLoad(output_texture, global_id.xy);
        color = mix(prev_color.rgb, color, 1.0 / f32(frame_counter + 1u));
    }

    textureStore(output_texture, global_id.xy, vec4<f32>(color, 1.0));

    rng_state_buffer[idx] = rng_state;
    // frame_buffer[idx] = vec4<f32>(color, 1.0);
}


fn ray_color(ray: Ray, rng_state: ptr<function, u32>) -> vec3<f32> {
    var color = vec3<f32>(0.0, 0.0, 0.0); // 最终累积的颜色
    var throughput = vec3<f32>(1.0, 1.0, 1.0); // 光线的衰减因子

    var current_ray = ray; // 当前光线
    var ray_interval = Interval(0.001, INF); // 避免自相交
    var rec: HitRecord;

    let max_depth = 50; // 最大递归深度
    for (var depth = 0; depth < max_depth; depth = depth + 1) {
        var hit_anything = false;

        // 遍历场景中的所有物体，寻找最近的交点
        for (var i = 0u; i < OBJECTS_COUNT_IN_SCENE; i = i + 1u) {
            var sphere: Sphere = spheres[i];
            if hit(current_ray, &sphere, ray_interval, &rec) {
                hit_anything = true;
                ray_interval.max = rec.t;
            }
        }

        if hit_anything {
            // 生成半球内的随机方向
            // let random_vec = rngNextInUnitHemisphereN(rng_state, rec.normal);

            var random_vec = rec.normal + rngNextVec3InUnitSphere(rng_state);
            // 归一化随机向量
            random_vec = normalize(random_vec);
            
            // 更新光线的起点和方向
            current_ray = Ray(rec.p, random_vec);

            // 累积颜色
            throughput = 0.5 * throughput; // 假设材质是漫反射的，衰减因子为0.5
        } else {
            // 没有命中物体，返回背景颜色
            let unit_direction = normalize(current_ray.direction);
            let a = 0.5 * (unit_direction.y + 1.0);
            color = mix(vec3<f32>(1.0, 1.0, 1.0), vec3<f32>(0.5, 0.7, 1.0), a);
            break; // 光线逃逸，退出循环
        }
    }

    return throughput * color;
}