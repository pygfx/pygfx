struct CameraUniforms {
    origin: vec3<f32>,
    aspect_ratio: f32,
    u: vec3<f32>,
    fov: f32,
    v: vec3<f32>,
    focal_length: f32,
    w: vec3<f32>,
    defocus_angle: f32,
};


struct CommonUniforms {
    width: u32,
    height: u32,
    frame_counter: u32,
};


struct Camera {
    center: vec3<f32>,
    pixel00_loc: vec3<f32>,
    pixel_delta_u: vec3<f32>,
    pixel_delta_v: vec3<f32>,

    defocus_disk_u: vec3<f32>,
    defocus_disk_v: vec3<f32>,

}

// todo: init camera in cpu
fn init_camera() -> Camera{
    var camera: Camera;
    camera.center = camera_uniforms.origin;

    let focal_length = camera_uniforms.focal_length;

    let viewport_height = 2.0 * focal_length * tan( camera_uniforms.fov * 0.5);
    let viewport_width = camera_uniforms.aspect_ratio * viewport_height;

    // Calculate the vectors across the horizontal and down the vertical viewport edges.
    let horizontal = viewport_width * camera_uniforms.u; 
    let vertical = viewport_height * -camera_uniforms.v;  // y is down in the viewport

    // Calculate the horizontal and vertical delta vectors from pixel to pixel.
    camera.pixel_delta_u = horizontal / f32(common_uniforms.width);
    camera.pixel_delta_v = vertical / f32(common_uniforms.height);

    // Calculate the location of the pixel at (0,0) in world space.

    let upper_left_corner = camera.center - horizontal / 2.0 - vertical / 2.0 + (focal_length * camera_uniforms.w);

    camera.pixel00_loc = upper_left_corner + 0.5 * (camera.pixel_delta_u + camera.pixel_delta_v);

    let defocus_radius = focal_length * tan(camera_uniforms.defocus_angle * 0.5);

    camera.defocus_disk_u = camera_uniforms.u * defocus_radius;
    camera.defocus_disk_v = camera_uniforms.v * defocus_radius;

    return camera;
}

fn get_camera_ray(camera: Camera, pixel: vec2u) -> Ray {
    let pixel_center = camera.pixel00_loc + camera.pixel_delta_u * f32(pixel.x) + camera.pixel_delta_v * f32(pixel.y);

    let pixel_sample = pixel_center + pixel_sample_square(camera);
    let ray_origin = select(defocus_disk_sample(camera), camera.center, camera_uniforms.defocus_angle <= 0.0);
    let ray_dir = normalize(pixel_sample - ray_origin);
    return Ray(ray_origin, ray_dir);
}

fn defocus_disk_sample(camera: Camera) -> vec3<f32> {
    // Returns a random point in the camera defocus disk.
    let p = sample_in_disk();
    return camera.center + (p.x * camera.defocus_disk_u) + (p.y * camera.defocus_disk_v);
}


fn pixel_sample_square(camera: Camera) -> vec3<f32> {
    let px = -0.5 + rand_f32();
    let py = -0.5 + rand_f32();
    return (px * camera.pixel_delta_u) + (py * camera.pixel_delta_v);
}