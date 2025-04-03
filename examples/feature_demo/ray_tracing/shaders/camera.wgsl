struct Camera {
    viewport_size: vec2u,
    image_width: f32,
    image_height: f32,
    pixel00_loc: vec3<f32>,
    pixel_delta_u: vec3<f32>,
    pixel_delta_v: vec3<f32>,

    aspect_ratio: f32,
    center: vec3<f32>,
    // vfov: f32,

    // lookFrom: vec3f,
    // lookAt: vec3f,
    // vup: vec3f,

    // defocusAngle: f32,
    // focusDist: f32,

    // defocusDiscU: vec3f,
    // defocusDiscV: vec3f,
}

fn initCamera(camera: ptr<function, Camera>) {
    (*camera).image_height = (*camera).image_width / (*camera).aspect_ratio;
    (*camera).image_height = select((*camera).image_height, 1.0, (*camera).image_height < 1.0);

    // (*camera).center = (*camera).lookFrom;

    // Determine viewport dimensions.

    let focal_length = 1.0;
    let viewport_height = 2.0;
    let viewport_width = (*camera).aspect_ratio * viewport_height;

    // Calculate the vectors across the horizontal and down the vertical viewport edges.

    let viewport_u = vec3<f32>(viewport_width, 0.0, 0.0);
    let viewport_v = vec3<f32>(0.0, -viewport_height, 0.0);

    // Calculate the horizontal and vertical delta vectors from pixel to pixel.

    (*camera).pixel_delta_u = viewport_u / (*camera).image_width;
    (*camera).pixel_delta_v = viewport_v / (*camera).image_height;

    // Calculate the location of the pixel at (0,0) in world space.
    // This is the lower left corner of the viewport.

    let lower_left_corner = (*camera).center - viewport_u / 2.0 - viewport_v / 2.0 - vec3<f32>(0.0, 0.0, focal_length);

    (*camera).pixel00_loc = lower_left_corner + 0.5 * ((*camera).pixel_delta_u + (*camera).pixel_delta_v);

    // let theta = radians((*camera).vfov);
    // let h = tan(theta * 0.5);
    // let viewportHeight = 2.0 * h * (*camera).focusDist;
    // let viewportWidth = viewportHeight * ((*camera).imageWidth / (*camera).imageHeight);

    // let w = normalize((*camera).lookFrom - (*camera).lookAt);
    // let u = normalize(cross((*camera).vup, w));
    // let v = cross(w, u);

    // let viewportU = viewportWidth * u;
    // let viewportV = viewportHeight * -v;

    // (*camera).pixelDeltaU = viewportU / (*camera).imageWidth;
    // (*camera).pixelDeltaV = viewportV / (*camera).imageHeight;

    // let viewportUpperLeft = (*camera).center - ((*camera).focusDist * w) - viewportU / 2 - viewportV / 2;
    // (*camera).pixel00Loc = viewportUpperLeft + 0.5 * ((*camera).pixelDeltaU + (*camera).pixelDeltaV);

    // let defocusRadius = (*camera).focusDist * tan(radians((*camera).defocusAngle * 0.5));
    // (*camera).defocusDiscU = u * defocusRadius;
    // (*camera).defocusDiscV = v * defocusRadius;
}

fn getCameraRay(camera: ptr<function, Camera>, pixel: vec2f, rng_state: ptr<function, u32>) -> Ray {
    let pixel_center = (*camera).pixel00_loc + (*camera).pixel_delta_u * pixel.x + (*camera).pixel_delta_v * pixel.y;

    let pixel_sample = pixel_center + pixelSampleSquare(camera, rng_state);

    // 散焦
    // let ray_origin = select(defocusDiskSample(camera, rng_state), (*camera).center, (*camera).defocusAngle <= 0);
    let ray_origin = (*camera).center;
    let ray_dir = pixel_sample - ray_origin;
    return Ray(ray_origin, ray_dir);
}


fn pixelSampleSquare(camera: ptr<function, Camera>, rng_state: ptr<function, u32>) -> vec3<f32> {
    let px = -0.5 + rngNextFloat(rng_state);
    let py = -0.5 + rngNextFloat(rng_state);
    return (px * (*camera).pixel_delta_u) + (py * (*camera).pixel_delta_v);
}