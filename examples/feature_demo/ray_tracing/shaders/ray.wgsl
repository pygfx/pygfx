struct Ray {
    origin: vec3<f32>,
    direction: vec3<f32>,
};

fn point_on_ray(ray: Ray, t: f32) -> vec3<f32> {
    return ray.origin + t * ray.direction;
}


fn sky_color(ray: Ray) -> vec3f {
    let t = 0.5 * (normalize(ray.direction).y + 1.);
    // return (1. - t) * vec3(1.) + t * vec3(0.3, 0.5, 1.);
    return mix(vec3f(1.0), vec3(0.3, 0.5, 1.0), t);
}

struct Intersection {
    normal: vec3f,
    t: f32,
    material_index: u32,
}


fn no_intersection() -> Intersection {
    return Intersection(vec3f(0.0), -1.0, 0);
}

fn is_intersection_valid(hit: Intersection) -> bool {
    return hit.t > 0.0;
}



fn intersect_sphere(ray: Ray, sphere: Sphere) -> Intersection {
    let v = ray.origin - sphere.center;
    let a = dot(ray.direction, ray.direction);
    let b = dot(v, ray.direction);
    let c = dot(v, v) - sphere.radius * sphere.radius;

    let d = b * b - a * c;
    if d < 0.0 {
        return no_intersection();
    }

    let sqrt_d = sqrt(d);
    let recip_a = 1. / a;
    let mb = -b;
    let t1 = (mb - sqrt_d) * recip_a;
    let t2 = (mb + sqrt_d) * recip_a;
    let t = select(t2, t1, t1 > EPSILON);
    if t <= EPSILON {
        return no_intersection();
    }

    let p = point_on_ray(ray, t);
    let N = (p - sphere.center) / sphere.radius;
    return Intersection(N, t, sphere.material_index);
}


fn intersect_scene(ray: Ray) -> Intersection {
    var closest_hit = no_intersection();
    closest_hit.t = FLT_MAX;
    for (var i = 0u; i < OBJECTS_COUNT_IN_SCENE; i += 1u) {
        let sphere = spheres[i];
        // let hit = intersect_sphere(ray, sphere);
        let hit = intersect_triangle(ray, sphere);
        if hit.t > 0.0 && hit.t < closest_hit.t {
            closest_hit = hit;
        }
    }
    if closest_hit.t < FLT_MAX {
        return closest_hit;
    }
    return no_intersection();
}

// todo: use ptr
fn intersect_triangle(ray: Ray, triangle: Triangle) -> Intersection {
    // Moller–Trumbore intersection algorithm
    let edge1 = triangle.p1 - triangle.p0;
    let edge2 = triangle.p2 - triangle.p0;
    let h = cross(ray.direction, edge2);
    let a = dot(edge1, h);
    if a > -EPSILON && a < EPSILON {
        return no_intersection();
    }
    let f = 1.0 / a;
    let s = ray.origin - triangle.p0;
    let u = f * dot(s, h);
    if u < 0.0 || u > 1.0 {
        return no_intersection();
    }
    let q = cross(s, edge1);
    let v = f * dot(ray.direction, q);
    if v < 0.0 || u + v > 1.0 {
        return no_intersection();
    }
    let t = f * dot(edge2, q);
    if t > EPSILON {
        // let p = point_on_ray(ray, t);
        let p = triangle.p0 + u * edge1 + v * edge2;

        // let N = normalize(cross(edge1, edge2));
        let N = normalize(triangle.n0 * (1. - u - v) + triangle.n1 * u + triangle.n2 * v);
        return Intersection(N, t, triangle.material_index);
    }
    return no_intersection();
}


fn ray_color(primary_ray: Ray) -> vec3<f32> {
    var throughput = vec3f(1.);
    var radiance_sample = vec3(0.);

    var path_length = 0u;

    var ray = primary_ray;
    while path_length < MAX_BOUNCES {
        let hit = intersect_scene(ray);
        if !is_intersection_valid(hit) {
            // If no intersection was found, return the color of the sky and terminate the path.
            // radiance_sample += throughput * sky_color(ray);
            break;
        }

        let material = materials[hit.material_index];

        // emissive light source
        if any(material.emissive > vec3f(0.)) {
            // If the material is a light source, add its color to the radiance sample and terminate the path.
            // face
            let is_front_face = dot(ray.direction, hit.normal) < 0.;
            if is_front_face {
                radiance_sample += throughput * material.emissive;
            }
        }

        let scattered = scatter(ray, hit, material);
        throughput *= min(scattered.attenuation, vec3f(1.0));

        if all(throughput == vec3(0.)) {
            break;
        }

        ray = scattered.ray;
        path_length += 1u;
    }
    return radiance_sample;
}