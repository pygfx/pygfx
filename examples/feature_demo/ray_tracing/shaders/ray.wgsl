struct Sphere {
    center: vec3<f32>,
    radius: f32,
    albedo: vec3<f32>,
    material_type: i32,
};

struct Ray {
    origin: vec3<f32>,
    direction: vec3<f32>,
};

struct HitRecord {
    t: f32,
    p: vec3<f32>,
    normal: vec3<f32>,
    front_face: bool,
    // material_type: i32,
    // mesh_id: i32,
};

struct Interval {
    min: f32,
    max: f32,
}

fn ray_at(ray: ptr<function, Ray>, t: f32) -> vec3<f32> {
    return (*ray).origin + t * (*ray).direction;
}

fn hit(ray: Ray, sphere_p: ptr<function, Sphere>, ray_t: Interval, rec_p: ptr<function, HitRecord>) -> bool {
    var sphere = *sphere_p;
    
    let oc = sphere.center - ray.origin;
    let a = pow(length(ray.direction), 2.0);
    let h = dot(ray.direction, oc);
    let c = pow(length(oc), 2.0) - sphere.radius * sphere.radius;
    let discriminant = h * h - a * c;

    if (discriminant < 0.0) {
        return false;
    }

    let sqrt_d = sqrt(discriminant);

    // find the nearest root in [ray_t.min, ray_t.max]
    var root = (h - sqrt_d) / a;
    if (root < ray_t.min || root > ray_t.max) {
        root = (h + sqrt_d) / a;
        if (root < ray_t.min || root > ray_t.max) {
            return false;
        }
    }


    (*rec_p).t = root;
    (*rec_p).p = ray.origin + (*rec_p).t * ray.direction;
    (*rec_p).normal = ((*rec_p).p - sphere.center) / sphere.radius;
    // rec.normal = normalize(rec.p - sphere.center);
    (*rec_p).front_face = dot(ray.direction, (*rec_p).normal) < 0.0;
    if (!(*rec_p).front_face) {
        (*rec_p).normal = -(*rec_p).normal;
    }

    return true;
}
