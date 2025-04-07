struct Sphere {
    center: vec3<f32>,
    radius: f32,
    material_index: u32,
};

struct Triangle {
    p0: vec3<f32>,
    p1: vec3<f32>,
    p2: vec3<f32>,

    n0: vec3<f32>,
    n1: vec3<f32>,
    n2: vec3<f32>,

    material_index: u32,
};

struct Material {
    color: vec3<f32>,           // 基础颜色
    metallic: f32,              // 金属度
    emissive: vec3<f32>,        // 自发光颜色
    roughness: f32,             // 粗糙度
    ior: f32,                // 折射率
};

struct Scatter {
    attenuation: vec3f,
    ray: Ray,
}

fn scatter(ray: Ray, hit: Intersection, material: Material) -> Scatter {
    let incident = normalize(ray.direction);
    let view_dir = -incident;
    
    let incident_dot_normal = dot(incident, hit.normal);
    let is_front_face = incident_dot_normal < 0.;

    let normal = select(-hit.normal, hit.normal, is_front_face);

    var scatter_dir: vec3<f32>;
    var attenuation: vec3<f32>;

    if material.ior > 0.0 {
        // refract material
        let ior = material.ior;
        let ref_ratio = select(ior, 1. / ior, is_front_face);
        let cos_theta = abs(incident_dot_normal);

        let cannot_refract = ref_ratio * ref_ratio * (1.0 - cos_theta * cos_theta) > 1.;
        if cannot_refract || schlick(cos_theta, ref_ratio) > rand_f32() {
            scatter_dir = reflect(incident, normal);
        } else {
            scatter_dir = refract(incident, normal, ref_ratio);
        }
        attenuation = material.color;

    } else {
        if material.roughness < 0.001 {
            // Specular reflection

            if material.metallic > 0.999 {
                scatter_dir = reflect(incident, normal);
                attenuation = material.color;
            } else {
                scatter_dir = reflect(incident, normal);
                let cos_theta = abs(dot(view_dir, normal));
                let r0 = vec3f(0.04);
                attenuation = r0 + (1.0 - r0) * pow(1.0 - cos_theta, 5.0);
            }
        } else {
            // Rough surface reflection
            // Use disney BRDF model
            // 1. For metallic materials, use GGX distribution sampling
            // 2. For non-metallic materials, use Lambertian distribution sampling

            if material.metallic > 0.5{
                // GGX Importance Sampling
                let alpha = max(material.roughness * material.roughness, 0.001);
                
                // 1. Generate a GGX-distributed half-vector in tangent space
                let r1 = rand_f32();
                let r2 = rand_f32();
                
                let phi = 2.0 * PI * r1;
                let cos_theta = sqrt((1.0 - r2) / (1.0 + (alpha*alpha - 1.0) * r2));
                let sin_theta = sqrt(1.0 - cos_theta * cos_theta);

                // half-vector in tangent space
                let h_local = vec3f(sin_theta * cos(phi), sin_theta * sin(phi), cos_theta);

                // 2. convert half-vector to world space
                let up = select(vec3f(0.0, 0.0, 1.0), vec3f(1.0, 0.0, 0.0), abs(normal.z) > 0.999);
                let tangent = normalize(cross(up, normal));
                let bitangent = cross(normal, tangent);
                
                let half_vector = normalize(tangent * h_local.x + bitangent * h_local.y + normal * h_local.z);

                // 3. calculate the reflection direction
                scatter_dir = reflect(incident, half_vector);

                // 4. check if in the correct hemisphere
                if dot(scatter_dir, normal) <= 0.0 {
                    // In Monte Carlo method, we should reject samples in the wrong hemisphere
                    // but for simplicity, we can use a direction in the correct hemisphere
                    scatter_dir = normal;  // fallback to normal
                }
            }else{
                // Lambertian sampling
                scatter_dir = normal + sample_sphere();
                // scatter_dir = normalize(normal + material.roughness * sample_sphere());
            }

            scatter_dir = select(scatter_dir, normal, all(scatter_dir == vec3f(0.0)));
            scatter_dir = normalize(scatter_dir);

            // todo: simplify brdf and pdf calculation, avoid repeated calculation
            let brdf = disney_brdf(normal, view_dir, scatter_dir, material);
            let half_vector = normalize(view_dir + scatter_dir);
            let pdf = mixed_pdf(normal, scatter_dir, half_vector, view_dir, material);
            attenuation = brdf / max(pdf, EPSILON);
        }

    }

    let output_ray = Ray(point_on_ray(ray, hit.t), scatter_dir);
    return Scatter(attenuation, output_ray);
}

fn disney_diffuse(normal: vec3<f32>, view_dir: vec3<f32>, light_dir: vec3<f32>, roughness: f32) -> f32 {
    let nl = max(dot(normal, light_dir), 0.0);
    if nl <= 0.0 {
        return 0.0;
    }
    let nv = max(dot(normal, view_dir), 0.0);
    let lh = max(dot(light_dir, view_dir), 0.0);

    let fd90 = 0.5 + 2.0 * roughness * lh * lh;
    let light_scatter = mix(1.0, fd90, pow(1.0 - nl, 5.0));
    let view_scatter = mix(1.0, fd90, pow(1.0 - nv, 5.0));

    return light_scatter * view_scatter * nl / PI;
}

fn disney_specular(normal: vec3<f32>, view_dir: vec3<f32>, light_dir: vec3<f32>, roughness: f32, f0: vec3<f32>) -> vec3<f32> {
    
    let half_vector = normalize(view_dir + light_dir);
    let nh = max(dot(normal, half_vector), 0.0);
    let nv = max(dot(normal, view_dir), 0.0);
    let nl = max(dot(normal, light_dir), 0.0);
    let vh = max(dot(view_dir, half_vector), 0.0);

    // GGX Normal Distribution Function
    let alpha = roughness * roughness;
    let alpha2 = alpha * alpha;
    let denom = nh * nh * (alpha2 - 1.0) + 1.0;
    let D = alpha2 / (PI * denom * denom);

    // Schlick Fresnel Approximation
    let F = f0 + (1.0 - f0) * pow(1.0 - vh, 5.0);

    // Smith Geometry Function
    let k = (roughness + 1.0) * (roughness + 1.0) / 8.0;
    let G = (nv / (nv * (1.0 - k) + k)) * (nl / (nl * (1.0 - k) + k));

    return (D * F * G) / (4.0 * nv * nl + 0.001);
}

fn disney_brdf(normal: vec3<f32>, view_dir: vec3<f32>, light_dir: vec3<f32>, material: Material) -> vec3<f32> {
    let f0 = mix(vec3f(0.04), material.color, material.metallic);
    let diffuse_color = material.color * (1.0 - material.metallic);

    let diffuse = disney_diffuse(normal, view_dir, light_dir, material.roughness) * diffuse_color;
    let specular = disney_specular(normal, view_dir, light_dir, material.roughness, f0);

    return diffuse + specular;
}

fn lambertian_pdf(normal: vec3<f32>, light_dir: vec3<f32>) -> f32 {
    let cos_theta = max(dot(normal, light_dir), 0.0);
    return cos_theta / PI;
}

fn ggx_pdf(normal: vec3<f32>, half_vector: vec3<f32>, view_dir: vec3<f32>, roughness: f32) -> f32 {
    let alpha = roughness * roughness;
    let alpha2 = alpha * alpha;
    let nh = max(dot(normal, half_vector), 0.0);
    let vh = max(dot(view_dir, half_vector), 0.0);

    // GGX Normal Distribution Function
    let D = alpha2 / (PI * pow(nh * nh * (alpha2 - 1.0) + 1.0, 2.0));

    // PDF for GGX sampling
    return (D * nh) / (4.0 * vh + 0.001);
}

fn roughness_lambertian_pdf(normal: vec3<f32>, light_dir: vec3<f32>, roughness: f32) -> f32 {
    let cos_theta = max(dot(normal, light_dir), 0.0);
    
    if (roughness < 0.001) {
        return select(0.001, 100.0, cos_theta > 0.999);
    }
    
    // 基于roughness调整分布的"尖锐度"
    // roughness越低，分布越集中在法线周围
    let concentration = 1.0 / (roughness * roughness); 
    
    // 使用modified Phong分布来近似这种行为
    // 当light_dir接近normal时，PDF值更高
    let normalization_factor = (concentration + 1.0) / (2.0 * PI);
    return normalization_factor * pow(cos_theta, concentration);
}

// fn mixed_pdf(normal: vec3<f32>, light_dir: vec3<f32>, half_vector: vec3<f32>, view_dir: vec3<f32>, material: Material) -> f32 {
//     let lambertian = lambertian_pdf(normal, light_dir);
//     let ggx = ggx_pdf(normal, half_vector, view_dir, max(material.roughness, 0.001));

//     // Fresnel term (Schlick approximation)
//     let F = fresnel_schlick(max(dot(view_dir, half_vector), 0.0), vec3f(0.04));

//     let roughness_weight = smoothstep(0.0, 0.5, material.roughness);

//     // for metallic materials, we want to mix the two PDFs based on the metallic factor
//     let metallic_factor = material.metallic;
    
//     // let weight = mix(0.5, 0.9, metallic_factor);
//     // let weight = mix(F.x, mix(0.5, 0.9, metallic_factor), metallic_factor);

//     let weight = mix(
//         mix(0.2, 0.8, 1.0 - roughness_weight),  // non-metallic weight
//         0.95,                                   // metallic weight
//         material.metallic
//     );

//     // Mix the two PDFs
//     return max(mix(lambertian, ggx, weight), 0.001);
// }

fn mixed_pdf(normal: vec3<f32>, light_dir: vec3<f32>, half_vector: vec3<f32>, view_dir: vec3<f32>, material: Material) -> f32 {
    if (material.metallic > 0.5) {
        return ggx_pdf(normal, half_vector, view_dir, max(material.roughness, 0.001));
    } else {
        return lambertian_pdf(normal, light_dir);
    }
}

fn fresnel_schlick(cos_theta: f32, f0: vec3<f32>) -> vec3<f32> {
    return f0 + (1.0 - f0) * pow(1.0 - cos_theta, 5.0);
}

fn schlick(cosine: f32, ref_ratio: f32) -> f32 {
    var r0 = (1.0 - ref_ratio) / (1.0 + ref_ratio);
    r0 = r0 * r0;
    return r0 + (1.0 - r0) * pow((1.0 - cosine), 5.0);
}

fn sample_lambertian(normal: vec3f) -> vec3f {
    return normal + sample_sphere() * (1. - EPSILON);
}