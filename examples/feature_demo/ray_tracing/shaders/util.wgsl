const f32min = 0x1p-126f;
const f32max = 0x1.fffffep+127;
const PI: f32 = 3.1415926535897932385;
const INF: f32 = 0x1.fffffep+127;


fn degree_to_radian(degree: f32) -> f32 {
    return degree * PI / 180.0;
}

fn rngNextFloat(state: ptr<function, u32>) -> f32 {
    let x = rngNextInt(state);
    return f32(x) / f32(0xffffffffu);
}

fn rngNextInt(state: ptr<function, u32>) -> u32 {
    // PCG random number generator
    // Based on https://www.shadertoy.com/view/XlGcRh

    let new_state = *state * 747796405u + 2891336453u;
    *state = new_state;
    let word = ((new_state >> ((new_state >> 28u) + 4u)) ^ new_state) * 277803737u;
    return (word >> 22u) ^ word;
}

fn randInRange(min: f32, max: f32, state: ptr<function, u32>) -> f32 {
    return min + rngNextFloat(state) * (max - min);
}

fn rngNextVec3InUnitDisk(state: ptr<function, u32>) -> vec3<f32> {
    // Generate numbers uniformly in a disk:
    // https://stats.stackexchange.com/a/481559

    // r^2 is distributed as U(0, 1).
    let r = sqrt(rngNextFloat(state));
    let alpha = 2f * PI * rngNextFloat(state);

    let x = r * cos(alpha);
    let y = r * sin(alpha);

    return vec3(x, y, 0f);
}

fn rngNextVec3InUnitSphere(state: ptr<function, u32>) -> vec3<f32> {
    let r = pow(rngNextFloat(state), 0.33333f);
    let cosTheta = 1f - 2f * rngNextFloat(state);
    let sinTheta = sqrt(1f - cosTheta * cosTheta);
    let phi = 2f * PI * rngNextFloat(state);

    let x = r * sinTheta * cos(phi);
    let y = r * sinTheta * sin(phi);
    let z = cosTheta;

    return vec3(x, y, z);
}

fn rngNextInUnitHemisphere(state: ptr<function, u32>) -> vec3<f32> {
    let r1 = rngNextFloat(state);
    let r2 = rngNextFloat(state);

    let phi = 2f * PI * r1;
    let sinTheta = sqrt(1f - r2 * r2);

    let x = cos(phi) * sinTheta;
    let y = sin(phi) * sinTheta;
    let z = r2;

    return vec3(x, y, z);
}

fn rngNextInUnitHemisphereN(state: ptr<function, u32>, normal: vec3<f32>) -> vec3<f32> {
    let random_vec = rngNextInUnitHemisphere(state);

    var tangent: vec3<f32>;
    if (abs(normal.z) < 0.999f) {
        tangent = normalize(cross(vec3<f32>(0.0, 0.0, 1.0), normal));
    } else {
        tangent = normalize(cross(vec3<f32>(1.0, 0.0, 0.0), normal));
    }
    let bitangent = cross(normal, tangent);

    return random_vec.x * tangent + random_vec.y * bitangent + random_vec.z * normal;
}