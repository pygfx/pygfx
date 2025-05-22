const f32min = 0x1p-126f;
const f32max = 0x1.fffffep+127;
const PI: f32 = 3.1415926535897932385;
const TWO_PI: f32 = 6.2831853071795864769;
const FLT_MAX: f32 = 0x1.fffffep+127;
const EPSILON: f32 = 0.001;


struct Rng{
    state: u32,
}

var<private> rng: Rng;

fn init_rng(pixel_idx: u32, frame: u32) {
    // Seed the PRNG using the scalar index of the pixel and the current frame count.
    let seed = pixel_idx ^ jenkins_hash(frame);
    rng.state = jenkins_hash(seed);
}

fn jenkins_hash(i: u32) -> u32 {
    var x = i;
    x += x << 10u;
    x ^= x >> 6u;
    x += x << 3u;
    x ^= x >> 11u;
    x += x << 15u;
    return x;
}

// The 32-bit "xor" function from Marsaglia G., "Xorshift RNGs", Section 3.
fn xorshift32() -> u32 {
    var x = rng.state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    rng.state = x;
    return x;
}

// Returns a random float in the range [0...1]. This sets the floating point exponent to zero and
// sets the most significant 23 bits of a random 32-bit unsigned integer as the mantissa. That
// generates a number in the range [1, 1.9999999], which is then mapped to [0, 0.9999999] by
// subtraction. See Ray Tracing Gems II, Section 14.3.4.
fn rand_f32() -> f32 {
    return bitcast<f32>(0x3f800000u | (xorshift32() >> 9u)) - 1.;
}


fn degree_to_radian(degree: f32) -> f32 {
    return degree * PI / 180.0;
}


// Uniformly sample a unit sphere surface centered at the origin
fn sample_sphere() -> vec3f {
    let r0 = rand_f32();
    let r1 = rand_f32();

    // Map r0 to [-1, 1]
    let y = 1. - 2. * r0;

    // Compute the projected radius on the xz-plane using Pythagorean theorem
    let xz_r = sqrt(1. - y * y);

    let phi = TWO_PI * r1;
    return vec3(xz_r * cos(phi), y, xz_r * sin(phi));
}

// Uniformly sample a unit disk centered at the origin
fn sample_in_disk() -> vec3f {
    // r^2 is distributed as U(0, 1).
    let r = sqrt(rand_f32());
    let phi = TWO_PI * rand_f32();
    let x = r * cos(phi);
    let y = r * sin(phi);
    return vec3(x, y, 0.0);
}