// # The pygfx standard library
//
// Provides support for:
//
// * Clip planes
// * Picking
// * Constants
// * Common functions for math, transforms, colors.
//
// Also adds auto-generated code for:
//
// * FragmentOutput (implementation depends on blend mode and picking)
// * Bindings (as specified via shader.define_binding())
//
// In addition to this, the shader will also:
//
// * Add automatically generated Varyings struct (from the source).
// * Tweak the FragmentOutput if depth is set.


// ----- Constants

const PI = 3.141592653589793;
const RECIPROCAL_PI = 0.3183098861837907;
const SQRT_2 = 1.4142135623730951;

const ALPHA_COMPARE_EPSILON : f32 = 1e-6;
const EPSILON = 1e-6;


// ----- Math

fn pow2(x:f32) -> f32 { return x*x; }
fn pow4(x:f32) -> f32 { let x2 = x * x; return x2*x2; }


fn refract( light : vec3<f32>, normal : vec3<f32>, eta : f32 ) -> vec3<f32> {
    let cosTheta = dot( -light, normal );
    let rOutPerp = eta * ( light + cosTheta * normal );
    let rOutParallel = -sqrt( abs( 1.0 - dot( rOutPerp, rOutPerp ) ) ) * normal;
    return rOutPerp + rOutParallel;
}

fn hashu(val: u32 ) -> u32 {
    // https://stackoverflow.com/questions/4200224/random-noise-functions-for-glsl
    // http://amindforeverprogramming.blogspot.com/2013/07/random-floats-in-glsl-330.html
    // Bob Jenkins' OAT algorithm.
    var x: u32 = val;
    x += ( x << 10u );
    x ^= ( x >>  6u );
    x += ( x <<  3u );
    x ^= ( x >> 11u );
    x += ( x << 15u );
    return x;
}
fn hashf(val: f32 ) -> u32 {
   return hashu(bitcast<u32>(val));
}
fn hashi(val: i32 ) -> u32 {
   return hashu(bitcast<u32>(val));
}

fn hash_to_f32(h: u32) -> f32 {
    let mantissaMask: u32 = 0x007FFFFFu;
    let one: u32          = 0x3F800000u;
    var x: u32 = h;
    x &= mantissaMask;
    x |= one;
    return bitcast<f32>(x) - 1.0;
}
fn random(f: f32) -> f32 {
    // Produces a number between 0 and 1 (halfopen range). The result is deterministic based on the seed.
    return hash_to_f32( hashf(f) );
}
fn random2(f: vec2<f32>) -> f32 {
    return hash_to_f32( hashf(f.x) ^ hashf(f.y) );
}
fn random3(f: vec3<f32>) -> f32 {
    return hash_to_f32( hashf(f.x) ^ hashf(f.y) ^ hashf(f.z) );
}
fn random4(f: vec4<f32>) -> f32 {
    return hash_to_f32( hashf(f.x) ^ hashf(f.y) ^ hashf(f.z) ^ hashf(f.w) );
}

// ----- Transformations

fn ndc_to_world_pos(ndc_pos: vec4<f32>) -> vec3<f32> {
    let ndc_to_world = u_stdinfo.cam_transform_inv * u_stdinfo.projection_transform_inv;
    let world_pos = ndc_to_world * ndc_pos;
    return world_pos.xyz / world_pos.w;
}

fn is_orthographic() -> bool {
    return u_stdinfo.projection_transform[2][3] == 0.0;
}

// ----- Bindings

// Defines all bindings and functions to load from (storage) buffers
{{ bindings_code }}


// ----- Things related to output

// Implements FragmentOutput and optionally apply_virtual_fields_of_fragment_output()
{{ blending_code }}
struct StubColorWrapper {
    color: vec4<f32>,
}
// Temporary for backwards compat
fn get_fragment_output(position: vec4<f32>, color: vec4<f32>) -> StubColorWrapper {
    var wrapper : StubColorWrapper;
    wrapper.color = color;
    return wrapper;
}

fn srgb2physical(color: vec3<f32>) -> vec3<f32> {
    // In Python, the below reads as
    // c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
    let f = pow((color + 0.055) / 1.055, vec3<f32>(2.4));
    let t = color / 12.92;
    return select(f, t, color <= vec3<f32>(0.04045));
    // Simplified version with about 0.5% avg error
    // return pow(color, vec3<f32>(2.2));
}


// ----- Picking

var<private> p_pick_bits_used : i32 = 0;

fn pick_pack(value: u32, bits: i32) -> vec4<u32> {
    // Utility to pack multiple values into a rgba16uint (64 bits available).
    // Note that we store in a vec4<u32> but this gets written to a 4xu16.
    // See #212 for details.
    //
    // Clip the given value
    let maxval = u32(exp2(f32(bits))) - u32(1);
    let v = max(u32(0), min(value, maxval));
    // Determine bit-shift for each component
    let shift = vec4<i32>(
        p_pick_bits_used, p_pick_bits_used - 16, p_pick_bits_used - 32, p_pick_bits_used - 48,
    );
    // Prepare for next pack
    p_pick_bits_used = p_pick_bits_used + bits;
    // Apply the shift for each component
    let vv = vec4<u32>(v);
    let selector1 = vec4<bool>(shift[0] < 0, shift[1] < 0, shift[2] < 0, shift[3] < 0);
    let pick_new = select( vv << vec4<u32>(shift) , vv >> vec4<u32>(-shift) , selector1 );
    // Mask the components
    let mask = vec4<u32>(65535u);
    let selector2 = vec4<bool>( abs(shift[0]) < 32, abs(shift[1]) < 32, abs(shift[2]) < 32, abs(shift[3]) < 32 );
    return select( vec4<u32>(0u) , pick_new & mask , selector2 );
}

