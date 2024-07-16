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
// * FragmentOutput and get_fragment_output (implementation depends on blend mode and picking)
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

// Implements get_fragment_output
{{ blending_code }}


fn srgb2physical(color: vec3<f32>) -> vec3<f32> {
    // In Python, the below reads as
    // c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
    let f = pow((color + 0.055) / 1.055, vec3<f32>(2.4));
    let t = color / 12.92;
    return select(f, t, color <= vec3<f32>(0.04045));
    // Simplified version with about 0.5% avg error
    // return pow(color, vec3<f32>(2.2));
}


// ----- Clipping planes
// Use SDF for clipping planes
// Negative means inside the volume, 0 is on the surface, positive is outside.
$$ if not n_clipping_planes
    fn check_clipping_planes(world_pos: vec3<f32>) -> f32 { return -0.5; }
    fn apply_clipping_planes(world_pos: vec3<f32>) { }
$$ else
    fn check_clipping_planes(world_pos: vec3<f32>) -> f32 {
        // var clipped: bool = {{ 'false' if clipping_mode == 'ANY' else 'true' }};
        $$ if clipping_mode == 'ANY'
        // float max
        // https://github.com/gpuweb/gpuweb/issues/3431#issuecomment-1252519246
        var clip : f32 = 3.40282e+38;
        for (var i=0; i<{{ n_clipping_planes }}; i=i+1) {
            let plane = u_material.clipping_planes[i];
            clip = min(clip, dot(vec4(world_pos, -1.), plane));
        }
        $$ else
        var clip : f32 = -3.40282e+38;
        for (var i=0; i<{{ n_clipping_planes }}; i=i+1) {
            let plane = u_material.clipping_planes[i];
            clip = max(clip, dot(vec4(world_pos, -1.), plane));
        }
        $$ endif
        // Use SDF convention
        // Negative means inside the volume, 0 is on the surface, positive is outside.
        return -clip;
    }
    fn apply_clipping_planes(world_pos: vec3<f32>) {
        if (check_clipping_planes(world_pos) > 0) { discard; }
    }
$$ endif


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

