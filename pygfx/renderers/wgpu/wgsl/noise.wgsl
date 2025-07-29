
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
    var v = val;
    if abs(val) < 1.0 {
        v = val * 0.9 + sign(val) * 0.1; // avoid zeros
    }
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
    // White noise
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


fn _stepnoise(p1: vec2f, size: f32) -> vec2f {
    var p = p1;
    p = floor((p + 10.) / size) * size;
    p = fract( p * 0.1) + 1.0 + p * vec2f(2, 3) / 1e4;
    p = fract( 1e5 / (0.1 * p.x * (p.y + vec2f(0, 1)) + 1.0) );
    p = fract( 1e5 / ( p * vec2f(0.1234, 2.35) + 1.0) );
    return p;
}

fn blueNoise1(xy: vec2u) -> f32 {
    // https://www.shadertoy.com/view/ldyXDd
    // No idea how this works, but it produces fine results in just a few steps, though blueNoise2 produces a finer grain.
    var p = vec2f(xy);
    const dmul = 8.12235325;
    const size = 5.5;
    p += ( _stepnoise(p, size) - 0.5 ) * dmul;
    return fract( p.x * 1.705 + p.y * 0.5375 );
}

fn _xmix(x: u32, y:u32) -> u32 {
    return u32(f32((x * 212281 + y * 384817) & 0x5555555) * 0.003257328990228013);
}
fn _ymix(x: u32, y:u32) -> u32 {
    return u32(f32((x * 484829 + y * 112279) & 0x5555555) * 0.002004008016032064);
}

fn blueNoise2(xy: vec2u) -> f32 {
    // https://observablehq.com/@fil/pseudoblue
    var x = u32(xy.x);
    var y = u32(xy.y);
    const s = 8u;
    var v = 0u;
    var a: u32;
    var b: u32;
    for (var i = 0u; i < s; i+=1) {
        a = x;
        b = y;
        x = x >> 1u;
        y = y >> 1u;
        a = 1u & (a ^ _xmix(x, y));
        b = 1u & (b ^ _ymix(x, y));
        v = (v << 2u) | (a + (b << 1u) + 1u) % 4u;
    }
    return f32(v) / f32(1u << (s << 1u));
    }

fn whiteNoise(xy: vec2u) -> f32 {
    return hash_to_f32(hashu(xy.x) * hashu(xy.y));
}

fn bayerPattern(xy: vec2u) -> f32 {
    // From https://observablehq.com/@fil/pseudoblue
    // produces Bayer pattern. Looks interesting,
    // but is not so suited for blending multiple transparent layers, because there is no variation.
    var x = xy.x;
    var y = xy.y;
    var v = 0u;
    for (var i = 0u; i < 8; i+=1) {
        v = (v << 2u) | ((x & 1u) + ((y & 1u) << 1u) + 1u) % 4u;
        x >>= 1u;
        y >>= 1u;
    }
    return f32(v) / f32(1u << (8u << 1u));
}
