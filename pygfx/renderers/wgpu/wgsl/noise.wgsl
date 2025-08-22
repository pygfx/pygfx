fn wyman_hash2(in1: vec2f ) -> f32 {
    var in = in1;
    // if (in.x == 0.0) { in.x = 123.0; }
    // if (in.y == 0.0) { in.y = 321.0; }
    return fract( 1.0e4 * sin( 17.0*in.x + 0.1*in.y ) * ( 0.1 + abs( sin( 13.0*in.y + in.x ))));
}
fn wyman_hash3(in: vec3f) -> f32 {
    return wyman_hash2( vec2f( wyman_hash2(in.xy), in.z));
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
fn _xmix3(x: u32, y: u32, z: u32) -> u32 {
    return u32(f32((x * 212281u + y * 384817u + z * 918191u) & 0x5555555u) * 0.00325732899);
}

fn _ymix3(x: u32, y: u32, z: u32) -> u32 {
    return u32(f32((x * 484829u + y * 112279u + z * 729727u) & 0x5555555u) * 0.00200400801);
}

fn _zmix3(x: u32, y: u32, z: u32) -> u32 {
    return u32(f32((x * 338101u + y * 772349u + z * 192811u) & 0x5555555u) * 0.00163487738);
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
        let digit = (a + (b << 1u) + 1u) % 4u;  // 2 bit octree index
        v = (v << 2u) | digit;
    }
    return f32(v) / f32(1u << (s << 1u));  // denominator = 2^(2*s)
}

// A pseudo blue-noise generator for 3d input.
// ChatGTP helped me write this. It seems like the z-component is not really blue yet.
fn blueNoise3(xyz: vec3u) -> f32 {
    var x = xyz.z;
    var y = xyz.y;
    var z = xyz.x;
    const s = 8u;
    var v = 0u;
    var a: u32;
    var b: u32;
    var c: u32;
    for (var i = 0u; i < s; i += 1u) {
        let xx = x;
        let yy = y;
        let zz = z;
        x = x >> 1u;
        y = y >> 1u;
        z = z >> 1u;

        a = 1u & (xx ^ _xmix3(x, y, z));      // mix for x
        b = 1u & (yy ^ _ymix3(x, y, z));      // mix for y
        c = 1u & (zz ^ _zmix3(x, y, z));   // mix for z

        let digit = (a + (b << 1u) + (c << 2u) + 1u) % 8u;  // 3 bit octree index
        v = (v << 3u) | digit;
    }
    return f32(v) / f32(1u << (s * 3u));  // denominator = 2^(3*s).
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

fn wyman_hashed_dither(objCoord: vec3f) -> f32 {

    let g_HashScale = f32(1.0);  // he target noise scale in pixels (default 1.0).

    // Find the discretized derivatives of our coordinates
    let anisoDeriv: vec3f = max( abs(dpdx(objCoord.xyz)), abs(dpdy(objCoord.xyz)) );
    let anisoScales = vec3f(
        0.707 / (g_HashScale * anisoDeriv.x),
        0.707 / (g_HashScale * anisoDeriv.y),
        0.707 / (g_HashScale * anisoDeriv.z)
    );
    // Find log-discretized noise scales
    let scaleFlr = vec3f(
        exp2(floor(log2(anisoScales.x))),
        exp2(floor(log2(anisoScales.y))),
        exp2(floor(log2(anisoScales.z)))
    );
    let scaleCeil = vec3f(
        exp2(ceil(log2(anisoScales.x))),
        exp2(ceil(log2(anisoScales.y))),
        exp2(ceil(log2(anisoScales.z)))
    );


    // Compute alpha thresholds at our two noise scales

    // The original white noise hash in the paper
    //let alpha = vec2f(wyman_hash3(floor(scaleFlr * objCoord.xyz)), wyman_hash3(floor(scaleCeil * objCoord.xyz)));
    // A 2D variant
    //let alpha = vec2f(wyman_hash2(floor(scaleFlr.xy * objCoord.xy)), wyman_hash2(floor(scaleCeil.xy * objCoord.xy)));
    // Our blue noise version
    let alpha = vec2f(blueNoise2(vec2u(1000.0 + floor(scaleFlr * objCoord.xyz).xy)), blueNoise2(vec2u(1000.0 + floor(scaleCeil * objCoord.xyz).xy)));
    // An attempt to make the blue noise 3D, so a 3D input can be given (like a 3D model pos as in the paper)
    //let alpha = vec2f(blueNoise3(vec3u(1000.0 + floor(scaleFlr * objCoord.xyz))), blueNoise3(vec3u(1000.0 + floor(scaleCeil * objCoord.xyz))));

    // Factor to linearly interpolate with
    let fractLoc = vec3f(
        fract(log2( anisoScales.x )),
        fract(log2( anisoScales.y )),
        fract(log2( anisoScales.z ))
    );
    let toCorners = vec2f( length(fractLoc), length(vec3(1.0f)-fractLoc) );
    let lerpFactor: f32 = toCorners.x / (toCorners.x + toCorners.y);
    // Interpolate alpha threshold from noise at two scales
    let x: f32 = (1.0 - lerpFactor) * alpha.x + lerpFactor * alpha.y;
    // Pass into CDF to compute uniformly distrib threshold
    let a: f32 = min( lerpFactor, 1.0 - lerpFactor );
    let cases = vec3f(
        x * x / (2 * a * (1 - a)),
        (x - 0.5 * a) / (1 - a),
        1.0 - ((1 - x) * (1 - x) / (2 * a * (1 - a)))
    );
    // Find our final, uniformly distributed alpha threshold
    var at: f32 = cases.z;
    if  (x < (1-a)) {
        at = cases.y;
        if (x < a) {
            at = cases.x;
        }
    }
    // Avoids at == 0. Could also do at = 1-at
    at = clamp( at, 0.0, 1.0 );
    return at;
}

