from plot_shadertoy import Shadertoy

shader_code = """

// migrated from https://www.shadertoy.com/view/tlGfzd, By Kali

var<private> objcol: vec3<f32>;

fn hash12(p: vec2<f32>) -> f32 {
    var p3 = fract(vec3<f32>(p.xyx) * 0.1031);
    p3 += vec3<f32>( dot(p3, p3.yzx + vec3<f32>(33.33)) );
    return fract((p3.x + p3.y) * p3.z);
}

fn rot(a: f32) -> mat2x2<f32> {
    let s=sin(a);
    let c=cos(a);
    return mat2x2<f32>(c, s, -s, c);
}

fn mod( x : f32, y : f32 ) -> f32 {
    return x - y * floor( x / y );
}

fn de(pos: vec3<f32>) -> f32 {
    var t = mod(i_time, 17.0);
    var a = smoothStep(13.0, 15.0, t) * 8.0 - smoothStep(4.0, 0.0, t) * 4.0;
    var f = sin(i_time * 5.0 + sin(i_time * 20.0) * 0.2);

    var pos = pos;

    let pxz = pos.xz * rot(i_time + 0.5);
    pos.x = pxz.x;
    pos.z = pxz.y;

    let pyz = pos.yz * rot(i_time);
    pos.y = pyz.x;
    pos.z = pyz.y;

    var p = pos;
    var s = 1.0;

    for (var i = 0; i < 4; i+=1) {
        p = abs(p) * 1.3 - 0.5 - f * 0.1 - a;

        let pxy = p.xy * rot(radians(45.0));
        p.x = pxy.x;
        p.y = pxy.y;

        let pxz = p.xz * rot(radians(45.0));
        p.x = pxz.x;
        p.z = pxz.y;

        s *= 1.3;
    }

    var fra = length(p) / s - 0.5;

    let pxy = pos.xy * rot(i_time);
    pos.x = pxy.x;
    pos.y = pxy.y;

    p = abs(pos) - 2.0 - a;
    var d = length(p) - 0.7;

    d = min(d, max(length(p.xz) - 0.1, p.y));
    d = min(d, max(length(p.yz) - 0.1, p.x));
    d = min(d, max(length(p.xy) - 0.1, p.z));

    p = abs(pos);
    p.x -= 4.0 + a + f * 0.5;
    d = min(d, length(p) - 0.7);
    d = min(d, length(p.yz - abs(sin(p.x * 0.5 - i_time * 10.0) * 0.3)));

    p = abs(pos);
    p.y -= 4.0 + a + f * 0.5;

    d = min(d, length(p) - 0.7);
    d = min(d, max(length(p.xz) - 0.1, p.y));
    d = min(d, fra);

    objcol = abs(p);

    if (d == fra) {
        objcol = vec3<f32>(2.0, 0.0, 0.0);
    }

    return d;
}

fn normal(p: vec3<f32>) -> vec3<f32> {
    var d = vec2<f32>(0.0, 0.01);
    return normalize( vec3<f32>( de(p + d.yxx), de(p + d.xyx), de(p + d.xxy) ) - de(p) );
}

fn march(from: vec3<f32>, dir: vec3<f32>, frag_coord: vec2<f32>) -> vec3<f32> {
    var d = 0.0;
    var td = 0.0;
    var maxdist = 30.0;

    var p = from;
    var col = vec3<f32>(0.0);
    var dir = dir;

    for (var i = 0; i < 100; i+=1) {
        var d2 = de(p) * (1.0 - hash12(frag_coord.xy + i_time) * 0.2);
        if (d2 < 0.0) {
            var n = normal(p);
            dir = reflect(dir, n);
            d2 = 0.1;
        }

        d = max(0.01, abs(d2));
        p += d * dir;
        td += d;

        if (td > maxdist) {
            break;
        }

        col += 0.01 * objcol;
    }

    return pow(col, vec3<f32>(2.0));
}

fn shader_main(frag_coord: vec2<f32>) -> vec4<f32> {
    var uv = frag_coord / i_resolution - 0.5;
    uv.x *= i_resolution.x / i_resolution.y;

    var from = vec3<f32>(0.0, 0.0, -10.0);
    var dir = normalize(vec3<f32>(uv, 1.0));

    var col = march(from, dir, frag_coord);

    return vec4<f32>(col, 1.0);
}

"""

shader = Shadertoy(shader_code, resolution=(800, 450))

if __name__ == "__main__":
    shader.show()
