from plot_shadertoy import Shadertoy

shader_code = """

// migrated from: https://www.shadertoy.com/view/csjGDD, By Kali

var<private> det : f32 = 0.001;
var<private> br : f32 = 0.0;
var<private> tub : f32 = 0.0;
var<private> hit : f32 = 0.0;

var<private> pos: vec3<f32>;
var<private> sphpos: vec3<f32>;


fn lookat(dir: vec3<f32>, up: vec3<f32>) -> mat3x3<f32> {
    let rt = normalize(cross(dir, up));
    return mat3x3<f32>(rt, cross(rt, dir), dir);
}

fn path(t: f32) -> vec3<f32> {
    return vec3<f32>(sin(t+cos(t)*0.5)*0.5, cos(t*0.5), t);
}

fn rot(a: f32) -> mat2x2<f32> {
    let s=sin(a);
    let c=cos(a);
    return mat2x2<f32>(c, s, -s, c);
}

fn fractal(p: vec2<f32>) -> vec3<f32> {
    var p = fract(p*0.1);
    var m = 1000.0;
    for (var i = 0; i < 7; i = i + 1) {
        p = ( abs(p) / clamp( abs(p.x*p.y), 0.25, 2.0 ) ) - 1.2;
        m = min(m,abs(p.y)+fract(p.x*0.3 + i_time*0.5 + f32(i)*0.25));
    }
    m=exp(-6.0 * m);
    return m*vec3<f32>(abs(p.x),m,abs(p.y));
}

fn coso(pp: vec3<f32>) -> f32 {
    var pp = pp;
    pp*=.7;

    pp = vec3<f32>( pp.xy * rot(pp.z*2.0), pp.z);

    let ppxz = pp.xz * rot(i_time*2.0);
    pp = vec3<f32>(ppxz.x, pp.y, ppxz.y);

    pp = vec3<f32>(pp.x, pp.yz * rot(i_time));

    var sph=length(pp) - 0.04;
    sph-=length(sin(pp*40.))*.05;
    sph=max(sph,-length(pp)+.11);
    var br2=length(pp) - 0.03;
    br2=min(br2,length(pp.xy)+.005);
    br2=min(br2,length(pp.xz)+.005);
    br2=min(br2,length(pp.yz)+.005);
    br2=max(br2,length(pp) - 1.0);
    br=min(br2,br);
    let d=min(br,sph);
    return d;
}

fn de(p: vec3<f32>) -> f32 {
    hit=0.;
    br=1000.;
    let pp = p - sphpos;
    var p = p;
    let pxy = p.xy - path(p.z).xy;
    p.x = pxy.x;
    p.y = pxy.y;
    let rxy = p.xy * rot(p.z + i_time* 0.5);
    p.x = rxy.x;
    p.y = rxy.y;

    let s = sin(p.z*0.5 + i_time * 0.5);
    p.x *= ( 1.3 - s*s*0.7 );
    p.y *= ( 1.3 - s*s*0.7 );

    for(var i=0; i<6; i+=1) {
        p = abs(p) - 0.4;
    }

    pos = p;

    tub = -length(p.xy) + 0.45 + sin(p.z*10.0) * 0.1 * smoothStep(0.4,0.5,abs(0.5-fract(p.z*0.05))*2.0);
    var co = coso(pp);
    co=min(co, coso(pp + 0.7) );
    co=min(co, coso(pp - 0.7) );

    let d = min(tub,co);
    if (d==tub) {
        hit = step(fract(0.1 * length(sin(p*10.0))), 0.05);
    }
    return d * 0.3;
}


fn march(from: vec3<f32>, dir: vec3<f32>) -> vec3<f32> {
    var dir = dir;
    var uv: vec2<f32> = vec2<f32>( atan2( dir.x , dir.y ) + i_time * 0.5, length(dir.xy) + sin(i_time * 0.2));
    var col: vec3<f32> = fractal(uv);
    var d: f32 = 0.0;
    var td: f32 = 0.0;
    var g: f32 = 0.0;
    var ref: f32 = 0.0;
    var ltd: f32 = 0.0;
    var li: f32 = 0.0;
    var p: vec3<f32> = from;
    for(var i: i32 = 0; i < 200; i += 1) {
        p += dir * d;
        d = de(p);
        if (d < det && ref == 0.0 && hit == 1.0) {
            var e: vec2<f32> = vec2<f32>(0.0, 0.1);
            var n: vec3<f32> = normalize(vec3<f32>(de(p + e.yxx), de(p + e.xyx), de(p + e.xxy)) - de(p));
            p -= dir * d * 2.0;
            dir = reflect(dir, n);
            ref = 1.0;
            td = 0.0;
            ltd = td;
            continue;
        }
        if (d < det || td > 5.0) {
            break;
        }
        td += d;
        g += 0.1 / (0.1 + br * 13.0);
        li += 0.1 / (0.1 + tub * 5.0);
    }
    g = max(g, li * 0.15);
    var f: f32 = 1.0 - td / 3.0;
    if (ref == 1.0) {
        f = 1.0 - ltd / 3.0;
    }
    if (d < 0.01) {
        col = vec3<f32>(1.0);
        var e: vec2<f32> = vec2<f32>(0.0, det);
        var n: vec3<f32> = normalize(vec3<f32>( de(p + e.yxx), de(p + e.xyx), de(p + e.xxy)) - de(p));
        col = vec3<f32>(n.x) * 0.7;
        col += fract(pos.z * 5.0) * vec3<f32>(0.2, 0.1, 0.5);
        col += fractal(pos.xz * 2.0);
        if (tub > 0.01) {
            col = vec3<f32>(0.0);
        }
    }
    col *= f;
    let so = fract(sin(i_time)*123.456);
    var glo: vec3<f32> = g * 0.1 * vec3<f32>(2.0, 1.0, 2.0) * (0.5 + so * 1.5) * 0.5;

    let glo_rb = glo.rb * rot(dir.y * 1.5);
    glo = vec3<f32>(glo_rb.x, glo.y, glo_rb.y);
    col += glo;
    col *= vec3<f32>(0.8, 0.7, 0.7);
    col = mix(col, vec3<f32>(1.0), ref * 0.3);
    return col;
}

fn mod( x : f32, y : f32 ) -> f32 {
    return x - y * floor( x / y );
}

fn shader_main(frag_coord : vec2<f32>) -> vec4<f32> {
    var uv = frag_coord / i_resolution;
    uv = uv - 0.5;
    uv /= vec2<f32>(i_resolution.y / i_resolution.x, 1.0);

    var t = i_time;

    var from = path(t);
    if (mod(t, 10.0) > 5.0) {
        from = path(floor(t / 4.0 + 0.5) * 4.0);
    }
    sphpos = path(t + 0.5);
    from.x += 0.2;
    var fw = normalize(path(t + 0.5) - from);
    var dir = normalize(vec3<f32>(uv, 0.5));
    dir = lookat(fw, vec3<f32>(fw.x * 2.0, 1.0, 0.0)) * dir;
    dir = vec3<f32>(dir.x+sin(t) * 0.3, dir.y, dir.z+sin(t) * 0.3);
    var col = march(from, dir);
    col = mix(vec3<f32>(0.5) * length(col), col, 0.8);
    return vec4<f32>(col, 1.0);
}

"""
shader = Shadertoy(shader_code, resolution=(800, 450))

if __name__ == "__main__":
    shader.show()
