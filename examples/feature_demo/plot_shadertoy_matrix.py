from plot_shadertoy import Shadertoy

shader_code = """

// migrated from https://www.shadertoy.com/view/NlsXDH, By Kali

let det = 0.001;

var<private> t: f32;
var<private> boxhit: f32;

var<private> adv: vec3<f32>;
var<private> boxp: vec3<f32>;

fn hash(p: vec2<f32>) -> f32 {
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

fn mod_2( v: vec2<f32>, y : f32 ) -> vec2<f32> {
    return vec2<f32>(v.x - y * floor( v.x / y ), v.y - y * floor( v.y / y ));
}

fn path(t: f32) -> vec3<f32> {
    var p = vec3<f32>(sin(t*.1)*10., cos(t*.05)*10., t);
    p.x += smoothStep(.0,.5,abs(.5-fract(t*.02)))*10.;
    return p;
}

fn fractal(p: vec2<f32>) -> f32 {
    var p = abs( 5.0 - mod_2( p*0.2, 10.0 ) ) - 5.0;
    var ot = 1000.;
    for (var i = 0; i < 7; i+=1) {
        p = abs(p) / clamp(p.x*p.y, 0.25, 2.0) - 1.0;
        if (i > 0) {
            ot = min(ot, abs(p.x)+0.7*fract(abs(p.y)*0.05+t*0.05 + f32(i)*0.3));
        }
    }
    ot = exp(-10.*ot);
    return ot;
}

fn box(p: vec3<f32>, l: vec3<f32>) -> f32 {
    let c = abs(p)-l;
    return length(max(vec3<f32>(0.),c))+min(0.,max(c.x,max(c.y,c.z)));
}

fn de(p: vec3<f32>) -> f32 {
    boxhit = 0.0;
    var p2 = p-adv;

    let p2_xz = p2.xz*rot(t*0.2);
    p2.x = p2_xz.x;
    p2.z = p2_xz.y;

    let p2_xy = p2.xy*rot(t*0.1);
    p2.x = p2_xy.x;
    p2.y = p2_xy.y;

    let p2_yz = p2.yz*rot(t*0.15);
    p2.y = p2_yz.x;
    p2.z = p2_yz.y;

    let b = box(p2, vec3<f32>(1.0));

    var p = p;
    let p_xy = p.xy - path(p.z).xy;
    p.x = p_xy.x;
    p.y = p_xy.y;

    let s = sign(p.y);
    p.y = -abs(p.y) - 3.0;
    p.z = mod(p.z, 20.0) - 10.0;

    for (var i = 0; i < 5; i+=1) {
        p = abs(p) - 1.0;

        let p_xz = p.xz*rot(radians(s*-45.0));
        p.x = p_xz.x;
        p.z = p_xz.y;

        let p_yz = p.yz*rot(radians(90.0));
        p.y = p_yz.x;
        p.z = p_yz.y;

    }

    let f = -box(p, vec3<f32>(5.0, 5.0, 10.0));
    let d = min(f,b);
    if (d == b) {
        boxp = p2;
        boxhit = 1.0;
    }
    return d*0.7;
}

fn march(from: vec3<f32>, dir: vec3<f32>, frag_coord: vec2<f32>) -> vec3<f32> {
    var p = vec3<f32>(0.);
    var n = vec3<f32>(0.);
    var g = vec3<f32>(0.);

    var d = 0.0;
    var td = 0.0;

    for (var i = 0; i < 80; i+=1) {
        p = from + td*dir;
        d = de(p) * (1.0- hash( frag_coord.xy + vec2<f32>(t) )*0.3);
        if (d < det && boxhit < 0.5) {
            break;
        }
        td += max(det, abs(d));
        let f = fractal(p.xy)+fractal(p.xz)+fractal(p.yz);
        let b = fractal(boxp.xy)+fractal(boxp.xz)+fractal(boxp.yz);
        let colf = vec3<f32>(f*f,f,f*f*f);
        let colb = vec3<f32>(b+.1,b*b+.05,0.);
        g += colf / (3.0 + d*d*2.0) * exp(-0.0015*td*td) * step(5.0,td) / 2.0 * (1.0-boxhit);
        g += colb / (10.0 + d*d*20.0) * boxhit*0.5;
    }
    return g;
}

fn lookat(dir: vec3<f32>, up: vec3<f32>) -> mat3x3<f32> {
    let dir = normalize(dir);
    let rt = normalize(cross(dir, normalize(up)));
    return mat3x3<f32>(rt, cross(rt, dir), dir);
}

fn shader_main(frag_coord: vec2<f32>) -> vec4<f32> {
    let uv = (frag_coord-i_resolution.xy*.5)/i_resolution.y;
    t=i_time*7.0;
    let from=path(t);
    adv=path(t+6.+sin(t*.1)*3.);
    let dir=normalize(vec3<f32>(uv, 0.7));
    let dir=lookat(adv-from, vec3<f32>(0.0, 1.0, 0.0)) * dir;
    let col=march(from, dir, frag_coord);
    return vec4<f32>(col,1.0);
}

"""

shader = Shadertoy(shader_code, resolution=(800, 450))

if __name__ == "__main__":
    shader.show()
