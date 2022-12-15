from plot_shadertoy import Shadertoy

shader_code = """

// migrated from https://www.shadertoy.com/view/wlBcDK, By Kali

fn hsv2rgb(c: vec3<f32>) -> vec3<f32> {
    let K = vec4<f32>(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    let p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, vec3<f32>(0.0), vec3<f32>(1.0)), c.y);
}

fn rot(a: f32) -> mat2x2<f32> {
    let s=sin(a);
    let c=cos(a);
    return mat2x2<f32>(c, s, -s, c);
}

fn fractal(p: vec2<f32>) -> vec3<f32> {
    var p = vec2<f32>(p.x/p.y,1./p.y);
    p.y+=i_time*sign(p.y);
    p.x+=sin(i_time*.1)*sign(p.y)*4.;
    p.y=fract(p.y*.05);

    var ot1 = 1000.;
    var ot2 = ot1;
    var it = 0.;
    for (var i = 0.0; i < 10.0; i+=1.0) {
        p = abs(p);
        p = p / clamp(p.x*p.y, 0.15, 5.0) - vec2<f32>(1.5, 1.0);
        var m = abs(p.x);
        if (m < ot1) {
            ot1 = m + step(fract(i_time*0.2 + f32(i)*0.05), 0.5*abs(p.y));
            it = i;
        }
        ot2 = min(ot2, length(p));
    }

    ot1=exp(-30.0*ot1);
    ot2=exp(-30.0*ot2);
    return hsv2rgb(vec3<f32>(it*0.1+0.5,0.7,1.0))*ot1+ot2;
}

fn shader_main(frag_coord: vec2<f32>) -> vec4<f32> {
    var uv = frag_coord / i_resolution - 0.5;
    uv.x*=i_resolution.x/i_resolution.y;

    var aa = 6.0;
    uv *= rot(sin(i_time*0.1)*0.3);
    var sc = 1.0 / i_resolution.xy / (aa*2.0);
    var c = vec3<f32>(0.0);
    for (var i = -aa; i < aa; i+=1.0) {
        for (var j = -aa; j < aa; j+=1.0) {
            var p = uv + vec2<f32>(i, j)*sc;
            c += fractal(p);
        }
    }

    return vec4<f32>(c/(aa*aa*4.0)*(1.0-exp(-20.0*uv.y*uv.y)),1.0);

}

"""

shader = Shadertoy(shader_code, resolution=(800, 450))

if __name__ == "__main__":
    shader.show()
