from plot_shadertoy import Shadertoy

shader_code = """

// migrated from https://www.shadertoy.com/view/3sGfD3, By Kali

fn rot(a: f32) -> mat2x2<f32> {
    let s=sin(a);
    let c=cos(a);
    return mat2x2<f32>(c, s, -s, c);
}

fn render(p: vec2<f32>) -> vec3<f32> {
    var p = p;
    p*=rot(i_time*.1)*(.0002+.7*pow(smoothStep(0.0,0.5,abs(0.5-fract(i_time*.01))),3.));
    p.y-=.2266;
    p.x+=.2082;
    var ot = vec2<f32>(100.0);
    var m = 100.0;
    for (var i = 0; i < 150; i+=1) {
        var cp = vec2<f32>(p.x,-p.y);
        p=p+cp/dot(p,p)-vec2<f32>(0.0,0.25);
        p*=.1;
        p*=rot(1.5);
        ot=min(ot,abs(p)+.15*fract(max(abs(p.x),abs(p.y))*.25+i_time*0.1+f32(i)*0.15));
        m=min(m,abs(p.y));
    }
    ot=exp(-200.*ot)*2.;
    m=exp(-200.*m);
    return vec3<f32>(ot.x,ot.y*.5+ot.x*.3,ot.y)+m*.2;
}

fn shader_main(frag_coord: vec2<f32>) -> vec4<f32> {
    let uv = (frag_coord - i_resolution / 2.0) / i_resolution.y;
    let d=vec2<f32>(0.0,0.5)/i_resolution;
    let col = render(uv)+render(uv+d.xy)+render(uv-d.xy)+render(uv+d.yx)+render(uv-d.yx);
    return vec4<f32>(col/5.0, 1.0);
}


"""
shader = Shadertoy(shader_code, resolution=(800, 450))

if __name__ == "__main__":
    shader.show()
