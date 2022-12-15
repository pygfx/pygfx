from plot_shadertoy import Shadertoy

shader_code = """

fn render(p: vec2<f32>) -> vec3<f32> {
    let s = sin(i_time) * sin(i_time) * sin(i_time) + 0.5;
    var p = p;
    var d = length(p * 0.8) - pow(2.0 * abs(0.5 - fract(atan2(p.y, p.x) / 3.1416 * 2.5 + i_time * 0.3)), 2.5) * 0.1;

    var col = vec3<f32>(0.0);
    // star
    col += 0.01 / (d * d) * vec3<f32>(0.5, 0.7, 1.5) * (1.0 + s);

    // spiral
    d = sin(length(p * 2.0) * 10.0 - i_time * 3.0) - sin(atan2(p.y, p.x) * 5.0) * 1.0;
    col += 0.1 / (0.2 + d * d) * vec3<f32>(1.0, 0.0, 1.0);

    // background
    for (var i = 0; i < 6; i+=1) {
        p = abs(p) / dot(p, p) - 1.0;
    }
    col += vec3<f32>(0.001 / dot(p, p));

    return col;
}

fn shader_main(frag_coord: vec2<f32>) -> vec4<f32> {

    let uv = (frag_coord-i_resolution*0.5)/i_resolution.y;

    let col = render(uv);

    return vec4<f32>(col,1.0);

}

"""

shader = Shadertoy(shader_code, resolution=(800, 450))

if __name__ == "__main__":
    shader.show()
