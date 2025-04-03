struct Varyings {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> Varyings {
    var out: Varyings;
    if (index == u32(0)) {
        out.position = vec4<f32>(-1.0, -1.0, 0.0, 1.0);
        out.uv = vec2<f32>(0.0, 1.0);
    } else if (index == u32(1)) {
        out.position = vec4<f32>(3.0, -1.0, 0.0, 1.0);
        out.uv = vec2<f32>(2.0, 1.0);
    } else {
        out.position = vec4<f32>(-1.0, 3.0, 0.0, 1.0);
        out.uv = vec2<f32>(0.0, -1.0);
    }
    return out;
}


@group(0) @binding(0) var s: sampler;
@group(0) @binding(1) var t: texture_2d<f32>;
// @group(0) @binding(0)
// var<uniform> common_uniforms: CommonUniforms;
// @group(0) @binding(1)
// var<storage, read> frame_buffer: array<vec4f>;
@fragment
fn fs_main(in: Varyings) -> @location(0) vec4<f32> {
    // let uv = in.uv;
    // let x = u32(uv.x * f32(common_uniforms.viewport_size.x));
    // let y = u32(uv.y * f32(common_uniforms.viewport_size.y));
    // let idx = x + y * common_uniforms.viewport_size.x;
    // var color = frame_buffer[idx];

    let color = textureSample(t, s, in.uv);

    return vec4<f32>(color.rgb, 1.0);
}