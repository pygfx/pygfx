
fn sampled_value_to_color(value_rgba: vec4<f32>) -> vec4<f32> {

    // Make it the correct dimension
    $$ if img_nchannels == 1
        let value_raw = value_rgba.r;
        let gamma_vec = u_material.gamma;
    $$ elif img_nchannels == 2
        let value_raw = value_rgba.rg;
        let gamma_vec = vec2<f32>(u_material.gamma);
    $$ elif img_nchannels == 3
        let value_raw = value_rgba.rgb;
        let gamma_vec = vec3<f32>(u_material.gamma);
    $$ else
        let value_raw = value_rgba.rgba;
        let gamma_vec = vec4<f32>(u_material.gamma);
    $$ endif

    // Apply contrast limits
    let value_cor = value_raw {{ climcorrection }};
    let value_clim = (value_cor - u_material.clim[0]) / (u_material.clim[1] - u_material.clim[0]);

    // Apply gamma correction
    let value_gamma = saturate(pow(value_clim, gamma_vec));

    // Apply colormap or compose final color
    $$ if colormap_dim
        // In the render function we make sure that colormap_dim matches img_nchannels
        let color = sample_colormap(value_gamma);
    $$ else
        $$ if img_nchannels == 1
            let r = value_gamma;
            let color = vec4<f32>(r, r, r, 1.0);
        $$ elif img_nchannels == 2
            let color = vec4<f32>(value_gamma.rrr, value_raw.g);
        $$ elif img_nchannels == 3
            let color = vec4<f32>(value_gamma.rgb, 1.0);
        $$ else
            let color = vec4<f32>(value_gamma.rgb, value_raw.a);
        $$ endif
    $$ endif

    return color;
}