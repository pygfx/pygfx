
fn sampled_value_to_color(value_rgba: vec4<f32>) -> vec4<f32> {

    // Make it the correct dimension
    $$ if img_nchannels == 1
        let value_raw = value_rgba.r;
    $$ elif img_nchannels == 2
        let value_raw = value_rgba.rg;
    $$ elif img_nchannels == 3
        let value_raw = value_rgba.rgb;
    $$ else
        let value_raw = value_rgba.rgba;
    $$ endif

    // Apply contrast limits
    let value_cor = value_raw {{ climcorrection }};
    let value_clim = (value_cor - u_material.clim[0]) / (u_material.clim[1] - u_material.clim[0]);

    // Apply colormap or compose final color
    $$ if colormap_dim
        // In the render function we make sure that colormap_dim matches img_nchannels
        let color = sample_colormap(value_clim);
    $$ else
        $$ if img_nchannels == 1
            let r = value_clim;
            let color = vec4<f32>(r, r, r, 1.0);
        $$ elif img_nchannels == 2
            let color = vec4<f32>(value_clim.rrr, value_raw.g);
        $$ elif img_nchannels == 3
            let color = vec4<f32>(value_clim.rgb, 1.0);
        $$ else
            let color = vec4<f32>(value_clim.rgb, value_raw.a);
        $$ endif
    $$ endif

    return color;
}