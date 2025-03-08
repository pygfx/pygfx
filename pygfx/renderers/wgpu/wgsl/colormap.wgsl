// Colormap support

fn sample_colormap(texcoord: {{ colormap_coord_type }}) -> vec4<f32> {

    // Sample in the colormap. We get a vec4 color, but not all channels may be used.
    $$ if not colormap_dim
        let color_value = vec4<f32>(0.0);
    $$ elif colormap_format == 'f32'
        let color_value = textureSample(t_colormap, s_colormap, texcoord);
    $$ else
        $$ if colormap_dim == '1d'
        let texcoords_dim = f32(textureDimensions(t_colormap));
        let ti = i32(texcoord * texcoords_dim % texcoords_dim);
        $$ elif colormap_dim == '2d'
        let texcoords_dim = vec2<f32>(textureDimensions(t_colormap));
        let ti = vec2<i32>(texcoord * texcoords_dim % texcoords_dim);
        $$ elif colormap_dim == '3d'
        let texcoords_dim = vec3<f32>(textureDimensions(t_colormap));
        let ti = vec3<i32>(texcoord * texcoords_dim % texcoords_dim);
        $$ endif
        let color_value = vec4<f32>(textureLoad(t_colormap, ti, 0));
    $$ endif

    // Depending on the number of channels we makeGfxTextureView grayscale, rgb, etc.
    $$ if colormap_nchannels == 1
        let color = vec4<f32>(color_value.rrr, 1.0);
    $$ elif colormap_nchannels == 2
        let color = vec4<f32>(color_value.rrr, color_value.g);
    $$ elif colormap_nchannels == 3
        let color = vec4<f32>(color_value.rgb, 1.0);
    $$ else
        let color = vec4<f32>(color_value.rgb, color_value.a);
    $$ endif
    return color;
}
