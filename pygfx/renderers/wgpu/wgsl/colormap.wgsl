// Colormap support

fn sample_colormap(texcoord: {{ colormap_coord_type }}) -> vec4<f32> {

    // Determine colormap texture dimensions
    $$ if colormap_dim == '1d'
        let texcoords_dim = f32(textureDimensions(t_colormap));
    $$ elif colormap_dim == '2d'
        let texcoords_dim = vec2<f32>(textureDimensions(t_colormap));
    $$ elif colormap_dim == '3d'
        let texcoords_dim = vec3<f32>(textureDimensions(t_colormap));
    $$ endif

    // Get final texture coord. With linear interpolation, the colormap's endpoints represent the min and max.
    $$ if colormap_interpolation == 'nearest'
        let tf = texcoord;
    $$ else
        let tf = texcoord * (texcoords_dim - 1.0) / texcoords_dim + 0.5 / texcoords_dim;
    $$ endif

    // Sample in the colormap. We get a vec4 color, but not all channels may be used.
    $$ if not colormap_dim
        let color_value = vec4<f32>(0.0);
    $$ elif colormap_format == 'f32'
        let color_value = textureSample(t_colormap, s_colormap, tf);
    $$ else
        $$ if colormap_dim == '1d'
        let ti = i32(tf * texcoords_dim % texcoords_dim);
        $$ elif colormap_dim == '2d'
        let ti = vec2<i32>(tf * texcoords_dim % texcoords_dim);
        $$ elif colormap_dim == '3d'
        let ti = vec3<i32>(tf * texcoords_dim % texcoords_dim);
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

