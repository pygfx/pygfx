// Colormap support

fn sample_colormap(texcoord: {{ colormap_coord_type }}) -> vec4<f32> {
    
    $$ if use_colormap is defined and use_colormap
        // apply the colormap transform
        $$ if colormap_coord_type == "f32"
            let map_uv = (u_colormap.transform * vec3<f32>(texcoord, 1.0, 1.0)).x;
        $$ elif colormap_coord_type == "vec2<f32>"
            let map_uv = (u_colormap.transform * vec3<f32>(texcoord, 1.0)).xy;
        $$ elif colormap_coord_type == "vec3<f32>"
            let transformed_uv = (u_colormap.transform * vec3<f32>(texcoord.xy, 1.0));
            let map_uv = vec3f(transformed_uv.xy, texcoord.z);
        $$ endif
    $$ else
        let map_uv = texcoord;
    $$ endif

    // Sample in the colormap. We get a vec4 color, but not all channels may be used.
    $$ if not colormap_dim
        let color_value = vec4<f32>(0.0);
    $$ elif colormap_format == 'f32'
        let color_value = textureSample(t_colormap, s_colormap, map_uv);
    $$ else
        $$ if colormap_dim == '1d'
        let texcoords_dim = f32(textureDimensions(t_colormap));
        let ti = i32(map_uv * texcoords_dim % texcoords_dim);
        $$ elif colormap_dim == '2d'
        let texcoords_dim = vec2<f32>(textureDimensions(t_colormap));
        let ti = vec2<i32>(map_uv * texcoords_dim % texcoords_dim);
        $$ elif colormap_dim == '3d'
        let texcoords_dim = vec3<f32>(textureDimensions(t_colormap));
        let ti = vec3<i32>(map_uv * texcoords_dim % texcoords_dim);
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
