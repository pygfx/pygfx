$$ if USE_TRANSMISSION is defined

// Mipped Bicubic Texture Filtering by N8
// https://www.shadertoy.com/view/Dl2SDW

fn w0( a: f32 ) -> f32 {
    return ( 1.0 / 6.0 ) * ( a * ( a * ( - a + 3.0 ) - 3.0 ) + 1.0 );
}

fn w1( a: f32 ) -> f32 {
    return ( 1.0 / 6.0 ) * ( a *  a * ( 3.0 * a - 6.0 ) + 4.0 );
}

fn w2( a: f32 ) -> f32 {
    return ( 1.0 / 6.0 ) * ( a * ( a * ( - 3.0 * a + 3.0 ) + 3.0 ) + 1.0 );
}

fn w3( a: f32 ) -> f32 {
    return ( 1.0 / 6.0 ) * ( a * a * a );
}

// g0 and g1 are the two amplitude functions
fn g0( a: f32 ) -> f32 {
    return w0( a ) + w1( a );
}

fn g1( a: f32 ) -> f32 {
    return w2( a ) + w3( a );
}

// h0 and h1 are the two offset functions
fn h0( a: f32 ) -> f32 {
    return - 1.0 + w1( a ) / ( w0( a ) + w1( a ) );
}

fn h1( a: f32 ) -> f32 {
    return 1.0 + w3( a ) / ( w2( a ) + w3( a ) );
}

fn bicubic( tex: texture_2d<f32>, s: sampler,  uv: vec2f, texel_size: vec4f, lod: f32 ) -> vec4f {
    let uv_scaled = uv * texel_size.zw + 0.5;
    let iuv = floor( uv_scaled );
    let fuv = fract( uv_scaled );

    let g0x = g0( fuv.x );
    let g1x = g1( fuv.x );
    let h0x = h0( fuv.x );
    let h1x = h1( fuv.x );
    let h0y = h0( fuv.y );
    let h1y = h1( fuv.y );

    let p0 = ( vec2f( iuv.x + h0x, iuv.y + h0y ) - 0.5 ) * texel_size.xy;
    let p1 = ( vec2f( iuv.x + h1x, iuv.y + h0y ) - 0.5 ) * texel_size.xy;
    let p2 = ( vec2f( iuv.x + h0x, iuv.y + h1y ) - 0.5 ) * texel_size.xy;
    let p3 = ( vec2f( iuv.x + h1x, iuv.y + h1y ) - 0.5 ) * texel_size.xy;

    return g0( fuv.y ) * ( g0x * textureSampleLevel( tex, s, p0, lod ) + g1x * textureSampleLevel( tex, s, p1, lod ) ) +
        g1( fuv.y ) * ( g0x * textureSampleLevel( tex, s, p2, lod ) + g1x * textureSampleLevel( tex, s, p3, lod ) );
}


fn texture_bicubic( tex: texture_2d<f32>, s: sampler, uv: vec2f, lod: f32 ) -> vec4f {
    let f_lod_size = vec2f( textureDimensions( tex, i32(lod) ) );
    let c_lod_size = vec2f( textureDimensions( tex, i32(lod + 1.0) ) );
    let f_lod_size_inv = 1.0 / f_lod_size;
    let c_lod_size_inv = 1.0 / c_lod_size;
    let f_sample = bicubic( tex, s, uv, vec4f( f_lod_size_inv, f_lod_size ), floor( lod ) );
    let c_sample = bicubic( tex, s, uv, vec4f( c_lod_size_inv, c_lod_size ), ceil( lod ) );
    return mix( f_sample, c_sample, fract( lod ) );
}

fn get_volume_transmission_ray( n: vec3f, v: vec3f, thickness: f32, ior: f32, model_matrix: mat4x4<f32> ) -> vec3f {
    // Direction of refracted light.
    let refraction_vector = refract( -v, normalize( n ), 1.0 / ior );

    // Compute rotation-independent scaling of the model matrix.
    let model_scale = vec3f(
        length( model_matrix[0].xyz ),
        length( model_matrix[1].xyz ),
        length( model_matrix[2].xyz )
    );

    // The thickness is specified in local space.
    return normalize( refraction_vector ) * thickness * model_scale;
}

fn apply_ior_to_roughness( roughness: f32, ior: f32 ) -> f32 {
    // Scale roughness with IOR so that an IOR of 1.0 results in no microfacet refraction and
    // an IOR of 1.5 results in the default amount of microfacet refraction.
    return roughness * clamp( ior * 2.0 - 2.0, 0.0, 1.0 );
}

fn get_transmission_sample( frag_coord: vec2f, roughness: f32, ior: f32 ) -> vec4f {
    // transmission_sampler_map
    let size = textureDimensions( t_transmission_framebuffer, 0 );
    let lod = log2( f32(size.x) ) * apply_ior_to_roughness( roughness, ior );
    return texture_bicubic( t_transmission_framebuffer, s_transmission_framebuffer, frag_coord.xy, lod );
}

fn volume_attenuation( transmission_distance: f32, attenuation_color: vec3f, attenuation_distance: f32 ) -> vec3f {
    if (attenuation_distance == 0.0) {
        // Attenuation distance is +∞, i.e. the transmitted color is not attenuated at all.
        return vec3f( 1.0 );
    } else {
        // Compute light attenuation using Beer's law.
        let attenuation_coefficient = -log( attenuation_color ) / attenuation_distance;
        let transmittance = exp( - attenuation_coefficient * transmission_distance ); // Beer's law
        return transmittance;
    }
}

fn getIBLVolumeRefraction( n: vec3f, v: vec3f, roughness: f32, diffuse_color: vec3f,
    specular_color: vec3f, specular_f90: f32, position: vec3f, model_matrix: mat4x4f,
    view_matrix: mat4x4f, proj_matrix: mat4x4f, dispersion: f32, ior: f32, thickness: f32,
    attenuation_color: vec3f, attenuation_distance: f32 ) -> vec4f {

    var transmitted_light: vec4f;
    var transmittance: vec3f;

    $$ if USE_DISPERSION is defined

        let half_spread = ( ior - 1.0 ) * 0.025 * dispersion;
        let iors = vec3f( ior - half_spread, ior, ior + half_spread );

        for i in 0..3 {
            let transmission_ray = get_volume_transmission_ray( n, v, thickness, iors[i], model_matrix );
            let refracted_ray_exit = position + transmission_ray;

            // Project refracted vector on the framebuffer, while mapping to normalized device coordinates.
            let ndc_pos = proj_matrix * view_matrix * vec4f( refracted_ray_exit, 1.0 );
            var refraction_coords = ndc_pos.xy / ndc_pos.w;
            refraction_coords += 1.0;
            refraction_coords /= 2.0;

            refraction_coords = vec2f( refraction_coords.x, 1.0 - refraction_coords.y ); // webgpu

            // Sample framebuffer to get pixel the refracted ray hits.
            let transmission_sample = get_transmission_sample( refraction_coords, roughness, iors[i] );
            transmitted_light[i] = transmission_sample[i];
            transmitted_light.a += transmission_sample.a;

            transmittance[i] = diffuse_color[i] * volume_attenuation( length( transmission_ray ), attenuation_color, attenuation_distance )[i];
        }

        transmitted_light.a /= 3.0;

    $$ else

        let transmission_ray = get_volume_transmission_ray( n, v, thickness, ior, model_matrix );
        let refracted_ray_exit = position + transmission_ray;

        // Project refracted vector on the framebuffer, while mapping to normalized device coordinates.
        let ndc_pos = proj_matrix * view_matrix * vec4f( refracted_ray_exit, 1.0 );
        var refraction_coords = ndc_pos.xy / ndc_pos.w;
        refraction_coords += 1.0;
        refraction_coords /= 2.0;

        refraction_coords = vec2f( refraction_coords.x, 1.0 - refraction_coords.y ); // webgpu

        // Sample framebuffer to get pixel the refracted ray hits.
        transmitted_light = get_transmission_sample( refraction_coords, roughness, ior );
        transmittance = diffuse_color * volume_attenuation( length( transmission_ray ), attenuation_color, attenuation_distance );

    $$ endif

    let attenuated_color = transmittance * transmitted_light.rgb;

    // Get the specular component.
    let F = EnvironmentBRDF( n, v, specular_color, specular_f90, roughness );

    // As less light is transmitted, the opacity should be increased. This simple approximation does a decent job
    // of modulating a CSS background, and has no effect when the buffer is opaque, due to a solid object or clear color.
    let transmittance_factor = ( transmittance.r + transmittance.g + transmittance.b ) / 3.0;

    return vec4f( ( 1.0 - F ) * attenuated_color, 1.0 - ( 1.0 - transmitted_light.a ) * transmittance_factor );
}

$$ endif