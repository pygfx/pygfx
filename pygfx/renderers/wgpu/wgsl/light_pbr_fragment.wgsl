// Metalness
var metalness_factor: f32 = u_material.metalness;
$$ if use_metalness_map is defined
    metalness_factor *= textureSample( t_metalness_map, s_metalness_map, varyings.texcoord{{metalness_map_uv or ''}} ).b;
$$ endif

// Roughness
var roughness_factor: f32 = u_material.roughness;
$$ if use_roughness_map is defined
    roughness_factor *= textureSample( t_roughness_map, s_roughness_map, varyings.texcoord{{roughness_map_uv or ''}} ).g;
$$ endif


// Define material
var material: PhysicalMaterial;

material.diffuse_color = physical_albeido * ( 1.0 - metalness_factor );

let dxy = max( abs( dpdx( surface_normal ) ), abs( dpdy( surface_normal ) ) );
let geometry_roughness = max( max( dxy.x, dxy.y ), dxy.z );

material.roughness = max( roughness_factor, 0.0525 );
material.roughness += geometry_roughness;
material.roughness = min( material.roughness, 1.0 );


$$ if IOR is defined
    material.ior = u_material.ior;

    $$ if USE_SPECULAR
        var specular_intensity = u_material.specular_intensity;
        var specular_color = srgb2physical(u_material.specular_color.rgb);
        
        $$ if use_specular_map is defined
            specular_color *= srgb2physical(textureSample( t_specular_map, s_specular_map, varyings.texcoord{{specular_map_uv or ''}} ).rgb);
        $$ endif

        $$ if use_specular_intensity_map is defined
            specular_intensity *= textureSample( t_specular_intensity_map, s_specular_intensity_map, varyings.texcoord{{specular_intensity_map_uv or ''}} ).a;
        $$ endif

        material.specular_f90 = mix( specular_intensity, 1.0, metalness_factor );
    
    $$ else
        specular_intensity = 1.0;
        specular_color = vec3f( 1.0 );
        material.specular_f90 = 1.0;

    $$ endif

    material.specular_color = mix( min( pow2( ( material.ior - 1.0 ) / ( material.ior + 1.0 ) ) * specular_color, vec3f( 1.0 ) ) * specular_intensity, physical_albeido, metalness_factor );

$$ else

    material.specular_color = mix( vec3<f32>( 0.04 ), physical_albeido.rgb, metalness_factor );
    material.specular_f90 = 1.0;

$$ endif


$$ if USE_CLEARCOAT is defined

    material.clearcoat = u_material.clearcoat;
    material.clearcoat_roughness = u_material.clearcoat_roughness;
    material.clearcoat_f0 = vec3f( 0.04 );
    material.clearcoat_f90 = 1.0;

    $$ if use_clearcoat_map is defined
        material.clearcoat *= textureSample( t_clearcoat_map, s_clearcoat_map, varyings.texcoord{{clearcoat_map_uv or ''}} ).r;
    $$ endif

    $$ if use_clearcoat_roughness_map is defined
        material.clearcoat_roughness *= textureSample( t_clearcoat_roughness_map, s_clearcoat_roughness_map, varyings.texcoord{{clearcoat_roughness_map_uv or ''}} ).r;
    $$ endif

    material.clearcoat = saturate( material.clearcoat );
    material.clearcoat_roughness = max( material.clearcoat_roughness, 0.0525 );
    material.clearcoat_roughness += geometry_roughness;
    material.clearcoat_roughness = min( material.clearcoat_roughness, 1.0 );

$$ endif