// Metalness
var metalness_factor: f32 = u_material.metalness;
$$ if use_metalness_map is defined
    metalness_factor *= textureSample( t_metalness_map, s_metalness_map, metalness_map_uv ).b;
$$ endif

// Roughness
var roughness_factor: f32 = u_material.roughness;
$$ if use_roughness_map is defined
    roughness_factor *= textureSample( t_roughness_map, s_roughness_map, roughness_map_uv ).g;
$$ endif


// Define material
var material: PhysicalMaterial;

material.diffuse_color = physical_albeido * ( 1.0 - metalness_factor );

let dxy = max( abs( dpdx( surface_normal ) ), abs( dpdy( surface_normal ) ) );
let geometry_roughness = max( max( dxy.x, dxy.y ), dxy.z );

material.roughness = max( roughness_factor, 0.0525 );
material.roughness += geometry_roughness;
material.roughness = min( material.roughness, 1.0 );


$$ if USE_IOR is defined
    material.ior = u_material.ior;

    $$ if USE_SPECULAR
        var specular_intensity = u_material.specular_intensity;
        var specular_color = srgb2physical(u_material.specular_color.rgb);
        
        $$ if use_specular_map is defined
            specular_color *= srgb2physical(textureSample( t_specular_map, s_specular_map, specular_map_uv ).rgb);
        $$ endif

        $$ if use_specular_intensity_map is defined
            specular_intensity *= textureSample( t_specular_intensity_map, s_specular_intensity_map, specular_intensity_map_uv ).a;
        $$ endif

        material.specular_f90 = mix( specular_intensity, 1.0, metalness_factor );
    
    $$ else
        let specular_intensity = 1.0;
        let specular_color = vec3f( 1.0 );
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
        material.clearcoat *= textureSample( t_clearcoat_map, s_clearcoat_map, clearcoat_map_uv ).r;
    $$ endif

    $$ if use_clearcoat_roughness_map is defined
        material.clearcoat_roughness *= textureSample( t_clearcoat_roughness_map, s_clearcoat_roughness_map, clearcoat_roughness_map_uv ).g;
    $$ endif

    material.clearcoat = saturate( material.clearcoat );
    material.clearcoat_roughness = max( material.clearcoat_roughness, 0.0525 );
    material.clearcoat_roughness += geometry_roughness;
    material.clearcoat_roughness = min( material.clearcoat_roughness, 1.0 );

$$ endif


$$ if USE_IRIDESCENCE is defined
    material.iridescence = u_material.iridescence;
    material.iridescence_ior = u_material.iridescence_ior;

    $$ if use_iridescence_map is defined
        material.iridescence *= textureSample(t_iridescence_map, s_iridescence_map, iridescence_map_uv).r;
    $$ endif

    let iridescence_thickness_minimum = u_material.iridescence_thickness_range[0];
    let iridescence_thickness_maximum = u_material.iridescence_thickness_range[1];
    $$ if use_iridescence_thickness_map is defined
        material.iridescence_thickness = (iridescence_thickness_maximum - iridescence_thickness_minimum) * textureSample(t_iridescence_thickness_map, s_iridescence_thickness_map, iridescence_thickness_map_uv).g + iridescence_thickness_minimum;
    $$ else
        material.iridescence_thickness = iridescence_thickness_maximum;
    $$ endif

    if (material.iridescence_thickness == 0.0) {
        material.iridescence = 0.0;
    }else{
        material.iridescence = saturate( material.iridescence );
    }

    let dot_nvi = saturate( dot( normal, view ) );
    if material.iridescence > 0.0 {
        material.iridescence_fresnel = evalIridescence( 1.0, material.iridescence_ior, dot_nvi, material.iridescence_thickness, material.specular_color );
        material.iridescence_f0 = Schlick_to_F0( material.iridescence_fresnel, 1.0, dot_nvi );
    }
$$ endif


$$ if USE_SHEEN is defined

    material.sheen_color = srgb2physical(u_material.sheen_color.rgb) * u_material.sheen;

    $$ if use_sheen_color_map is defined
        material.sheen_color *= textureSample( t_sheen_color_map, s_sheen_color_map, varyings.texcoord{{sheen_color_map_uv or ''}} ).rgb;
    $$ endif

    material.sheen_roughness = clamp( u_material.sheen_roughness, 0.07, 1.0 );
    $$ if use_sheen_roughness_map is defined
        material.sheen_roughness *= textureSample( t_sheen_roughness_map, s_sheen_roughness_map, varyings.texcoord{{sheen_roughness_map_uv or ''}} ).a;
    $$ endif

$$ endif

$$ if USE_ANISOTROPY is defined
    let anisotropy_vector = u_material.anisotropy_vector;
    $$ if use_anisotropy_map is defined
        let anisotropy_polar = textureSample( t_anisotropy_map, s_anisotropy_map, anisotropy_map_uv ).rgb;
        let anisotropy_mat = mat2x2f( anisotropy_vector.x, anisotropy_vector.y, -anisotropy_vector.y, anisotropy_vector.x );
        var anisotropy_v = anisotropy_mat * normalize( 2.0 * anisotropy_polar.rg - vec2f( 1.0 ) ) * anisotropy_polar.b;
    $$ else
        var anisotropy_v = anisotropy_vector;
    $$ endif

    material.anisotropy = length( anisotropy_v );

    if(material.anisotropy == 0.0) {
        anisotropy_v = vec2f(1.0, 0.0);
    }else{
        anisotropy_v /= material.anisotropy;
        material.anisotropy = saturate( material.anisotropy );
    }

    // Roughness along the anisotropy bitangent is the material roughness, while the tangent roughness increases with anisotropy.
    material.alpha_t = mix( pow2( material.roughness ), 1.0, pow2( material.anisotropy ) );
    material.anisotropy_t = tbn[0] * anisotropy_v.x + tbn[1] * anisotropy_v.y;
    material.anisotropy_b = tbn[1] * anisotropy_v.x - tbn[0] * anisotropy_v.y;
$$ endif