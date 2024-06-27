// Metalness
var metalness_factor: f32 = u_material.metalness;
$$ if use_metalness_map is defined
    metalness_factor *= textureSample( t_metalness_map, s_metalness_map, varyings.texcoord ).b;
$$ endif

// Roughness
var roughness_factor: f32 = u_material.roughness;
$$ if use_roughness_map is defined
    roughness_factor *= textureSample( t_roughness_map, s_roughness_map, varyings.texcoord ).g;
$$ endif
roughness_factor = max( roughness_factor, 0.0525 );
let dxy = max( abs( dpdx( varyings.geometry_normal ) ), abs( dpdy( varyings.geometry_normal ) ) );
let geometry_roughness = max( max( dxy.x, dxy.y ), dxy.z );

// Define material
var material: PhysicalMaterial;
material.diffuse_color = physical_albeido * ( 1.0 - metalness_factor );
material.specular_color = mix( vec3<f32>( 0.04 ), physical_albeido.rgb, metalness_factor );
material.roughness = min( roughness_factor + geometry_roughness, 1.0 );
material.specular_f90 = 1.0;