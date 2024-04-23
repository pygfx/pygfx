// Provides lighting_phong()

{# Includes #}
{$ include 'pygfx.light_common.wgsl' $}


fn G_BlinnPhong_Implicit() -> f32 {
    return 0.25;
}

fn D_BlinnPhong(shininess: f32, dot_nh: f32) -> f32 {
    return RECIPROCAL_PI * ( shininess * 0.5 + 1.0 ) * pow( dot_nh, shininess );
}

fn BRDF_BlinnPhong(
    light_dir: vec3<f32>,
    view_dir: vec3<f32>,
    normal: vec3<f32>,
    specular_color: vec3<f32>,
    shininess: f32,
) -> vec3<f32> {
    let half_dir = normalize(light_dir + view_dir);
    let dot_nh = saturate(dot(normal, half_dir));
    let dot_vh = saturate(dot(view_dir, half_dir));
    let F = F_Schlick(specular_color, 1.0, dot_vh);
    let G = G_BlinnPhong_Implicit();
    let D = D_BlinnPhong(shininess, dot_nh);
    return F * ( G * D );
}

struct BlinnPhongMaterial {
    diffuse_color: vec3<f32>,
    specular_shininess: f32,
    specular_color: vec3<f32>,
    specular_strength: f32,
};

fn RE_Direct_BlinnPhong(
    direct_light: IncidentLight,
    geometry: GeometricContext,
    material: BlinnPhongMaterial,
    reflected_light: ReflectedLight,
) -> ReflectedLight {
    let dot_nl = saturate(dot(geometry.normal, direct_light.direction));
    let irradiance = dot_nl * direct_light.color;
    let direct_diffuse = irradiance * BRDF_Lambert( material.diffuse_color );
    let direct_specular = irradiance * BRDF_BlinnPhong( direct_light.direction, geometry.view_dir, geometry.normal, material.specular_color, material.specular_shininess ) * material.specular_strength;
    var out_reflected_light: ReflectedLight;
    out_reflected_light.direct_diffuse = reflected_light.direct_diffuse + direct_diffuse;
    out_reflected_light.direct_specular = reflected_light.direct_specular + direct_specular;
    out_reflected_light.indirect_diffuse = reflected_light.indirect_diffuse;
    out_reflected_light.indirect_specular = reflected_light.indirect_specular;
    return out_reflected_light;
}

fn RE_IndirectDiffuse_BlinnPhong(
    irradiance: vec3<f32>,
    geometry: GeometricContext,
    material: BlinnPhongMaterial,
    reflected_light: ReflectedLight,
) -> ReflectedLight {
    let indirect_diffuse = irradiance * BRDF_Lambert( material.diffuse_color );
    var out_reflected_light: ReflectedLight;
    out_reflected_light.direct_diffuse = reflected_light.direct_diffuse;
    out_reflected_light.direct_specular = reflected_light.direct_specular;
    out_reflected_light.indirect_diffuse = reflected_light.indirect_diffuse + indirect_diffuse;
    out_reflected_light.indirect_specular = reflected_light.indirect_specular;
    return out_reflected_light;
}

fn lighting_phong(
    varyings: Varyings,
    normal: vec3<f32>,
    view_dir: vec3<f32>,
    albeido: vec3<f32>,
) -> vec3<f32> {

    // Colors incoming via uniforms
    let specular_color = srgb2physical(u_material.specular_color.rgb);
    let ambient_color = u_ambient_light.color.rgb; // the one exception that is already physical

    var material: BlinnPhongMaterial;
    material.diffuse_color = albeido;
    material.specular_color = specular_color;
    material.specular_shininess = u_material.shininess;
    material.specular_strength = 1.0;   //  We could provide a specular map
    var reflected_light: ReflectedLight = ReflectedLight(vec3<f32>(0.0), vec3<f32>(0.0), vec3<f32>(0.0), vec3<f32>(0.0));

    var geometry: GeometricContext;
    geometry.position = varyings.world_pos;
    geometry.normal = normal;
    geometry.view_dir = view_dir;
    var i = 0;
    $$ if num_point_lights > 0
        loop {
            if (i >= {{ num_point_lights }}) { break; }
            let point_light = u_point_lights[i];
            var light = getPointLightInfo(point_light, geometry);
            if (! light.visible) { continue; }
            $$ if receive_shadow
            if (point_light.cast_shadow != 0){
                let shadow = get_cube_shadow(u_shadow_map_point_light, u_shadow_sampler, i, point_light.light_view_proj_matrix, geometry.position, light.direction, point_light.shadow_bias);
                light.color *= shadow;
            }
            $$ endif
            reflected_light = RE_Direct_BlinnPhong( light, geometry, material, reflected_light );
            continuing {
                i += 1;
            }
        }
    $$ endif
    $$ if num_spot_lights > 0
        i = 0;
        loop {
            if (i >= {{ num_spot_lights }}) { break; }
            let spot_light = u_spot_lights[i];
            var light = getSpotLightInfo(spot_light, geometry);
            if (! light.visible) { continue; }
            $$ if receive_shadow
            if (spot_light.cast_shadow != 0){
                let coords = spot_light.light_view_proj_matrix * vec4<f32>(geometry.position,1.0);
                let bias = spot_light.shadow_bias;
                let shadow = get_shadow(u_shadow_map_spot_light, u_shadow_sampler, i, coords, bias);
                light.color *= shadow;
            }
            $$ endif
            reflected_light = RE_Direct_BlinnPhong( light, geometry, material, reflected_light );
            continuing {
                i += 1;
            }
        }
    $$ endif
    $$ if num_dir_lights > 0
        i = 0;
        loop {
            if (i >= {{ num_dir_lights }}) { break; }
            let dir_light = u_directional_lights[i];
            var light = getDirectionalLightInfo(dir_light, geometry);
            if (! light.visible) { continue; }
            $$ if receive_shadow
            if (dir_light.cast_shadow != 0) {
                let coords = dir_light.light_view_proj_matrix * vec4<f32>(geometry.position,1.0);
                let bias = dir_light.shadow_bias;
                let shadow = get_shadow(u_shadow_map_dir_light, u_shadow_sampler, i, coords, bias);
                light.color *= shadow;
            }
            $$ endif
            reflected_light = RE_Direct_BlinnPhong( light, geometry, material, reflected_light );
            continuing {
                i += 1;
            }
        }
    $$ endif
    var irradiance = getAmbientLightIrradiance( ambient_color );

    // Light map (pre-baked lighting)
    $$ if use_light_map is defined
    let light_map_color = srgb2physical( textureSample( t_light_map, s_light_map, varyings.texcoord1 ).rgb );
    irradiance += light_map_color * u_material.light_map_intensity;
    $$ endif

    reflected_light = RE_IndirectDiffuse_BlinnPhong( irradiance, geometry, material, reflected_light );

    // Ambient occlusion
    $$ if use_ao_map is defined
    let ao_map_intensity = u_material.ao_map_intensity;
    let ambientOcclusion = ( textureSample( t_ao_map, s_ao_map, varyings.texcoord1 ).r - 1.0 ) * ao_map_intensity + 1.0;
    reflected_light.indirect_diffuse *= ambientOcclusion;
    $$ endif

    return reflected_light.direct_diffuse + reflected_light.direct_specular + reflected_light.indirect_diffuse + reflected_light.indirect_specular + u_material.emissive_color.rgb;
}