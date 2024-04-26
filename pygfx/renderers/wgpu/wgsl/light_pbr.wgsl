// Provides lighting_pbr()


{$ include 'pygfx.light_common.wgsl' $}


fn V_GGX_SmithCorrelated(alpha: f32, dot_nl: f32, dot_nv: f32) -> f32 {
    let a2 = pow(alpha, 2.0);
    let gv = dot_nl * sqrt(a2 + (1.0-a2) * pow(dot_nv, 2.0));
    let gl = dot_nv * sqrt(a2 + (1.0-a2) * pow(dot_nl, 2.0 ));
    return 0.5/ max( gv+gl, EPSILON);
}

fn D_GGX(alpha: f32, dot_nh: f32) -> f32 {
    let a2 = pow( alpha, 2.0 );
    let denom = pow(dot_nh, 2.0) * (a2 - 1.0) + 1.0;
    return RECIPROCAL_PI * a2/pow(denom, 2.0);
}

fn BRDF_GGX(light_dir: vec3<f32>, view_dir: vec3<f32>, normal: vec3<f32>, f0: vec3<f32>, f90: f32, roughness: f32) -> vec3<f32> {
    let alpha = pow( roughness, 2.0 );
    let half_dir = normalize( light_dir + view_dir );
    let dot_nl = saturate( dot( normal, light_dir ) );
    let dot_nv = saturate( dot( normal, view_dir ) );
    let dot_nh = saturate( dot( normal, half_dir ) );
    let dot_vh = saturate( dot( view_dir, half_dir ) );
    let F = F_Schlick( f0, f90, dot_vh);
    let V = V_GGX_SmithCorrelated( alpha, dot_nl, dot_nv );
    let D = D_GGX( alpha, dot_nh );
    return F * ( V * D );
}

fn DFGApprox( normal: vec3<f32>, view_dir: vec3<f32>, roughness: f32 ) -> vec2<f32>{
    let dot_nv = saturate( dot( normal, view_dir ) );
    let c0 = vec4<f32>(- 1.0, - 0.0275, - 0.572, 0.022);
    let c1 = vec4<f32>(1.0, 0.0425, 1.04, - 0.04);
    let r = roughness * c0 + c1;
    let a004 = min( r.x * r.x, exp2( - 9.28 * dot_nv ) ) * r.x + r.y;
    let fab: vec2<f32> = vec2<f32>( - 1.04, 1.04 ) * a004 + r.zw;
    return fab;
}


struct PhysicalMaterial {
    diffuse_color: vec3<f32>,
    roughness: f32,
    specular_color: vec3<f32>,
    specular_f90: f32,
};

struct LightScatter {
    single_scatter: vec3<f32>,
    multi_scatter: vec3<f32>,
};

fn perturbNormal2Arb( eye_pos: vec3<f32>, surf_norm: vec3<f32>, mapN: vec3<f32>, uv: vec2<f32>, is_front: bool) -> vec3<f32> {
    let q0 = dpdx( eye_pos.xyz );
    let q1 = dpdy( eye_pos.xyz );
    let st0 = dpdx( uv.xy );
    let st1 = dpdy( uv.xy );
    let N = surf_norm; //  normalized
    let q1perp = cross( q1, N );
    let q0perp = cross( N, q0 );
    let T = q1perp * st0.x + q0perp * st1.x;
    let B = q1perp * st0.y + q0perp * st1.y;
    let det = max( dot( T, T ), dot( B, B ) );
    let faceDirection = f32(is_front) * 2.0 - 1.0;
    let scale = faceDirection * inverseSqrt(det);
    return normalize(T * mapN.x * scale + B * mapN.y * scale + N * mapN.z);
}

fn getMipLevel(maxMIPLevelScalar: f32, level: f32) -> f32 {
    let sigma = (3.141592653589793 * level * level) / (1.0 + level);
    let desiredMIPLevel = maxMIPLevelScalar + log2(sigma);
    let mip_level = clamp(desiredMIPLevel, 0.0, maxMIPLevelScalar);
    return mip_level;
}

fn getIBLIrradiance( normal: vec3<f32>, env_map: texture_cube<f32>, env_map_sampler: sampler, mip_level: f32) -> vec3<f32> {
    let envMapColor_srgb = textureSampleLevel( env_map, env_map_sampler, vec3<f32>( -normal.x, normal.yz), mip_level );
    return srgb2physical(envMapColor_srgb.rgb) * u_material.env_map_intensity * PI;
}

fn getIBLRadiance( reflectVec: vec3<f32>, env_map: texture_cube<f32>, env_map_sampler: sampler, mip_level: f32 ) -> vec3<f32> {
    let envMapColor_srgb = textureSampleLevel( env_map, env_map_sampler, vec3<f32>( -reflectVec.x, reflectVec.yz), mip_level );
    return srgb2physical(envMapColor_srgb.rgb) * u_material.env_map_intensity;
}

fn computeMultiscattering(normal: vec3<f32>, view_dir: vec3<f32>, specular_color: vec3<f32>, specular_f90: f32, roughness: f32) -> LightScatter {
    let fab = DFGApprox( normal, view_dir, roughness );
    let FssEss = specular_color * fab.x + specular_f90 * fab.y;
    let Ess: f32 = fab.x + fab.y;
    let Ems: f32 = 1.0 - Ess;
    let Favg = specular_color + ( 1.0 - specular_color ) * 0.047619; // 1/21
    let Fms = FssEss * Favg / ( 1.0 - Ems * Favg );
    var scatter: LightScatter;
    scatter.single_scatter = FssEss;
    scatter.multi_scatter = Fms * Ems;
    return scatter;
}

fn RE_IndirectSpecular_Physical(radiance: vec3<f32>, irradiance: vec3<f32>,
        geometry: GeometricContext, material: PhysicalMaterial, reflected_light: ReflectedLight) -> ReflectedLight{
    let cosineWeightedIrradiance: vec3<f32> = irradiance * RECIPROCAL_PI;
    let scatter = computeMultiscattering( geometry.normal, geometry.view_dir, material.specular_color, material.specular_f90, material.roughness);
    //let total_scattering = scatter.single_scatter + scatter.multi_scatter;
    //let diffuse = material.diffuse_color * ( 1.0 - max( max( total_scattering.r, total_scattering.g ), total_scattering.b ) );
    let diffuse = material.diffuse_color * ( 1.0 - scatter.single_scatter - scatter.multi_scatter);
    var out_reflected_light: ReflectedLight = reflected_light;
    out_reflected_light.indirect_specular += (radiance * scatter.single_scatter + scatter.multi_scatter * cosineWeightedIrradiance);
    out_reflected_light.indirect_diffuse += diffuse * cosineWeightedIrradiance;
    return out_reflected_light;
}

fn RE_IndirectDiffuse_Physical(irradiance: vec3<f32>, geometry: GeometricContext, material: PhysicalMaterial, reflected_light: ReflectedLight) -> ReflectedLight {
    var out_reflected_light: ReflectedLight = reflected_light;
    out_reflected_light.indirect_diffuse += irradiance * BRDF_Lambert( material.diffuse_color );
    return out_reflected_light;
}

fn RE_Direct_Physical(direct_light: IncidentLight, geometry: GeometricContext, material: PhysicalMaterial, reflected_light: ReflectedLight) -> ReflectedLight {
    let dot_nl = saturate( dot( geometry.normal, direct_light.direction ));
    let irradiance = dot_nl * direct_light.color;
    var out_reflected_light: ReflectedLight = reflected_light;
    out_reflected_light.direct_specular += irradiance * BRDF_GGX( direct_light.direction, geometry.view_dir, geometry.normal, material.specular_color, material.specular_f90, material.roughness );
    out_reflected_light.indirect_diffuse += irradiance * BRDF_Lambert( material.diffuse_color );
    return out_reflected_light;
}

fn RE_AmbientOcclusion_Physical(ambientOcclusion: f32, geometry: GeometricContext, material: PhysicalMaterial, reflected_light: ReflectedLight) -> ReflectedLight {
    let dot_nv = saturate( dot( geometry.normal, geometry.view_dir ) );
    let ao_nv = dot_nv + ambientOcclusion;
    let ao_exp = exp2( -16.0 * material.roughness - 1.0 );
    let ao = saturate( pow(ao_nv, ao_exp) - 1.0 + ambientOcclusion, );
    var out_reflected_light: ReflectedLight = reflected_light;
    out_reflected_light.indirect_diffuse *= ambientOcclusion;
    out_reflected_light.indirect_specular *= ao;
    return out_reflected_light;
}


fn lighting_pbr(
    varyings: Varyings,
    normal: vec3<f32>,
    view_dir: vec3<f32>,
    albeido: vec3<f32>,
) -> vec3<f32> {

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
    material.diffuse_color = albeido * ( 1.0 - metalness_factor );
    material.specular_color = mix( vec3<f32>( 0.04 ), albeido.rgb, metalness_factor );
    material.roughness = min( roughness_factor + geometry_roughness, 1.0 );
    material.specular_f90 = 1.0;

    // Define geometry
    var geometry: GeometricContext;
    geometry.position = varyings.world_pos;
    geometry.normal = normal;
    geometry.view_dir = view_dir;

    // Init the reflected light. Defines diffuse and specular, both direct and indirect
    var reflected_light: ReflectedLight = ReflectedLight(vec3<f32>(0.0), vec3<f32>(0.0), vec3<f32>(0.0), vec3<f32>(0.0));

    // Direct light from light sources
    var i = 0;
    $$ if num_point_lights > 0
        i = 0;
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
            reflected_light = RE_Direct_Physical( light, geometry, material, reflected_light );
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
                let shadow = get_shadow(u_shadow_map_spot_light, u_shadow_sampler, i, coords, spot_light.shadow_bias);
                light.color *= shadow;
            }
            $$ endif
            reflected_light = RE_Direct_Physical( light, geometry, material, reflected_light );
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
                let shadow = get_shadow(u_shadow_map_dir_light, u_shadow_sampler, i, coords, dir_light.shadow_bias);
                light.color *= shadow;
            }
            $$ endif
            reflected_light = RE_Direct_Physical( light, geometry, material, reflected_light );
            continuing {
                i += 1;
            }
        }
    $$ endif

    // The rest is for indirect light

    let ambient_color = u_ambient_light.color.rgb;  // the one exception that is already physical
    var irradiance = getAmbientLightIrradiance( ambient_color );

    // Light map (pre-baked lighting)
    $$ if use_light_map is defined
    let light_map_color = srgb2physical( textureSample( t_light_map, s_light_map, varyings.texcoord1 ).rgb );
    irradiance += light_map_color * u_material.light_map_intensity;
    // Note that if we implement light map for MeshBasicMaterial, we must multiply the intensity with the reciprocal PI.
    $$ endif

    // Process irradiance
    reflected_light = RE_IndirectDiffuse_Physical( irradiance, geometry, material, reflected_light );

    // IBL (srgb2physical and intensity is handled in the getter functions)
    $$ if use_IBL is defined
    $$ if env_mapping_mode == "CUBE-REFLECTION"
        var reflectVec = reflect( -view_dir, normal );
        let mip_level_r = getMipLevel(u_material.env_map_max_mip_level, material.roughness);
    $$ elif env_mapping_mode == "CUBE-REFRACTION"
        var reflectVec = refract( -view_dir, normal, u_material.refraction_ratio );
        let mip_level_r = 1.0;
    $$ endif
    reflectVec = normalize(mix(reflectVec, normal, material.roughness*material.roughness));
    let ibl_radiance = getIBLRadiance( reflectVec, t_env_map, s_env_map, mip_level_r );
    let mip_level_i = getMipLevel(u_material.env_map_max_mip_level, 1.0);
    let ibl_irradiance = getIBLIrradiance( geometry.normal, t_env_map, s_env_map, mip_level_i );
    reflected_light = RE_IndirectSpecular_Physical(ibl_radiance, ibl_irradiance, geometry, material, reflected_light);
    $$ endif

    // Ambient occlusion
    $$ if use_ao_map is defined
    let ao_map_intensity = u_material.ao_map_intensity;
    let ambientOcclusion = ( textureSample( t_ao_map, s_ao_map, varyings.texcoord1 ).r - 1.0 ) * ao_map_intensity + 1.0;
    reflected_light = RE_AmbientOcclusion_Physical(ambientOcclusion, geometry, material, reflected_light);
    $$ endif

    // Combine direct and indirect light
    var lit_color = reflected_light.direct_diffuse + reflected_light.direct_specular + reflected_light.indirect_diffuse + reflected_light.indirect_specular;

    // Add emissive color
    var emissive_color = srgb2physical(u_material.emissive_color.rgb);
    $$ if use_emissive_map is defined
    emissive_color *= srgb2physical(textureSample(t_emissive_map, s_emissive_map, varyings.texcoord).rgb);
    $$ endif
    lit_color += emissive_color * u_material.emissive_intensity;

    return lit_color;
}