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
        geometry: GeometricContext, material: PhysicalMaterial, reflected_light: ptr<function, ReflectedLight>){
    let cosineWeightedIrradiance: vec3<f32> = irradiance * RECIPROCAL_PI;
    let scatter = computeMultiscattering( geometry.normal, geometry.view_dir, material.specular_color, material.specular_f90, material.roughness);
    //let total_scattering = scatter.single_scatter + scatter.multi_scatter;
    //let diffuse = material.diffuse_color * ( 1.0 - max( max( total_scattering.r, total_scattering.g ), total_scattering.b ) );
    let diffuse = material.diffuse_color * ( 1.0 - scatter.single_scatter - scatter.multi_scatter);
    (*reflected_light).indirect_specular += (radiance * scatter.single_scatter + scatter.multi_scatter * cosineWeightedIrradiance);
    (*reflected_light).indirect_diffuse += diffuse * cosineWeightedIrradiance;
}

fn RE_IndirectDiffuse_Physical(irradiance: vec3<f32>, geometry: GeometricContext, material: PhysicalMaterial, reflected_light: ptr<function, ReflectedLight>) {
    (*reflected_light).indirect_diffuse += irradiance * BRDF_Lambert( material.diffuse_color );
}

fn RE_Direct_Physical(direct_light: IncidentLight, geometry: GeometricContext, material: PhysicalMaterial, reflected_light: ptr<function, ReflectedLight>) {
    let dot_nl = saturate( dot( geometry.normal, direct_light.direction ));
    let irradiance = dot_nl * direct_light.color;
    (*reflected_light).direct_specular += irradiance * BRDF_GGX( direct_light.direction, geometry.view_dir, geometry.normal, material.specular_color, material.specular_f90, material.roughness );
    (*reflected_light).indirect_diffuse += irradiance * BRDF_Lambert( material.diffuse_color );
}

fn RE_AmbientOcclusion_Physical(ambientOcclusion: f32, geometry: GeometricContext, material: PhysicalMaterial, reflected_light: ptr<function, ReflectedLight>) {
    let dot_nv = saturate( dot( geometry.normal, geometry.view_dir ) );
    let ao_nv = dot_nv + ambientOcclusion;
    let ao_exp = exp2( -16.0 * material.roughness - 1.0 );
    let ao = saturate( pow(ao_nv, ao_exp) - 1.0 + ambientOcclusion, );
    (*reflected_light).indirect_diffuse *= ambientOcclusion;
    (*reflected_light).indirect_specular *= ao;
}


fn lighting_pbr(
    reflected_light: ptr<function, ReflectedLight>,
    geometry: GeometricContext,
    material: PhysicalMaterial,
)  {
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
            RE_Direct_Physical( light, geometry, material, reflected_light );
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
            RE_Direct_Physical( light, geometry, material, reflected_light );
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
            RE_Direct_Physical( light, geometry, material, reflected_light );
            continuing {
                i += 1;
            }
        }
    $$ endif

}