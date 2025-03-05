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

$$ if USE_CLEARCOAT is defined
fn BRDF_GGX_CC(light_dir: vec3<f32>, view_dir: vec3<f32>, normal: vec3<f32>, f0: vec3<f32>, f90: f32, roughness: f32) -> vec3<f32> {
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
$$ endif

fn BRDF_GGX(light_dir: vec3<f32>, view_dir: vec3<f32>, normal: vec3<f32>, f0: vec3<f32>, f90: f32, roughness: f32, material: PhysicalMaterial) -> vec3<f32> {
    let alpha = pow( roughness, 2.0 );
    let half_dir = normalize( light_dir + view_dir );
    let dot_nl = saturate( dot( normal, light_dir ) );
    let dot_nv = saturate( dot( normal, view_dir ) );
    let dot_nh = saturate( dot( normal, half_dir ) );
    let dot_vh = saturate( dot( view_dir, half_dir ) );
    var F = F_Schlick( f0, f90, dot_vh);

    $$ if USE_IRIDESCENCE is defined
        F = mix( F, material.iridescence_fresnel, material.iridescence );
    $$ endif

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

fn EnvironmentBRDF(normal: vec3f, view_dir: vec3f, specular_color: vec3f, specular_f90: f32, roughness: f32) -> vec3f{
    let fab = DFGApprox( normal, view_dir, roughness );
    return specular_color * fab.x + specular_f90 * fab.y;
}


struct PhysicalMaterial {
    diffuse_color: vec3<f32>,
    roughness: f32,
    specular_color: vec3<f32>,
    specular_f90: f32,

    $$ if USE_IOR is defined
        ior: f32,
    $$ endif

    $$ if USE_CLEARCOAT is defined
        clearcoat: f32,
        clearcoat_roughness: f32,
        clearcoat_f0: vec3<f32>,
        clearcoat_f90: f32,
    $$ endif

    $$ if USE_IRIDESCENCE is defined
        iridescence: f32,
        iridescence_ior: f32,
        iridescence_thickness: f32,
        iridescence_fresnel: vec3<f32>,
        iridescence_f0: vec3<f32>,
    $$ endif

};

struct LightScatter {
    single_scatter: vec3<f32>,
    multi_scatter: vec3<f32>,
};

var<private> clearcoat_specular_direct: vec3f = vec3f(0.0);
var<private> clearcoat_specular_indirect: vec3f = vec3f(0.0);
var<private> sheen_specular_direct: vec3f = vec3f(0.0);
var<private> sheen_specular_indirect: vec3f = vec3f(0.0);

fn Schlick_to_F0( f: vec3<f32>, f90: f32, dot_vh: f32 ) -> vec3<f32> {
    let x = clamp( 1.0 - dot_vh, 0.0, 1.0 );
    let x2 = x * x;
    let x5 = clamp( x * x2 * x2, 0.0, 0.9999 );
    return ( f - vec3<f32>( f90 ) * x5 ) / ( 1.0 - x5 );
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

$$ if USE_IRIDESCENCE is defined
fn computeMultiscatteringIridescence(normal: vec3<f32>, view_dir: vec3<f32>, specular_color: vec3<f32>, specular_f90: f32, roughness: f32, iridescence_f0: vec3<f32>, iridescence: f32) -> LightScatter {
$$ else
fn computeMultiscattering(normal: vec3<f32>, view_dir: vec3<f32>, specular_color: vec3<f32>, specular_f90: f32, roughness: f32 ) -> LightScatter {
$$ endif

    let fab = DFGApprox( normal, view_dir, roughness );

    $$ if USE_IRIDESCENCE is defined
        let Fr = mix(specular_color, iridescence_f0, iridescence );
    $$ else
        let Fr = specular_color;
    $$ endif

    let FssEss = Fr * fab.x + specular_f90 * fab.y;
    let Ess: f32 = fab.x + fab.y;
    let Ems: f32 = 1.0 - Ess;
    let Favg = specular_color + ( 1.0 - specular_color ) * 0.047619; // 1/21
    let Fms = FssEss * Favg / ( 1.0 - Ems * Favg );
    var scatter: LightScatter;
    scatter.single_scatter = FssEss;
    scatter.multi_scatter = Fms * Ems;
    return scatter;
}

fn RE_IndirectSpecular(radiance: vec3<f32>, irradiance: vec3<f32>, clearcoat_radiance: vec3<f32>,
        geometry: GeometricContext, material: PhysicalMaterial, reflected_light: ptr<function, ReflectedLight>){
        
    $$ if USE_CLEARCOAT is defined
        clearcoat_specular_indirect += clearcoat_radiance * EnvironmentBRDF( geometry.clearcoat_normal, geometry.view_dir, material.clearcoat_f0, material.clearcoat_f90, material.clearcoat_roughness );
    $$ endif
    let cosineWeightedIrradiance: vec3<f32> = irradiance * RECIPROCAL_PI;

    $$ if USE_IRIDESCENCE is defined
        let scatter = computeMultiscatteringIridescence( geometry.normal, geometry.view_dir, material.specular_color, material.specular_f90, material.roughness, material.iridescence_f0, material.iridescence );
    $$ else
        let scatter = computeMultiscattering( geometry.normal, geometry.view_dir, material.specular_color, material.specular_f90, material.roughness);
    $$ endif
    let total_scattering = scatter.single_scatter + scatter.multi_scatter;
    let diffuse = material.diffuse_color * ( 1.0 - max( max( total_scattering.r, total_scattering.g ), total_scattering.b ) );
    // let diffuse = material.diffuse_color * ( 1.0 - scatter.single_scatter - scatter.multi_scatter);
    (*reflected_light).indirect_specular += (radiance * scatter.single_scatter + scatter.multi_scatter * cosineWeightedIrradiance);
    (*reflected_light).indirect_diffuse += diffuse * cosineWeightedIrradiance;
}

fn RE_IndirectDiffuse(irradiance: vec3<f32>, geometry: GeometricContext, material: PhysicalMaterial, reflected_light: ptr<function, ReflectedLight>) {
    (*reflected_light).indirect_diffuse += irradiance * BRDF_Lambert( material.diffuse_color );
}

fn RE_Direct(
    direct_light: IncidentLight, 
    geometry: GeometricContext, 
    material: PhysicalMaterial, 
    reflected_light: ptr<function, ReflectedLight>
) {
    let dot_nl = saturate( dot( geometry.normal, direct_light.direction ));
    let irradiance = dot_nl * direct_light.color;

    $$ if USE_CLEARCOAT is defined
        let dot_nl_cc = saturate( dot( geometry.clearcoat_normal, direct_light.direction ));
        let clearcoat_irradiance = dot_nl_cc * direct_light.color;
        clearcoat_specular_direct += clearcoat_irradiance * BRDF_GGX_CC( direct_light.direction, geometry.view_dir, geometry.clearcoat_normal, material.specular_color, material.clearcoat_f90, material.clearcoat_roughness );
    $$ endif

    (*reflected_light).direct_specular += irradiance * BRDF_GGX( direct_light.direction, geometry.view_dir, geometry.normal, material.specular_color, material.specular_f90, material.roughness, material );
    (*reflected_light).direct_diffuse += irradiance * BRDF_Lambert( material.diffuse_color );
}

fn computeSpecularOcclusion(dot_nv: f32, ambient_occlusion: f32, roughness: f32) -> f32 {
    let ao_nv = dot_nv + ambient_occlusion;
    let ao_exp = exp2( -16.0 * roughness - 1.0 );
    return saturate( pow(ao_nv, ao_exp) - 1.0 + ambient_occlusion );
}
