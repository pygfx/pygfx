// Provides lighting_pbr()

{$ include 'pygfx.light_common.wgsl' $}

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

    $$ if USE_SHEEN is defined
        sheen_color: vec3<f32>,
        sheen_roughness: f32,
    $$ endif

    $$ if USE_ANISOTROPY is defined
        anisotropy: f32,
        alpha_t: f32,
        anisotropy_t: vec3<f32>,
        anisotropy_b: vec3<f32>,
    $$ endif
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

$$ if USE_ANISOTROPY is defined
fn V_GGX_SmithCorrelated_Anisotropic(alpha_t: f32, alpha_b: f32, dot_tv: f32, dot_bv: f32, dot_tl: f32, dot_bl: f32, dot_nv: f32, dot_nl: f32) -> f32 {
    let gv = dot_nl * length( vec3f(alpha_t * dot_tv, alpha_b * dot_bv, dot_nv) );
    let gl = dot_nv * length( vec3f(alpha_t * dot_tl, alpha_b * dot_bl, dot_nl) );
    let v = 0.5 / ( gv + gl );

    return saturate( v );
}

fn D_GGX_Anisotropic(alpha_t: f32, alpha_b: f32, dot_nh: f32, dot_th: f32, dot_bh: f32) -> f32 {
    let a2 = alpha_t * alpha_b;
    let v = vec3f( alpha_b * dot_th, alpha_t * dot_bh, a2 * dot_nh );
    let v2 = dot(v, v);
    let w2 = a2 / v2;

    return RECIPROCAL_PI * a2 * pow2(w2);
}
$$ endif

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

    $$ if USE_ANISOTROPY is defined
        let dot_tl = dot( material.anisotropy_t, light_dir );
        let dot_tv = dot( material.anisotropy_t, view_dir );
        let dot_th = dot( material.anisotropy_t, half_dir );
        let dot_bl = dot( material.anisotropy_b, light_dir );
        let dot_bv = dot( material.anisotropy_b, view_dir );
        let dot_bh = dot( material.anisotropy_b, half_dir );

        let V = V_GGX_SmithCorrelated_Anisotropic( material.alpha_t, alpha, dot_tv, dot_bv, dot_tl, dot_bl, dot_nv, dot_nl );
        let D = D_GGX_Anisotropic( material.alpha_t, alpha, dot_nh, dot_th, dot_bh );
    $$ else
        let V = V_GGX_SmithCorrelated( alpha, dot_nl, dot_nv );
        let D = D_GGX( alpha, dot_nh );
    $$ endif

    return F * ( V * D );
}

$$ if USE_SHEEN is defined

// https://github.com/google/filament/blob/main/shaders/src/surface_brdf.fs
fn D_Charlie( roughness: f32, dot_nh: f32 ) -> f32 {
    let alpha = pow2( roughness );

    // Estevez and Kulla 2017, "Production Friendly Microfacet Sheen BRDF"
    let inv_alpha = 1.0 / alpha;
    let cos2h = dot_nh * dot_nh;
    let sin2h = max( 1.0 - cos2h, 0.0078125 ); // 2^(-14/2), so sin2h^2 > 0 in fp16
    return ( 2.0 + inv_alpha ) * pow( sin2h, inv_alpha * 0.5 ) / ( 2.0 * PI );
}

// https://github.com/google/filament/blob/main/shaders/src/surface_brdf.fs
fn V_Neubelt( dot_nv: f32, dot_nl: f32 ) -> f32 {
    // Neubelt and Pettineo 2013, "Crafting a Next-gen Material Pipeline for The Order: 1886"
    return saturate( 1.0 / ( 4.0 * ( dot_nl + dot_nv - dot_nl * dot_nv ) ) );
}

fn BRDF_Sheen(light_dir: vec3<f32>, view_dir: vec3<f32>, normal: vec3<f32>, sheen_color: vec3<f32>, sheen_roughness: f32) -> vec3<f32> {
    let half_dir = normalize( light_dir + view_dir );

    let dot_nl = saturate( dot( normal, light_dir ) );
    let dot_nv = saturate( dot( normal, view_dir ) );
    let dot_nh = saturate( dot( normal, half_dir ) );

    let D = D_Charlie( sheen_roughness, dot_nh );
    let V = V_Neubelt( dot_nv, dot_nl );

    return sheen_color * ( D * V );
}

// This is a curve-fit approximation to the "Charlie sheen" BRDF integrated over the hemisphere from
// Estevez and Kulla 2017, "Production Friendly Microfacet Sheen BRDF". The analysis can be found
// in the Sheen section of https://drive.google.com/file/d/1T0D1VSyR4AllqIJTQAraEIzjlb5h4FKH/view?usp=sharing
fn IBLSheenBRDF(normal: vec3<f32>, view_dir: vec3<f32>, roughness: f32) -> f32 {
    let dot_nv = saturate( dot( normal, view_dir ) );
    let r2 = roughness * roughness;
    let a = select( (-8.48 * r2 + 14.3 * roughness - 9.95), (-339.2 * r2 + 161.4 * roughness - 25.9), roughness < 0.25 );
    let b = select( (1.97 * r2 - 3.27 * roughness + 0.72), (44.0 * r2 - 23.7 * roughness + 3.26), roughness < 0.25 );
    let DG = exp( a * dot_nv + b ) + select( 0.1 * (roughness - 0.25), 0.0, roughness < 0.25 );
    return saturate( DG * RECIPROCAL_PI );
}

$$ endif

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

$$ if USE_IBL is defined

fn getMipLevel(maxMIPLevelScalar: f32, level: f32) -> f32 {
    let sigma = (3.141592653589793 * level * level) / (1.0 + level);
    let desiredMIPLevel = maxMIPLevelScalar + log2(sigma);
    let mip_level = clamp(desiredMIPLevel, 0.0, maxMIPLevelScalar);
    return mip_level;
}

fn getIBLIrradiance( normal: vec3<f32> ) -> vec3<f32> {
    let mip_level = getMipLevel(u_material.env_map_max_mip_level, 1.0);
    let envMapColor_srgb = textureSampleLevel( t_env_map, s_env_map, vec3<f32>( -normal.x, normal.yz), mip_level );
    return srgb2physical(envMapColor_srgb.rgb) * u_material.env_map_intensity * PI;
}

fn getIBLRadiance(view_dir: vec3<f32>, normal: vec3<f32>, roughness: f32) -> vec3<f32> {
    $$ if env_mapping_mode == "CUBE-REFLECTION"
        var reflectVec = reflect( -view_dir, normal );
        let mip_level = getMipLevel(u_material.env_map_max_mip_level, roughness);
    $$ elif env_mapping_mode == "CUBE-REFRACTION"
        var reflectVec = refract( -view_dir, normal, u_material.refraction_ratio );
        let mip_level = 1.0;
    $$ endif
    reflectVec = normalize(mix(reflectVec, normal, roughness*roughness));
    let envMapColor_srgb = textureSampleLevel( t_env_map, s_env_map, vec3<f32>( -reflectVec.x, reflectVec.yz), mip_level );
    return srgb2physical(envMapColor_srgb.rgb) * u_material.env_map_intensity;
}


$$ if USE_ANISOTROPY is defined
fn getIBLAnisotropyRadiance(view_dir: vec3f, normal: vec3f, roughness: f32, bitangent: vec3f, anisotropy: f32) -> vec3f {
    var bent_normal = cross( bitangent, view_dir );
    bent_normal = normalize( cross( bent_normal, bitangent ) );
    bent_normal = normalize( mix( bent_normal, normal, pow2( pow2( 1.0 - anisotropy * ( 1.0 - roughness ) ) ) ) );

    return getIBLRadiance( view_dir, bent_normal, roughness );
}
$$ endif


$$ if USE_IRIDESCENCE is defined
fn computeMultiscatteringIridescence(normal: vec3<f32>, view_dir: vec3<f32>, specular_color: vec3<f32>,
        specular_f90: f32, roughness: f32, iridescence_f0: vec3<f32>, iridescence: f32,
        single_scatter: ptr<function, vec3f>, multi_scatter: ptr<function, vec3f>) {
$$ else
fn computeMultiscattering(normal: vec3<f32>, view_dir: vec3<f32>, specular_color: vec3<f32>, specular_f90: f32, roughness: f32,
        single_scatter: ptr<function, vec3f>, multi_scatter: ptr<function, vec3f>) {
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

    *single_scatter = FssEss;
    *multi_scatter = Fms * Ems;
}

fn RE_IndirectSpecular(radiance: vec3<f32>, irradiance: vec3<f32>, clearcoat_radiance: vec3<f32>,
        geometry: GeometricContext, material: PhysicalMaterial, reflected_light: ptr<function, ReflectedLight>){

    $$ if USE_CLEARCOAT is defined
        clearcoat_specular_indirect += clearcoat_radiance * EnvironmentBRDF( geometry.clearcoat_normal, geometry.view_dir, material.clearcoat_f0, material.clearcoat_f90, material.clearcoat_roughness );
    $$ endif

    $$ if USE_SHEEN is defined
        sheen_specular_indirect += irradiance * material.sheen_color * IBLSheenBRDF( geometry.normal, geometry.view_dir, material.sheen_roughness );
    $$ endif

    let cosine_weighted_irradiance: vec3<f32> = irradiance * RECIPROCAL_PI;
    var single_scatter: vec3<f32>;
    var multi_scatter: vec3<f32>;
    $$ if USE_IRIDESCENCE is defined
        computeMultiscatteringIridescence( geometry.normal, geometry.view_dir, material.specular_color, material.specular_f90, material.roughness, material.iridescence_f0, material.iridescence, &single_scatter, &multi_scatter );
    $$ else
        computeMultiscattering( geometry.normal, geometry.view_dir, material.specular_color, material.specular_f90, material.roughness, &single_scatter, &multi_scatter );
    $$ endif
    let total_scattering = single_scatter + multi_scatter;
    let diffuse = material.diffuse_color * ( 1.0 - max( max( total_scattering.r, total_scattering.g ), total_scattering.b ) );
    (*reflected_light).indirect_specular += (radiance * single_scatter + multi_scatter * cosine_weighted_irradiance);
    (*reflected_light).indirect_diffuse += diffuse * cosine_weighted_irradiance;
}

 //end of USE_IBL
$$ endif

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

    $$ if USE_SHEEN is defined
        sheen_specular_direct += irradiance * BRDF_Sheen( direct_light.direction, geometry.view_dir, geometry.normal, material.sheen_color, material.sheen_roughness );
    $$ endif

    (*reflected_light).direct_specular += irradiance * BRDF_GGX( direct_light.direction, geometry.view_dir, geometry.normal, material.specular_color, material.specular_f90, material.roughness, material );
    (*reflected_light).direct_diffuse += irradiance * BRDF_Lambert( material.diffuse_color );
}

fn computeSpecularOcclusion(dot_nv: f32, ambient_occlusion: f32, roughness: f32) -> f32 {
    let ao_nv = dot_nv + ambient_occlusion;
    let ao_exp = exp2( -16.0 * roughness - 1.0 );
    return saturate( pow(ao_nv, ao_exp) - 1.0 + ambient_occlusion );
}
