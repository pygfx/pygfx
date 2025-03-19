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

fn RE_Direct(
    direct_light: IncidentLight,
    geometry: GeometricContext,
    material: BlinnPhongMaterial,
    reflected_light: ptr<function, ReflectedLight>,
) {
    let dot_nl = saturate(dot(geometry.normal, direct_light.direction));
    let irradiance = dot_nl * direct_light.color;
    let direct_diffuse = irradiance * BRDF_Lambert( material.diffuse_color );
    let direct_specular = irradiance * BRDF_BlinnPhong( direct_light.direction, geometry.view_dir, geometry.normal, material.specular_color, material.specular_shininess ) * material.specular_strength;
    (*reflected_light).direct_diffuse += direct_diffuse;
    (*reflected_light).direct_specular += direct_specular;
}

fn RE_IndirectDiffuse(
    irradiance: vec3<f32>,
    geometry: GeometricContext,
    material: BlinnPhongMaterial,
    reflected_light: ptr<function, ReflectedLight>,
) {
    let indirect_diffuse = irradiance * BRDF_Lambert( material.diffuse_color );
    (*reflected_light).indirect_diffuse += indirect_diffuse;
}
