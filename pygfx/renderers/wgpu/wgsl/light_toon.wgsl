{$ include 'pygfx.light_common.wgsl' $}

struct ToonMaterial {
    diffuse_color: vec3<f32>,
};

fn getGradientIrradiance( normal: vec3<f32>, light_dir: vec3<f32> ) -> vec3<f32> {
    let dot_nl = dot(normal, light_dir);
    let coord = vec2f(dot_nl *0.5 + 0.5, 0.0);

    $$ if use_gradient_map is defined
        return vec3f(textureSample(t_gradient_map, s_gradient_map, coord).r);
    $$ else
        let fw = fwidth(coord) * 0.5;
        return mix( vec3f(0.7), vec3f(1.0), smoothstep(0.7-fw.x, 0.7+fw.x, coord.x) );
    $$ endif
}

fn RE_Direct(
    direct_light: IncidentLight,
    geometry: GeometricContext,
    material: ToonMaterial,
    reflected_light: ptr<function, ReflectedLight>,
) {
    let irradiance = getGradientIrradiance( geometry.normal, direct_light.direction ) * direct_light.color;
    let direct_diffuse = irradiance * BRDF_Lambert( material.diffuse_color );
    (*reflected_light).direct_diffuse += direct_diffuse;
}

fn RE_IndirectDiffuse(
    irradiance: vec3<f32>,
    geometry: GeometricContext,
    material: ToonMaterial,
    reflected_light: ptr<function, ReflectedLight>,
) {
    let indirect_diffuse = irradiance * BRDF_Lambert( material.diffuse_color );
    (*reflected_light).indirect_diffuse += indirect_diffuse;
}