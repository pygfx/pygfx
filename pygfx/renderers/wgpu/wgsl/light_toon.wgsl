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

fn RE_Direct_Toon(
    direct_light: IncidentLight,
    geometry: GeometricContext,
    material: ToonMaterial,
    reflected_light: ptr<function, ReflectedLight>,
) {
    let irradiance = getGradientIrradiance( geometry.normal, direct_light.direction ) * direct_light.color;
    let direct_diffuse = irradiance * BRDF_Lambert( material.diffuse_color );
    (*reflected_light).direct_diffuse += direct_diffuse;
}

fn RE_IndirectDiffuse_Toon(
    irradiance: vec3<f32>,
    geometry: GeometricContext,
    material: ToonMaterial,
    reflected_light: ptr<function, ReflectedLight>,
) {
    let indirect_diffuse = irradiance * BRDF_Lambert( material.diffuse_color );
    (*reflected_light).indirect_diffuse += indirect_diffuse;
}

fn lighting_toon(
    reflected_light: ptr<function, ReflectedLight>,
    geometry: GeometricContext,
    material: ToonMaterial,
)  {

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
            RE_Direct_Toon( light, geometry, material, reflected_light );
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
            RE_Direct_Toon( light, geometry, material, reflected_light );
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
            RE_Direct_Toon( light, geometry, material, reflected_light );
            continuing {
                i += 1;
            }
        }
    $$ endif
}