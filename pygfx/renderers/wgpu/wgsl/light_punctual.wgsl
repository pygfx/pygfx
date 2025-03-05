$$ if num_point_lights > 0
    for ( var i = 0; i < {{num_point_lights}}; i ++ ) {
        let point_light = u_point_lights[i];
        var punctual_light = getPointLightInfo(point_light, geometry);
        if (! punctual_light.visible) { continue; }
        $$ if receive_shadow
        if (point_light.cast_shadow != 0){
            let shadow = get_cube_shadow(u_shadow_map_point_light, u_shadow_sampler, i, point_light.light_view_proj_matrix, geometry.position, punctual_light.direction, point_light.shadow_bias);
            punctual_light.color *= shadow;
        }
        $$ endif
        RE_Direct( punctual_light, geometry, material, &reflected_light );
    }
$$ endif

$$ if num_spot_lights > 0
    for ( var i = 0; i < {{num_spot_lights}}; i ++ ) {
        let spot_light = u_spot_lights[i];
        var punctual_light = getSpotLightInfo(spot_light, geometry);
        if (! punctual_light.visible) { continue; }
        $$ if receive_shadow
        if (spot_light.cast_shadow != 0){
            let coords = spot_light.light_view_proj_matrix * vec4<f32>(geometry.position,1.0);
            let shadow = get_shadow(u_shadow_map_spot_light, u_shadow_sampler, i, coords, spot_light.shadow_bias);
            punctual_light.color *= shadow;
        }
        $$ endif
        RE_Direct( punctual_light, geometry, material, &reflected_light );
    }
$$ endif

$$ if num_dir_lights > 0
    for ( var i = 0; i < {{num_dir_lights}}; i ++ ) {
        let dir_light = u_directional_lights[i];
        var punctual_light = getDirectionalLightInfo(dir_light, geometry);
        if (! punctual_light.visible) { continue; }
        $$ if receive_shadow
        if (dir_light.cast_shadow != 0) {
            let coords = dir_light.light_view_proj_matrix * vec4<f32>(geometry.position,1.0);
            let shadow = get_shadow(u_shadow_map_dir_light, u_shadow_sampler, i, coords, dir_light.shadow_bias);
            punctual_light.color *= shadow;
        }
        $$ endif
        RE_Direct( punctual_light, geometry, material, &reflected_light );
    }
$$ endif