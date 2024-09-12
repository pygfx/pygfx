
// Get info on the number of lights, their types, positions, and colors
{{ light_definitions }}


fn getDistanceAttenuation(light_distance: f32, cutoff_distance: f32, decay_exponent: f32) -> f32 {
    var distance_falloff: f32 = 1.0 / max( pow( light_distance, decay_exponent ), 0.01 );
    if ( cutoff_distance > 0.0 ) {
        distance_falloff *= pow2( saturate( 1.0 - pow4( light_distance / cutoff_distance ) ) );
    }
    return distance_falloff;
}

fn getSpotAttenuation( cone_cosine: f32, penumbra_cosine: f32, angle_cosine: f32 ) -> f32 {
    return smoothstep( cone_cosine, penumbra_cosine, angle_cosine );
}

fn getAmbientLightIrradiance( ambientlight_color: vec3<f32> ) -> vec3<f32> {
    let irradiance = ambientlight_color;
    return irradiance;
}

struct IncidentLight {
    color: vec3<f32>,
    visible: bool,
    direction: vec3<f32>,
};

struct GeometricContext {
    position: vec3<f32>,
    normal: vec3<f32>,
    view_dir: vec3<f32>,
};

$$ if num_dir_lights > 0
fn getDirectionalLightInfo( directional_light: DirectionalLight, geometry: GeometricContext ) -> IncidentLight {
    var light: IncidentLight;
    light.color = srgb2physical(directional_light.color.rgb) * directional_light.intensity;
    light.direction = -directional_light.direction.xyz;
    light.visible = true;
    return light;
}
$$ endif

$$ if num_point_lights > 0
fn getPointLightInfo( point_light: PointLight, geometry: GeometricContext ) -> IncidentLight {
    var light: IncidentLight;
    let i_vector = point_light.world_transform[3].xyz - geometry.position;
    light.direction = normalize(i_vector);
    let light_distance = length(i_vector);
    light.color = srgb2physical(point_light.color.rgb) * point_light.intensity;
    light.color *= getDistanceAttenuation( light_distance, point_light.distance, point_light.decay );
    light.visible = any(light.color != vec3<f32>(0.0));
    return light;
}
$$ endif

$$ if num_spot_lights > 0
fn getSpotLightInfo( spot_light: SpotLight, geometry: GeometricContext ) -> IncidentLight {
    var light: IncidentLight;
    let i_vector = spot_light.world_transform[3].xyz - geometry.position;
    light.direction = normalize(i_vector);
    let angle_cos = dot(light.direction, -spot_light.direction.xyz);
    let spot_attenuation = getSpotAttenuation(spot_light.cone_cos, spot_light.penumbra_cos, angle_cos);
    if ( spot_attenuation > 0.0 ) {
        let light_distance = length( i_vector );
        light.color = srgb2physical(spot_light.color.rgb) * spot_light.intensity;
        light.color *= spot_attenuation;
        light.color *= getDistanceAttenuation( light_distance, spot_light.distance, spot_light.decay );
        light.visible = any(light.color != vec3<f32>(0.0));
    } else {
        light.color = vec3<f32>( 0.0 );
        light.visible = false;
    }
    return light;
}
$$ endif

// Bidirectional scattering distribution function

fn BRDF_Lambert(diffuse_color: vec3<f32>) -> vec3<f32> {
    return RECIPROCAL_PI * diffuse_color;
}

fn F_Schlick(f0: vec3<f32>, f90: f32, dot_vh: f32,) -> vec3<f32> {
    // Optimized variant (presented by Epic at SIGGRAPH '13)
    // https://cdn2.unrealengine.com/Resources/files/2013SiggraphPresentationsNotes-26915738.pdf
    let fresnel = exp2( ( - 5.55473 * dot_vh - 6.98316 ) * dot_vh );
    return f0 * ( 1.0 - fresnel ) + ( f90 * fresnel );
}

$$ if use_normal_map is defined
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
$$ endif