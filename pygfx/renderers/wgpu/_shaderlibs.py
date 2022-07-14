mesh_vertex_shader = """
    @stage(vertex)
    $$ if instanced
    fn vs_main(in: VertexInput, instance_info: InstanceInfo) -> Varyings {
    $$ else
    fn vs_main(in: VertexInput) -> Varyings {
    $$ endif

        // Select what face we're at
        let index = i32(in.vertex_index);
        let face_index = index / 3;
        var sub_index = index % 3;

        // If the camera flips a dimension, it flips the face winding.
        // We can correct for this by adjusting the order (sub_index) here.
        sub_index = select(sub_index, -1 * (sub_index - 1) + 1, u_stdinfo.flipped_winding > 0);

        // Get world transform
        $$ if instanced
            let instance_transform =  mat4x4<f32>(
                instance_info.transform0,
                instance_info.transform1,
                instance_info.transform2,
                instance_info.transform3,
            );
            let world_transform = u_wobject.world_transform * instance_transform;
        $$ else
            let world_transform = u_wobject.world_transform;
        $$ endif

        // Get vertex position
        let raw_pos = in.position;
        let world_pos = world_transform * vec4<f32>(raw_pos, 1.0);
        var ndc_pos = u_stdinfo.projection_transform * u_stdinfo.cam_transform * world_pos;

        // Prepare output
        var varyings: Varyings;

        // Set position
        varyings.world_pos = vec3<f32>(world_pos.xyz / world_pos.w);
        varyings.position = vec4<f32>(ndc_pos.xyz, ndc_pos.w);

        // Per-vertex colors TODO: use unified color format to remove this?
        $$ if vertex_color_channels == 1
        let cvalue = in.color;
        varyings.color = vec4<f32>(cvalue, cvalue, cvalue, 1.0);
        $$ elif vertex_color_channels == 2
        let cvalue = in.color;
        varyings.color = vec4<f32>(cvalue.r, cvalue.r, cvalue.r, cvalue.g);
        $$ elif vertex_color_channels == 3
        varyings.color = vec4<f32>(in.color, 1.0);
        $$ elif vertex_color_channels == 4
        varyings.color = in.color;
        $$ endif

        $$ if has_uv is defined
        $$ if uv_size == 1
        varyings.texcoord = f32(in.texcoord);
        $$ elif uv_size == 2
        varyings.texcoord = vec2<f32>(in.texcoord);
        $$ elif uv_size == 3
        varyings.texcoord = vec3<f32>(in.texcoord);
        $$ endif

        $$ endif

        // Set the normal
        let raw_normal = in.normal;
        let world_pos_n = world_transform * vec4<f32>(raw_pos + raw_normal, 1.0);
        let world_normal = normalize(world_pos_n - world_pos).xyz;
        varyings.normal = vec3<f32>(world_normal);

        varyings.geometry_normal = vec3<f32>(raw_normal);

        // Set varyings for picking. We store the face_index, and 3 weights
        // that indicate how close the fragment is to each vertex (barycentric
        // coordinates). This allows the selection of the nearest vertex or edge.
        $$ if instanced
            let pick_id = instance_info.id;
        $$ else
            let pick_id = u_wobject.id;
        $$ endif

        varyings.pick_id = u32(pick_id);
        varyings.pick_idx = u32(face_index);
        var arr_pick_coords = array<vec3<f32>, 3>(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(0.0, 0.0, 1.0));
        varyings.pick_coords = vec3<f32>(arr_pick_coords[sub_index]);

        return varyings;
    }
    """


lights = """
    fn getDistanceAttenuation(light_distance: f32, cutoff_distance: f32, decay_exponent: f32) -> f32 {
        if ( cutoff_distance > 0.0 && decay_exponent > 0.0 ) {
            return pow( clamp( - light_distance / cutoff_distance + 1.0, 0.0, 1.0), decay_exponent );
        }
        return 1.0;
    }
    fn smoothstep( low : f32, high : f32, x : f32 ) -> f32 {
        let t = clamp( ( x - low ) / ( high - low ), 0.0, 1.0 );
        return t * t * ( 3.0 - 2.0 * t );
    }
    fn getSpotAttenuation( coneCosine: f32, penumbraCosine: f32, angleCosine: f32 ) -> f32 {
        return smoothstep( coneCosine, penumbraCosine, angleCosine );
    }
    fn getAmbientLightIrradiance( ambientlight_color: vec3<f32> ) -> vec3<f32> {
        let irradiance = ambientlight_color;
        return irradiance;
    }
    struct IncidentLight {
        color: vec3<f32>,
        direction: vec3<f32>,
        visible: bool,
    };
    struct ReflectedLight {
        direct_diffuse: vec3<f32>,
        direct_specular: vec3<f32>,
        indirect_diffuse: vec3<f32>,
        indirect_specular: vec3<f32>,
    };
    struct GeometricContext {
        position: vec3<f32>,
        normal: vec3<f32>,
        view_dir: vec3<f32>,
    };
    $$ if num_dir_lights > 0
    fn getDirectionalLightInfo( directional_light: DirectionalLight, geometry: GeometricContext ) -> IncidentLight {
        var light: IncidentLight;
        light.color = directional_light.color.rgb;
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
        light.color = point_light.color.rgb;
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
            light.color = spot_light.color.rgb * spot_attenuation;
            light.color *= getDistanceAttenuation( light_distance, spot_light.distance, spot_light.decay );
            light.visible = any(light.color != vec3<f32>(0.0));
        } else {
            light.color = vec3<f32>( 0.0 );
            light.visible = false;
        }
        return light;
    }
    $$ endif
    """

bsdfs = """
    fn BRDF_Lambert(diffuse_color: vec3<f32>) -> vec3<f32> {
        return 0.3183098861837907 * diffuse_color;  //  1/pi = 0.3183098861837907
    }
    fn F_Schlick(f0: vec3<f32>, f90: f32, dot_vh: f32,) -> vec3<f32> {
        // Optimized variant (presented by Epic at SIGGRAPH '13)
        // https://cdn2.unrealengine.com/Resources/files/2013SiggraphPresentationsNotes-26915738.pdf
        let fresnel = exp2( ( - 5.55473 * dot_vh - 6.98316 ) * dot_vh );
        return f0 * ( 1.0 - fresnel ) + ( f90 * fresnel );
    }
"""

bsdfs_blinn_phong = """
    fn G_BlinnPhong_Implicit() -> f32 {
        return 0.25;
    }
    fn D_BlinnPhong(shininess: f32, dot_nh: f32) -> f32 {
        return 0.3183098861837907 * ( shininess * 0.5 + 1.0 ) * pow( dot_nh, shininess );
    }
    fn BRDF_BlinnPhong(
        light_dir: vec3<f32>,
        view_dir: vec3<f32>,
        normal: vec3<f32>,
        specular_color: vec3<f32>,
        shininess: f32,
    ) -> vec3<f32> {
        let half_dir = normalize(light_dir + view_dir);
        let dot_nh = clamp(dot(normal, half_dir), 0.0, 1.0);
        let dot_vh = clamp(dot(view_dir, half_dir), 0.0, 1.0);
        let F = F_Schlick(specular_color, 1.0, dot_vh);
        let G = G_BlinnPhong_Implicit();
        let D = D_BlinnPhong(shininess, dot_nh);
        return F * ( G * D );
    }
"""

blinn_phong_lighting = """
    struct BlinnPhongMaterial {
        diffuse_color: vec3<f32>,
        specular_color: vec3<f32>,
        specular_shininess: f32,
        specular_strength: f32,
    };
    fn RE_Direct_BlinnPhong(
        direct_light: IncidentLight,
        geometry: GeometricContext,
        material: BlinnPhongMaterial,
        reflected_light: ReflectedLight,
    ) -> ReflectedLight {
        let dot_nl = clamp(dot(geometry.normal, direct_light.direction), 0.0, 1.0);
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
"""


bsdfs_physical = """
    fn V_GGX_SmithCorrelated(alpha: f32, dot_nl: f32, dot_nv: f32) -> f32 {
        let a2 = pow(alpha, 2.0);
        let gv = dot_nl * sqrt(a2 + (1.0-a2) * pow(dot_nv, 2.0));
        let gl = dot_nv * sqrt(a2 + (1.0-a2) * pow(dot_nl, 2.0 ));
        let epsilon = 1.0e-6;
        return 0.5/ max( gv+gl, epsilon);
    }
    fn D_GGX(alpha: f32, dot_nh: f32) -> f32 {
        let a2 = pow( alpha, 2.0 );
        let denom = pow(dot_nh, 2.0) * (a2 - 1.0) + 1.0;
        return 0.3183098861837907 * a2/pow(denom, 2.0);
    }
    fn BRDF_GGX(light_dir: vec3<f32>, view_dir: vec3<f32>, normal: vec3<f32>, f0: vec3<f32>, f90: f32, roughness: f32) -> vec3<f32> {
        let alpha = pow( roughness, 2.0 );
        let half_dir = normalize( light_dir + view_dir );
        let dot_nl = clamp( dot( normal, light_dir ), 0.0, 1.0 );
        let dot_nv = clamp( dot( normal, view_dir ), 0.0, 1.0 );
        let dot_nh = clamp( dot( normal, half_dir ), 0.0, 1.0 );
        let dot_vh = clamp( dot( view_dir, half_dir ), 0.0, 1.0 );
        let F = F_Schlick( f0, f90, dot_vh);
        let V = V_GGX_SmithCorrelated( alpha, dot_nl, dot_nv );
        let D = D_GGX( alpha, dot_nh );
        return F * ( V * D );
    }
    fn DFGApprox( normal: vec3<f32>, view_dir: vec3<f32>, roughness: f32 ) -> vec2<f32>{
        let dot_nv = clamp( dot( normal, view_dir ), 0.0, 1.0);
        let c0 = vec4<f32>(- 1.0, - 0.0275, - 0.572, 0.022);
        let c1 = vec4<f32>(1.0, 0.0425, 1.04, - 0.04);
        let r = roughness * c0 + c1;
        let a004 = min( r.x * r.x, exp2( - 9.28 * dot_nv ) ) * r.x + r.y;
        let fab: vec2<f32> = vec2<f32>( - 1.04, 1.04 ) * a004 + r.zw;
        return fab;
    }
"""

physical_lighting = """
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
    fn getIBLIrradiance( normal: vec3<f32>, env_map: texture_cube<f32>, env_map_sampler: sampler, mip_level: f32) -> vec4<f32> {
        let envMapColor = textureSampleLevel( env_map, env_map_sampler, vec3<f32>( -normal.x, normal.yz), mip_level );
        return envMapColor;
    }
    fn getIBLRadiance( view_dir: vec3<f32>, normal: vec3<f32>, roughness: f32, env_map: texture_cube<f32>, env_map_sampler: sampler, mip_level: f32 ) -> vec4<f32> {
        var reflectVec = reflect( - view_dir, normal );
        reflectVec = normalize(mix(reflectVec, normal, roughness*roughness));
        let envMapColor = textureSampleLevel( env_map, env_map_sampler, vec3<f32>( -reflectVec.x, reflectVec.yz), mip_level );
        return envMapColor;
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
        let cosineWeightedIrradiance: vec3<f32> = irradiance * 0.3183098861837907;
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
        let dot_nl = clamp( dot( geometry.normal, direct_light.direction ), 0.0, 1.0 );
        let irradiance = dot_nl * direct_light.color;
        var out_reflected_light: ReflectedLight = reflected_light;
        out_reflected_light.direct_specular += irradiance * BRDF_GGX( direct_light.direction, geometry.view_dir, geometry.normal, material.specular_color, material.specular_f90, material.roughness );
        out_reflected_light.indirect_diffuse += irradiance * BRDF_Lambert( material.diffuse_color );
        return out_reflected_light;
    }
    fn RE_AmbientOcclusion_Physical(ambientOcclusion: f32, geometry: GeometricContext, material: PhysicalMaterial, reflected_light: ReflectedLight) -> ReflectedLight {
        let dot_nv = clamp( dot( geometry.normal, geometry.view_dir ), 0.0, 1.0);
        let ao_nv = dot_nv + ambientOcclusion;
        let ao_exp = exp2( -16.0 * material.roughness - 1.0 );
        let ao = clamp( pow(ao_nv, ao_exp) - 1.0 + ambientOcclusion, 0.0, 1.0 );
        var out_reflected_light: ReflectedLight = reflected_light;
        out_reflected_light.indirect_diffuse *= ambientOcclusion;
        out_reflected_light.indirect_specular *= ao;
        return out_reflected_light;
    }
"""


shadow = """
    fn get_shadow(t_shadow: texture_depth_2d_array, u_shadow_sampler: sampler_comparison, layer_index: i32, shadow_coords: vec4<f32>, bias: f32) -> f32 {
        if (shadow_coords.w <= 0.0) {
            return 1.0;
        }
        // compensate for the Y-flip difference between the NDC and texture coordinates
        let proj_coords = shadow_coords.xyz / shadow_coords.w;
        let flip_correction = vec2<f32>(0.5, -0.5);
        let light_local = proj_coords.xy * flip_correction + vec2<f32>(0.5, 0.5);
        if (light_local.x < 0.0 || light_local.x > 1.0 || light_local.y < 0.0 || light_local.y > 1.0) {
            return 1.0;
        }
        var depth:f32 = proj_coords.z - bias;
        depth = clamp(depth, 0.0, 1.0);
        var shadow: f32 = 0.0;
        // for (var i:i32 = -1; i <= 1; i++) {
        //    for (var j:i32 = -1; j <= 1; j++) {
        //        shadow += textureSampleCompareLevel(t_shadow, u_shadow_sampler, light_local, layer_index, depth, vec2<i32>(i, j));
        //    }
        // }
        // use for loop?
        // PCF
        shadow += textureSampleCompareLevel(t_shadow, u_shadow_sampler, light_local, layer_index, depth, vec2<i32>(0, 0));
        shadow += textureSampleCompareLevel(t_shadow, u_shadow_sampler, light_local, layer_index, depth, vec2<i32>(1, 0));
        shadow += textureSampleCompareLevel(t_shadow, u_shadow_sampler, light_local, layer_index, depth, vec2<i32>(0, 1));
        shadow += textureSampleCompareLevel(t_shadow, u_shadow_sampler, light_local, layer_index, depth, vec2<i32>(1, 1));
        shadow += textureSampleCompareLevel(t_shadow, u_shadow_sampler, light_local, layer_index, depth, vec2<i32>(-1, 0));
        shadow += textureSampleCompareLevel(t_shadow, u_shadow_sampler, light_local, layer_index, depth, vec2<i32>(0, -1));
        shadow += textureSampleCompareLevel(t_shadow, u_shadow_sampler, light_local, layer_index, depth, vec2<i32>(-1, -1));
        shadow += textureSampleCompareLevel(t_shadow, u_shadow_sampler, light_local, layer_index, depth, vec2<i32>(-1, 1));
        shadow += textureSampleCompareLevel(t_shadow, u_shadow_sampler, light_local, layer_index, depth, vec2<i32>(1, -1));
        shadow += textureSampleCompareLevel(t_shadow, u_shadow_sampler, light_local, layer_index, depth, vec2<i32>(2, 0));
        shadow += textureSampleCompareLevel(t_shadow, u_shadow_sampler, light_local, layer_index, depth, vec2<i32>(0, 2));
        shadow += textureSampleCompareLevel(t_shadow, u_shadow_sampler, light_local, layer_index, depth, vec2<i32>(2, 2));
        shadow += textureSampleCompareLevel(t_shadow, u_shadow_sampler, light_local, layer_index, depth, vec2<i32>(-2, 0));
        shadow += textureSampleCompareLevel(t_shadow, u_shadow_sampler, light_local, layer_index, depth, vec2<i32>(0, -2));
        shadow += textureSampleCompareLevel(t_shadow, u_shadow_sampler, light_local, layer_index, depth, vec2<i32>(-2, -2));
        shadow += textureSampleCompareLevel(t_shadow, u_shadow_sampler, light_local, layer_index, depth, vec2<i32>(-2, 2));
        shadow += textureSampleCompareLevel(t_shadow, u_shadow_sampler, light_local, layer_index, depth, vec2<i32>(2, -2));
        shadow /= 17.0;
        return shadow;
    }
    fn uv_to_direction( face:i32,  uv:vec2<f32>)->vec3<f32> {
        var u = 2.0 * uv.x - 1.0;
        var v = -2.0 * uv.y + 1.0;
        switch face{
            case 0: {
                return vec3<f32>(1.0, v, -u);
            }
            case 1: {
                return vec3<f32>(-1.0, v, u);
            }
            case 2: {
                return vec3<f32>(u, 1.0, -v);
            }
            case 3: {
                return vec3<f32>(u, -1.0, v);
            }
            case 4: {
                return vec3<f32>(u, v, 1.0);
            }
            case 5: {
                return vec3<f32>(-u, v, -1.0);
            }
            default: {
                return vec3<f32>(1.0, 1.0, 1.0);
            }
        }
    }
    fn get_cube_shadow(t_shadow: texture_depth_cube_array, u_shadow_sampler: sampler_comparison, layer_index: i32,
        light_view_proj: array<mat4x4<f32>,6>, world_pos: vec3<f32>, light_direction: vec3<f32>,  bias: f32) -> f32 {
        //var direction = world_pos - light_pos;
        var direction = -light_direction;
        let scale = 1.0 / max(max(abs(direction.x), abs(direction.y)), abs(direction.z));
        direction = direction * scale;
        let epsilon = 1.0e-6;
        var faceIndex = 0;
        var view_proj: mat4x4<f32>;
        if (abs(direction.x - 1.0) < epsilon) {
            faceIndex = 0;
            view_proj = light_view_proj[0];
        } else if (abs(direction.x + 1.0) < epsilon) {
            faceIndex = 1;
            view_proj = light_view_proj[1];
        }  else if (abs(direction.y - 1.0) < epsilon) {
            faceIndex = 2;
            view_proj = light_view_proj[2];
        } else if (abs(direction.y + 1.0) < epsilon) {
            faceIndex = 3;
            view_proj = light_view_proj[3];
        } else if (abs(direction.z - 1.0) < epsilon) {
            faceIndex = 4;
            view_proj = light_view_proj[4];
        } else if (abs(direction.z + 1.0) < epsilon) {
            faceIndex = 5;
            view_proj = light_view_proj[5];
        }
        var shadow_coords = view_proj * vec4<f32>(world_pos, 1.0);
        if (shadow_coords.w <= 0.0) {
            return 1.0;
        }
        let proj_coords = shadow_coords.xyz / shadow_coords.w;
        let flip_correction = vec2<f32>(0.5, -0.5);
        let light_local = proj_coords.xy * flip_correction + vec2<f32>(0.5, 0.5);
        var depth:f32 = proj_coords.z - bias; // bias?
        depth = clamp(depth, 0.0, 1.0);
        var dir = uv_to_direction(faceIndex, light_local);
        var shadow = textureSampleCompareLevel(t_shadow, u_shadow_sampler, dir, layer_index, depth);
        return shadow;
    }
"""
