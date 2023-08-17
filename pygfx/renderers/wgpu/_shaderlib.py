# Some guidelines:
#
# Put code in logical blocks using methods. That way they can be reused
# and also show up in the "source structure" of most IDE's.
#
# Colors should be converted to the physical colorspace in the fragment
# shader (because you generally want to interolate in srgb). Other than
# that, the color should be turned to physical using srgb2physical()
# as soon as reasonably possible. For clarity it helps to add a "_srgb"
# suffix to colors not yet converted. Most colors on a uniform can also
# be considered srgb.
#
# The function names are left camelCase, many come from ThreeJS and this makes it
# easy to look them up. In general, the variables are snake_case.
#
# The PI and RECIPROCAL scale factors for the lights is a complex story
# related to artist-friendly colors, historic design choices, and
# undoing the PI factor in other places. We just try follow whay ThreeJS
# is doing here.


class Shaderlib:
    def light_deps_basic(self):
        return """
        // TODO smoothstep and saturate probably exists when Naga gets updated
        fn _smoothstep( low : f32, high : f32, x : f32 ) -> f32 {
            let t = clamp( ( x - low ) / ( high - low ), 0.0, 1.0 );
            return t * t * ( 3.0 - 2.0 * t );
        }
        fn _saturate( x: f32) -> f32 {
            return clamp(x, 0.0, 1.0);
        }
        fn getDistanceAttenuation(light_distance: f32, cutoff_distance: f32, decay_exponent: f32) -> f32 {
            var distance_falloff: f32 = 1.0 / max( pow( light_distance, decay_exponent ), 0.01 );
            if ( cutoff_distance > 0.0 ) {
                distance_falloff *= pow2( _saturate( 1.0 - pow4( light_distance / cutoff_distance ) ) );
            }
            return distance_falloff;
        }
        fn getSpotAttenuation( cone_cosine: f32, penumbra_cosine: f32, angle_cosine: f32 ) -> f32 {
            return _smoothstep( cone_cosine, penumbra_cosine, angle_cosine );
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
        """

    def light_deps_pbr(self):
        return """
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

    def bsdfs_basic(self):
        # Bidirectional scattering distribution function
        return """
        fn BRDF_Lambert(diffuse_color: vec3<f32>) -> vec3<f32> {
            return RECIPROCAL_PI * diffuse_color;
        }
        fn F_Schlick(f0: vec3<f32>, f90: f32, dot_vh: f32,) -> vec3<f32> {
            // Optimized variant (presented by Epic at SIGGRAPH '13)
            // https://cdn2.unrealengine.com/Resources/files/2013SiggraphPresentationsNotes-26915738.pdf
            let fresnel = exp2( ( - 5.55473 * dot_vh - 6.98316 ) * dot_vh );
            return f0 * ( 1.0 - fresnel ) + ( f90 * fresnel );
        }
    """

    def bsdfs_pbr(self):
        return """
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
            return RECIPROCAL_PI * a2/pow(denom, 2.0);
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

    def shadow(self):
        return """
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
                case 0: { return vec3<f32>(1.0, v, -u); }
                case 1: { return vec3<f32>(-1.0, v, u); }
                case 2: { return vec3<f32>(u, 1.0, -v); }
                case 3: { return vec3<f32>(u, -1.0, v); }
                case 4: { return vec3<f32>(u, v, 1.0); }
                case 5: { return vec3<f32>(-u, v, -1.0); }
                default: { return vec3<f32>(1.0, 1.0, 1.0); }
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

    def lighting_phong_simple(self):
        # Simple phong shading with builtin lights. What we used before #324
        return """
        fn lighting_phong(
            is_front: bool,
            varyings: Varyings,
            normal: vec3<f32>,
            view_dir: vec3<f32>,
            albeido: vec3<f32>,
        ) -> vec3<f32> {
            let light_color = srgb2physical(vec3<f32>(1.0, 1.0, 1.0));

            // Light parameters
            let ambient_factor = 0.1;
            let diffuse_factor = 0.7;
            let specular_factor = 0.3;
            let shininess = u_material.shininess;

            // Base vectors
            let view = normalize(view_dir);
            let light = view;
            var normal = select(-normal, normal, is_front);  // See pygfx/issues/#105 for details

            // Ambient
            let ambient_color = light_color * ambient_factor;

            // Diffuse (blinn-phong reflection model)
            let lambert_term = clamp(dot(light, normal), 0.0, 1.0);
            let diffuse_color = diffuse_factor * light_color * lambert_term;

            // Specular
            let halfway = normalize(light + view);  // halfway vector
            var specular_term = pow(clamp(dot(halfway,  normal), 0.0, 1.0), shininess);
            specular_term = select(0.0, specular_term, shininess > 0.0);
            let specular_color = specular_factor * specular_term * light_color;

            // Emissive color is additive and unaffected by lights
            let emissive_color = srgb2physical(u_material.emissive_color.rgb);

            // Put together
            return albeido * (ambient_color + diffuse_color) + specular_color + emissive_color;
        }
        """

    # %%%%% The actual entrypoints used by the shaders %%%%%

    def lighting_phong(self):
        # Phong shading using the lights present in the scene
        return (
            self.bsdfs_basic()
            + self.light_deps_basic()
            + """

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
            let dot_nh = clamp(dot(normal, half_dir), 0.0, 1.0);
            let dot_vh = clamp(dot(view_dir, half_dir), 0.0, 1.0);
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

        fn lighting_phong(
            varyings: Varyings,
            normal: vec3<f32>,
            view_dir: vec3<f32>,
            albeido: vec3<f32>,
        ) -> vec3<f32> {

            // Colors incoming via uniforms
            let specular_color = srgb2physical(u_material.specular_color.rgb);
            let ambient_color = u_ambient_light.color.rgb; // the one exception that is already physical

            var material: BlinnPhongMaterial;
            material.diffuse_color = albeido;
            material.specular_color = specular_color;
            material.specular_shininess = u_material.shininess;
            material.specular_strength = 1.0;   //  We could provide a specular map
            var reflected_light: ReflectedLight = ReflectedLight(vec3<f32>(0.0), vec3<f32>(0.0), vec3<f32>(0.0), vec3<f32>(0.0));

            var geometry: GeometricContext;
            geometry.position = varyings.world_pos;
            geometry.normal = normal;
            geometry.view_dir = view_dir;
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
                    reflected_light = RE_Direct_BlinnPhong( light, geometry, material, reflected_light );
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
                    reflected_light = RE_Direct_BlinnPhong( light, geometry, material, reflected_light );
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
                    reflected_light = RE_Direct_BlinnPhong( light, geometry, material, reflected_light );
                    continuing {
                        i += 1;
                    }
                }
            $$ endif
            var irradiance = getAmbientLightIrradiance( ambient_color );

            // Light map (pre-baked lighting)
            $$ if use_light_map is defined
            let light_map_color = srgb2physical( textureSample( t_light_map, s_light_map, varyings.texcoord1 ).rgb );
            irradiance += light_map_color * u_material.light_map_intensity;
            $$ endif

            reflected_light = RE_IndirectDiffuse_BlinnPhong( irradiance, geometry, material, reflected_light );

            // Ambient occlusion
            $$ if use_ao_map is defined
            let ao_map_intensity = u_material.ao_map_intensity;
            let ambientOcclusion = ( textureSample( t_ao_map, s_ao_map, varyings.texcoord1 ).r - 1.0 ) * ao_map_intensity + 1.0;
            reflected_light.indirect_diffuse *= ambientOcclusion;
            $$ endif

            return reflected_light.direct_diffuse + reflected_light.direct_specular + reflected_light.indirect_diffuse + reflected_light.indirect_specular + u_material.emissive_color.rgb;
        }
        """
        )

    def lighting_pbr(self):
        return (
            self.bsdfs_basic()
            + self.bsdfs_pbr()
            + self.light_deps_basic()
            + self.light_deps_pbr()
            + """

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
        """
        )


shaderlib = Shaderlib()
