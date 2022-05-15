mesh_vertex_shader = """
    struct VertexInput {
        @builtin(vertex_index) vertex_index : u32,
        $$ if instanced
        @builtin(instance_index) instance_index : u32,
        $$ endif

        $$ for vertex_attribute in vertex_attributes
        @location({{loop.index0}}) {{vertex_attribute[0]}} : {{vertex_attribute[1]}},
        $$ endfor

    };

    $$ if instanced
    struct InstanceInfo {
        @location({{ vertex_attributes|length }}) transform0: vec4<f32>,
        @location({{ vertex_attributes|length + 1 }}) transform1: vec4<f32>,
        @location({{ vertex_attributes|length + 2 }}) transform2: vec4<f32>,
        @location({{ vertex_attributes|length + 3 }}) transform3: vec4<f32>,
        @location({{ vertex_attributes|length + 4 }}) id: u32,
    };
    $$ endif

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
            // let instance_info = s_instance_infos[in.instance_index];
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

        // Per-vertex colors
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

        // Set texture coords
        $$ if colormap_dim == '1d'
        varyings.texcoord = f32(in.texcoord);
        $$ elif colormap_dim == '2d'
        varyings.texcoord = vec2<f32>(in.texcoord);
        $$ elif colormap_dim == '3d'
        varyings.texcoord = vec3<f32>(in.texcoord);
        $$ endif

        // Set the normal
        let raw_normal = in.normal;
        let world_pos_n = world_transform * vec4<f32>(raw_pos + raw_normal, 1.0);
        let world_normal = normalize(world_pos_n - world_pos).xyz;
        varyings.normal = vec3<f32>(world_normal);


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

    """


bsdfs = """
    fn BRDF_Lambert(diffuse_color: vec3<f32>) -> vec3<f32> {
        return 0.3183098861837907 * diffuse_color;  // 1 / pi = 0.3183098861837907
    }

    fn F_Schlick(f0: vec3<f32>, f90: f32, dot_vh: f32,) -> vec3<f32> {
        // Optimized variant (presented by Epic at SIGGRAPH '13)
        // https://cdn2.unrealengine.com/Resources/files/2013SiggraphPresentationsNotes-26915738.pdf
        let fresnel = exp2( ( - 5.55473 * dot_vh - 6.98316 ) * dot_vh );

        return f0 * ( 1.0 - fresnel ) + ( f90 * fresnel );
    }

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

blinn_phong = """
    struct BlinnPhongMaterial {
        diffuse_color: vec3<f32>,
        specular_color: vec3<f32>,
        specular_shininess: f32,
        specular_strength: f32,
    };


    fn RE_Direct_BlinnPhong(
        direct_light: IncidentLight,
        reflected_light: ReflectedLight,
        view_dir: vec3<f32>,
        normal: vec3<f32>,
        material: BlinnPhongMaterial,
    ) -> ReflectedLight {

        let dot_nl = clamp(dot(normal, direct_light.direction), 0.0, 1.0);

        let irradiance = dot_nl * direct_light.color;

        let direct_diffuse = irradiance * BRDF_Lambert( material.diffuse_color );

        let direct_specular = irradiance * BRDF_BlinnPhong( direct_light.direction, view_dir, normal, material.specular_color, material.specular_shininess ) * material.specular_strength;


        var out_reflected_light: ReflectedLight;

        out_reflected_light.direct_diffuse = reflected_light.direct_diffuse + direct_diffuse;
        out_reflected_light.direct_specular = reflected_light.direct_specular + direct_specular;
        out_reflected_light.indirect_diffuse = reflected_light.indirect_diffuse;
        out_reflected_light.indirect_specular = reflected_light.indirect_specular;

        return out_reflected_light;
    }

    fn RE_IndirectDiffuse_BlinnPhong(
        irradiance: vec3<f32>,
        reflected_light: ReflectedLight,
        material: BlinnPhongMaterial,
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
