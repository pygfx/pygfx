// Mesh shader


{# Includes #}
{$ include 'pygfx.std.wgsl' $}

$$ if use_colormap is defined
    {$ include 'pygfx.colormap.wgsl' $}
$$ endif

$$ if receive_shadow
    {$ include 'pygfx.light_shadow.wgsl' $}
$$ endif
$$ if lighting == 'phong'
    {$ include 'pygfx.light_phong.wgsl' $}
$$ elif lighting == 'pbr'
    {$ include 'pygfx.light_pbr.wgsl' $}
$$ elif lighting == 'toon'
    {$ include 'pygfx.light_toon.wgsl' $}
$$ endif

$$ if USE_IRIDESCENCE is defined
    {$ include 'pygfx.iridescence.wgsl' $}
$$ endif

struct VertexInput {
    @builtin(vertex_index) vertex_index : u32,
    $$ if instanced
    @builtin(instance_index) instance_index : u32,
    $$ endif
};

$$ if instanced
struct InstanceInfo {
    transform: mat4x4<f32>,
    id: u32,
};
@group(1) @binding(0)
var<storage,read> s_instance_infos: array<InstanceInfo>;
$$ endif

fn get_sign_of_det_of_4x4(m: mat4x4<f32>) -> f32 {
    // We know/assume that the matrix is a homogeneous matrix,
    // so that only the 3x3 region is relevant for the determinant,
    // which is faster to calculate than the det of the 4x4.
    let m3 = mat3x3<f32>(m[0].xyz, m[1].xyz, m[2].xyz);
    return sign(determinant(m3));
}

fn dist_pt_line(x1: f32, y1: f32, x2: f32, y2: f32, x3: f32, y3: f32) -> f32 {
    // Distance of pt (x3,y3) to line with coords(x1,y1) (x2,y2)
    return abs((x2 - x1) * (y1 - y3) - (x1 - x3) * (y2 - y1)) / sqrt((x2-x1)*(x2-x1) + (y2-y1)*(y2-y1));
}

$$ if use_morph_targets
fn get_morph( tex: texture_2d_array<f32>, vertex_index: u32, stride: u32, width: u32, morph_index: u32 , offset: u32) -> vec4<f32> {
    let texel_index = vertex_index * stride + offset;
    let y = texel_index / width;
    let x = texel_index - y * width;
    let morph_uv = vec2<u32>( x, y );
    return textureLoad( tex, morph_uv, morph_index, 0 );
}
struct MorphTargetInfluence {
    //@size(16) influence: f32, -> Does not work on Metal, likely an upstream bug in Naga
    influence: f32,
    padding0: f32,
    padding1: f32,
    padding2: f32,
};
@group(1) @binding(1)
var<uniform> u_morph_target_influences: array<MorphTargetInfluence, {{influences_buffer_size}}>;

$$ endif

@vertex
fn vs_main(in: VertexInput) -> Varyings {

    // Get world transform
    $$ if instanced
        let instance_info = s_instance_infos[in.instance_index];
        let world_transform = u_wobject.world_transform * instance_info.transform;
    $$ else
        let world_transform = u_wobject.world_transform;
    $$ endif

    // Select what face we're at
    let index = i32(in.vertex_index);
    let face_index = index / {{indexer}};
    var sub_index = index % {{indexer}};
    var face_sub_index = 0;

    // for quads assuming the vertices are oriented, the triangles are 0 1 2 and 0 2 3
    $$ if indexer == 6
        var quad_map = array<i32,6>(0, 1, 2, 0, 2, 3);
        //face_sub_index returns 0 or 1 in picking for quads. So the triangle of the quad can be identified.
        var face_map = array<i32,6>(0, 0, 0, 1, 1, 1);
        face_sub_index = face_map[sub_index];
        sub_index = quad_map[sub_index];
    $$ endif

    // If a transform has an uneven number of negative scales, the 3 vertices
    // that make up the face are such that the GPU will mix up front and back
    // faces, producing an incorrect is_front. We can detect this from the
    // sign of the determinant, and reorder the faces to fix it. Note that
    // the projection_transform is not included here, because it cannot be
    // set with the public API and we assume that it does not include a flip.
    let winding_world = get_sign_of_det_of_4x4(world_transform);
    let winding_cam = get_sign_of_det_of_4x4(u_stdinfo.cam_transform);
    let must_flip_sub_index = winding_world * winding_cam < 0.0;
    // If necessary, and the sub_index is even, e.g. 0 or 2, we flip it to the other.
    // Flipping 0 and 2, because they are present in both triangles of a quad.
    if (must_flip_sub_index && sub_index % 2 == 0) {
        sub_index = select(0, 2, sub_index == 0);
    }

    // Sample
    let vii = load_s_indices(face_index);
    let i0 = i32(vii[sub_index]);

    // Get raw vertex position and normal
    var raw_pos = load_s_positions(i0);
    var raw_normal = load_s_normals(i0);

    $$ if use_tangent is defined
        let raw_tangent = load_s_tangents(i0);
        let object_tangent = raw_tangent.xyz;
    $$ endif

    // morph targets
    $$ if use_morph_targets
        let base_influence = u_morph_target_influences[{{influences_buffer_size-1}}];
        let stride = u32({{morph_targets_stride}});
        let width = u32({{morph_targets_texture_width}});

        raw_pos = raw_pos * base_influence.influence;
        if stride == 2 { // has normals
            raw_normal = raw_normal * base_influence.influence;
        }
        for (var i = 0; i < {{morph_targets_count}}; i = i + 1) {
            let position_morph = get_morph(t_morph_targets, u32(i0), stride, width, u32(i), u32(0));
            raw_pos += position_morph.xyz * u_morph_target_influences[i].influence;
            if stride == 2 { // has normals
                let normal_morph = get_morph(t_morph_targets, u32(i0), stride, width, u32(i), u32(1));
                raw_normal += normal_morph.xyz * u_morph_target_influences[i].influence;
            }

        }

    $$ endif

    // skinning
    $$ if use_skinning
        let skin_index = load_s_skin_indices(i0);
        let skin_weight = load_s_skin_weights(i0);
        let bind_matrix = u_wobject.bind_matrix;
        let bind_matrix_inv = u_wobject.bind_matrix_inv;

        let bone_mat_x = u_bone_matrices[skin_index.x].bone_matrices;
        let bone_mat_y = u_bone_matrices[skin_index.y].bone_matrices;
        let bone_mat_z = u_bone_matrices[skin_index.z].bone_matrices;
        let bone_mat_w = u_bone_matrices[skin_index.w].bone_matrices;

        // Calculate the skinned position and normal

        var skin_matrix = mat4x4<f32>();
        skin_matrix += skin_weight.x * bone_mat_x;
        skin_matrix += skin_weight.y * bone_mat_y;
        skin_matrix += skin_weight.z * bone_mat_z;
        skin_matrix += skin_weight.w * bone_mat_w;
        skin_matrix = bind_matrix_inv * skin_matrix * bind_matrix;

        raw_pos = (skin_matrix * vec4<f32>(raw_pos, 1.0)).xyz;
        raw_normal = (skin_matrix * vec4<f32>(raw_normal, 0.0)).xyz;

        $$ if use_tangent is defined
            object_tangent = (skin_matrix * vec4f(object_tangent, 0.0)).xyz;
        $$ endif

    $$ endif


    // Get vertex position

    let world_pos = world_transform * vec4<f32>(raw_pos, 1.0);
    var ndc_pos = u_stdinfo.projection_transform * u_stdinfo.cam_transform * world_pos;

    // For the wireframe we also need the ndc_pos of the other vertices of this face
    $$ if wireframe
        $$ for i in ((1, 2, 3) if indexer == 3 else (1, 2, 3, 4))
            let raw_pos{{ i }} = load_s_positions(i32(vii[{{ i - 1 }}]));
            let world_pos{{ i }} = world_transform * vec4<f32>(raw_pos{{ i }}, 1.0);
            let ndc_pos{{ i }} = u_stdinfo.projection_transform * u_stdinfo.cam_transform * world_pos{{ i }};
        $$ endfor
        let depth_offset = -0.0001;  // to put the mesh slice atop a mesh
        ndc_pos.z = ndc_pos.z + depth_offset;
    $$ endif

    // Prepare output
    var varyings: Varyings;

    // Set position
    varyings.world_pos = vec3<f32>(world_pos.xyz / world_pos.w);
    varyings.position = vec4<f32>(ndc_pos.xyz, ndc_pos.w);

    // per-vertex or per-face coloring
    $$ if use_vertex_color
        $$ if color_mode == 'face'
            let color_index = face_index;
        $$ else
            let color_index = i0;
        $$ endif
        $$ if color_buffer_channels == 1
            let cvalue = load_s_colors(color_index);
            varyings.color = vec4<f32>(cvalue, cvalue, cvalue, 1.0);
        $$ elif color_buffer_channels == 2
            let cvalue = load_s_colors(color_index);
            varyings.color = vec4<f32>(cvalue.r, cvalue.r, cvalue.r, cvalue.g);
        $$ elif color_buffer_channels == 3
            varyings.color = vec4<f32>(load_s_colors(color_index), 1.0);
        $$ elif color_buffer_channels == 4
            varyings.color = vec4<f32>(load_s_colors(color_index));
        $$ endif
    $$ endif

    // How to index into tex-coords
    $$ if color_mode == 'face_map'
    let tex_coord_index = face_index;
    $$ else
    let tex_coord_index = i0;
    $$ endif


    // used_uv
    $$ for uv, ndim in used_uv.items()
    $$ if ndim == 1
    varyings.texcoord{{uv or ""}} = f32(load_s_texcoords{{uv or ""}}(tex_coord_index));
    $$ elif ndim == 2
    varyings.texcoord{{uv or ""}} = vec2<f32>(load_s_texcoords{{uv or ""}}(tex_coord_index));
    $$ elif ndim == 3
    varyings.texcoord{{uv or ""}} = vec3<f32>(load_s_texcoords{{uv or ""}}(tex_coord_index));
    $$ endif
    $$ endfor



    // Set the normal
    // Transform the normal to world space
    // Note that the world transform matrix cannot be directly applied to the normal

    $$ if instanced
        // this is in lieu of a per-instance normal-matrix
        // shear transforms in the instance matrix are not supported
        let im = mat3x3f( instance_info.transform[0].xyz, instance_info.transform[1].xyz, instance_info.transform[2].xyz );
        raw_normal /= vec3f(dot(im[0], im[0]), dot(im[1], im[1]), dot(im[2], im[2]));
        raw_normal = im * raw_normal;

        $$ if use_tangent is defined
            object_tangent = im * object_tangent;
        $$ endif
    $$ endif

    let normal_matrix = transpose(u_wobject.world_transform_inv);
    let world_normal = normalize((normal_matrix * vec4<f32>(raw_normal, 0.0)).xyz);

    $$ if use_tangent is defined
        let v_tangent = normalize(( world_transform * vec4f(object_tangent, 0.0) ).xyz);
        let v_bitangent = normalize(cross(world_normal, v_tangent) * raw_tangent.w);
        varyings.v_tangent = vec3<f32>(v_tangent);
        varyings.v_bitangent = vec3<f32>(v_bitangent);
    $$ endif

    varyings.normal = vec3<f32>(world_normal);
    varyings.geometry_normal = vec3<f32>(raw_normal);
    varyings.winding_cam = f32(winding_cam);

    // Set wireframe barycentric-like coordinates
    $$ if wireframe
        $$ if indexer == 3
            $$ for i in (1, 2, 3)
                let p{{ i }} = (ndc_pos{{ i }}.xy / ndc_pos{{ i }}.w) * u_stdinfo.logical_size * 0.5;
            $$ endfor
            let dist1 = dist_pt_line(p2.x,p2.y,p3.x,p3.y,p1.x,p1.y);
            let dist2 = dist_pt_line(p1.x,p1.y,p3.x,p3.y,p2.x,p2.y);
            let dist3 = dist_pt_line(p1.x,p1.y,p2.x,p2.y,p3.x,p3.y);
            var arr_wireframe_coords = array<vec3<f32>, 3>(
                vec3<f32>(dist1, 0.0, 0.0), vec3<f32>(0.0, dist2, 0.0), vec3<f32>(0.0, 0.0, dist3)
            );
            varyings.wireframe_coords = vec3<f32>(arr_wireframe_coords[sub_index]);  // in logical pixels
        $$ elif indexer == 6
            $$ for i in (1, 2, 3, 4)
                let p{{ i }} = (ndc_pos{{ i }}.xy / ndc_pos{{ i }}.w) * u_stdinfo.logical_size * 0.5;
            $$ endfor
            //dist of vertex 1 to segment 23
            let dist1_23 = dist_pt_line(p2.x,p2.y,p3.x,p3.y,p1.x,p1.y);
            //dist of vertex 1 to segment 34
            let dist1_34 = dist_pt_line(p3.x,p3.y,p4.x,p4.y,p1.x,p1.y);

            //dist of vertex 2 to segment 34
            let dist2_34 = dist_pt_line(p3.x,p3.y,p4.x,p4.y,p2.x,p2.y);
            //dist of vertex 2 to segment 14
            let dist2_14 = dist_pt_line(p1.x,p1.y,p4.x,p4.y,p2.x,p2.y);

            //dist of vertex 3 to segment 12
            let dist3_12 = dist_pt_line(p1.x,p1.y,p2.x,p2.y,p3.x,p3.y);
            //dist of vertex 3 to segment 14
            let dist3_14 = dist_pt_line(p1.x,p1.y,p4.x,p4.y,p3.x,p3.y);

            //dist of vertex 4 to segment 12
            let dist4_12 = dist_pt_line(p2.x,p2.y,p1.x,p1.y,p4.x,p4.y);
            //dist of vertex 4 to segment 23
            let dist4_23 = dist_pt_line(p2.x,p2.y,p3.x,p3.y,p4.x,p4.y);

            //segments 12 23 34 41
            var arr_wireframe_coords = array<vec4<f32>, 4>(
                vec4<f32>( 0.0, dist1_23,dist1_34, 0.0),
                vec4<f32>(0.0, 0.0, dist2_34, dist2_14),
                vec4<f32>( dist3_12 ,0.0, 0.0, dist3_14),
                vec4<f32>( dist4_12,dist4_23, 0.0, 0.0)
                );
            varyings.wireframe_coords = vec4<f32>(arr_wireframe_coords[sub_index]);  // in logical pixels
        $$ endif
    $$ endif

    // Set varyings for picking. We store the face_index, and 3 weights
    // that indicate how close the fragment is to each vertex (barycentric
    // coordinates). This allows the selection of the nearest vertex or edge.
    $$ if instanced
        let pick_id = instance_info.id;
    $$ else
        let pick_id = u_wobject.id;
    $$ endif

    varyings.pick_id = u32(pick_id);
    $$ if indexer == 3
    varyings.pick_idx = u32(face_index);
    $$ else
    varyings.pick_idx = u32(face_index * 2 + face_sub_index);
    $$ endif

    var arr_pick_coords = array<vec3<f32>, 4>(vec3<f32>(1.0, 0.0, 0.0),
                                                vec3<f32>(0.0, 1.0, 0.0),
                                                vec3<f32>(0.0, 0.0, 1.0),
                                                vec3<f32>(0.0, 1.0, 0.0),  // the 2nd triangle in a quad
                                                );
    varyings.pick_coords = vec3<f32>(arr_pick_coords[sub_index]);

    return varyings;
}


struct ReflectedLight {
    direct_diffuse: vec3<f32>,
    direct_specular: vec3<f32>,
    indirect_diffuse: vec3<f32>,
    indirect_specular: vec3<f32>,
};

@fragment
fn fs_main(varyings: Varyings, @builtin(front_facing) is_front: bool) -> FragmentOutput {
    // clipping planes
    {$ include 'pygfx.clipping_planes.wgsl' $}

    // Get the surface normal from the geometry.
    // This is the unflipped normal, because thet NormalMaterial needs that.
    var surface_normal = normalize(vec3<f32>(varyings.normal));
    $$ if flat_shading
        let u = dpdx(varyings.world_pos);
        let v = dpdy(varyings.world_pos);
        surface_normal = normalize(cross(u, v));
        // Because this normal is derived from the world_pos, it has been corrected
        // for some of the winding, but not all. We apply the below steps to
        // bring it in the same state as the regular (non-flat) shading.
        surface_normal = select(-surface_normal, surface_normal, varyings.winding_cam < 0.0);
        surface_normal = select(-surface_normal, surface_normal, is_front);
    $$ endif


    $$ if color_mode == 'normal'
        var diffuse_color = vec4<f32>((normalize(surface_normal) * 0.5 + 0.5), 1.0);
    $$ else
        // to support color modes
        var diffuse_color = u_material.color;
        $$ if colorspace == 'srgb'
            diffuse_color = vec4f(srgb2physical(diffuse_color.rgb), diffuse_color.a);
        $$ endif

        $$ if color_mode != 'uniform'
            $$ if use_map
                $$ if use_colormap is defined
                    // special case for 'generic' colormap
                    var diffuse_map = sample_colormap(varyings.texcoord);
                $$ else
                    var diffuse_map = textureSample(t_map, s_map, varyings.texcoord{{map_uv or ''}});
                $$ endif

                $$ if colorspace == 'srgb'
                    diffuse_map = vec4f(srgb2physical(diffuse_map.rgb), diffuse_map.a);
                $$ endif

                $$ if color_mode == 'vertex_map' or color_mode == 'face_map'
                    diffuse_color = diffuse_map;
                $$ else
                    // default mode
                    diffuse_color *= diffuse_map;
                $$ endif
            $$ endif

            $$ if use_vertex_color
                // The vertex color should already in physical space
                $$ if color_mode == 'vertex' or color_mode == 'face'
                    diffuse_color = varyings.color;
                $$ else
                    // default mode
                    diffuse_color *= varyings.color;
                $$ endif
            $$ endif

        // uniform
        $$ endif


    $$ endif
    // Apply opacity
    diffuse_color.a = diffuse_color.a * u_material.opacity;

    let physical_albeido = diffuse_color.rgb;

    // Get normal used to calculate lighting or reflection
    $$ if lighting or use_env_map is defined
        // Get view direction
        let view = select(
            normalize(u_stdinfo.cam_transform_inv[3].xyz - varyings.world_pos),
            ( u_stdinfo.cam_transform_inv * vec4<f32>(0.0, 0.0, 1.0, 0.0) ).xyz,
            is_orthographic()
        );
        // Get normal used to calculate lighting
        surface_normal = select(-surface_normal, surface_normal, is_front);
        var normal = surface_normal;
        let face_direction = f32(is_front) * 2.0 - 1.0;

        $$ if use_normal_map is defined or USE_ANISOTROPY is defined
            $$ if use_tangent is defined
                var tbn = mat3x3f(varyings.v_tangent, varyings.v_bitangent, surface_normal);
            $$ else
                var tbn = getTangentFrame(view, normal, varyings.texcoord{{normal_map_uv or ''}} );
            $$ endif

            tbn[0] = tbn[0] * face_direction;
            tbn[1] = tbn[1] * face_direction;
        $$ endif

        $$ if use_normal_map is defined
            let normal_map = textureSample( t_normal_map, s_normal_map, varyings.texcoord{{normal_map_uv or ''}} ) * 2.0 - 1.0;
            let map_n = vec3f(normal_map.xy * u_material.normal_scale, normal_map.z);
            normal = normalize(tbn * map_n);
        $$ endif

        $$ if USE_CLEARCOAT is defined
            var clearcoat_normal = surface_normal;
            $$ if use_clearcoat_normal_map is defined
                $$ if use_tangent is defined
                    var tbn_cc = mat3x3f(varyings.v_tangent, varyings.v_bitangent, surface_normal);
                $$ else
                    var tbn_cc = getTangentFrame( view, clearcoat_normal, varyings.texcoord{{clearcoat_normal_map_uv or ''}} );
                $$ endif

                tbn_cc[0] = tbn_cc[0] * face_direction;
                tbn_cc[1] = tbn_cc[1] * face_direction;

                var clearcoat_normal_map = textureSample( t_clearcoat_normal_map, s_clearcoat_normal_map, varyings.texcoord{{clearcoat_normal_map_uv or ''}} ) * 2.0 - 1.0;
                let clearcoat_map_n = vec3f(clearcoat_normal_map.xy * u_material.clearcoat_normal_scale, clearcoat_normal_map.z);
                clearcoat_normal = normalize(tbn_cc * clearcoat_map_n);
            $$ endif
        $$ endif
    $$ endif

    $$ if use_specular_map is defined
        let specular_map = textureSample( t_specular_map, s_specular_map, varyings.texcoord{{specular_map_uv or ''}} );
        let specular_strength = specular_map.r;
    $$ else
        let specular_strength = 1.0;
    $$ endif

    // Init the reflected light. Defines diffuse and specular, both direct and indirect
    var reflected_light: ReflectedLight = ReflectedLight(vec3<f32>(0.0), vec3<f32>(0.0), vec3<f32>(0.0), vec3<f32>(0.0));

    // Lighting
    $$ if lighting
        var geometry: GeometricContext;
        geometry.position = varyings.world_pos;
        geometry.normal = normal;
        geometry.view_dir = view;

        $$ if USE_CLEARCOAT is defined
            geometry.clearcoat_normal = clearcoat_normal;
        $$ endif

        $$ if lighting == 'phong'
            {$ include 'pygfx.light_phong_fragment.wgsl' $}
        $$ elif lighting == 'pbr'
            {$ include 'pygfx.light_pbr_fragment.wgsl' $}
        $$ elif lighting == 'toon'
            {$ include 'pygfx.light_toon_fragment.wgsl' $}
        $$ endif

        // Do the math

        // Punctual light
        {$ include 'pygfx.light_punctual.wgsl' $}

        // Indirect Diffuse Light
        let ambient_color = u_ambient_light.color.rgb;  // the one exception that is already physical
        var irradiance = getAmbientLightIrradiance( ambient_color );
        // Light map (pre-baked lighting)
        $$ if use_light_map is defined
            let light_map_color = srgb2physical( textureSample( t_light_map, s_light_map, varyings.texcoord{{light_map_uv or ''}} ).rgb );
            irradiance += light_map_color * u_material.light_map_intensity;
        $$ endif

        // Process irradiance
        RE_IndirectDiffuse( irradiance, geometry, material, &reflected_light );

        // Indirect Specular Light
        // IBL (srgb2physical and intensity is handled in the getter functions)
        $$ if USE_IBL is defined

            $$ if USE_ANISOTROPY is defined
                let ibl_radiance = getIBLAnisotropyRadiance( view, normal, material.roughness, material.anisotropy_b, material.anisotropy );
            $$ else
                let ibl_radiance = getIBLRadiance( view, normal, material.roughness);
            $$ endif

            var clearcoat_ibl_radiance = vec3<f32>(0.0);
            $$ if USE_CLEARCOAT is defined
                clearcoat_ibl_radiance += getIBLRadiance( view, clearcoat_normal, material.clearcoat_roughness );
            $$ endif

            let ibl_irradiance = getIBLIrradiance( geometry.normal );
            RE_IndirectSpecular(ibl_radiance, ibl_irradiance, clearcoat_ibl_radiance, geometry, material, &reflected_light);
        $$ endif

    $$ else
        // for basic material
        // Light map (pre-baked lighting)
        $$ if use_light_map is defined
            let light_map_color = srgb2physical( textureSample( t_light_map, s_light_map, varyings.texcoord{{light_map_uv or ''}} ).rgb );
            reflected_light.indirect_diffuse += light_map_color * u_material.light_map_intensity * RECIPROCAL_PI;
        $$ else
            reflected_light.indirect_diffuse += vec3<f32>(1.0);
        $$ endif

        reflected_light.indirect_diffuse *= physical_albeido;
    $$ endif

    // Ambient occlusion
    $$ if use_ao_map is defined
        let ao_map_intensity = u_material.ao_map_intensity;
        let ambient_occlusion = ( textureSample( t_ao_map, s_ao_map, varyings.texcoord{{ao_map_uv or ''}} ).r - 1.0 ) * ao_map_intensity + 1.0;

        reflected_light.indirect_diffuse *= ambient_occlusion;

        $$ if USE_CLEARCOAT is defined
            clearcoat_specular_indirect *= ambient_occlusion;
        $$ endif

        $$ if USE_SHEEN is defined
            sheen_specular_indirect *= ambient_occlusion;
        $$ endif

        $$ if lighting == 'pbr' and USE_IBL is defined
            let dot_nv = saturate( dot( geometry.normal, geometry.view_dir ) );
            reflected_light.indirect_specular *= computeSpecularOcclusion( dot_nv, ambient_occlusion, material.roughness );
        $$ endif
    $$ endif

    // Combine direct and indirect light
    var physical_color = reflected_light.direct_diffuse + reflected_light.direct_specular + reflected_light.indirect_diffuse + reflected_light.indirect_specular;

    // Add emissive color
    // Now for phong、pbr and toon lighting
    $$ if lighting
        var emissive_color = srgb2physical(u_material.emissive_color.rgb) * u_material.emissive_intensity;
        $$ if use_emissive_map is defined
        emissive_color *= srgb2physical(textureSample(t_emissive_map, s_emissive_map, varyings.texcoord{{emissive_map_uv or ''}}).rgb);
        $$ endif
        physical_color += emissive_color;
    $$ endif

    $$ if USE_SHEEN is defined
        // Sheen energy compensation approximation calculation can be found at the end of
        // https://drive.google.com/file/d/1T0D1VSyR4AllqIJTQAraEIzjlb5h4FKH/view?usp=sharing
        let sheen_energy_comp = 1.0 - 0.157 * max(material.sheen_color.r, max(material.sheen_color.g, material.sheen_color.b));
        physical_color = physical_color * sheen_energy_comp + (sheen_specular_direct + sheen_specular_indirect);
    $$ endif

    $$ if USE_CLEARCOAT is defined
        let dot_nv_cc = saturate(dot(clearcoat_normal, view));
        let fcc = F_Schlick( material.clearcoat_f0, material.clearcoat_f90, dot_nv_cc );
        physical_color = physical_color * (1.0 - material.clearcoat * fcc) + (clearcoat_specular_direct + clearcoat_specular_indirect) * material.clearcoat;
    $$ endif

    // Environment mapping
    $$ if use_env_map is defined
        let reflectivity = u_material.reflectivity;
        $$ if env_mapping_mode == "CUBE-REFLECTION"
            var reflectVec = reflect( -view, normal );
        $$ elif env_mapping_mode == "CUBE-REFRACTION"
            var reflectVec = refract( -view, normal, u_material.refraction_ratio );
        $$ endif
        var env_color_srgb = textureSample( t_env_map, s_env_map, vec3<f32>( -reflectVec.x, reflectVec.yz) );
        let env_color = srgb2physical(env_color_srgb.rgb); // TODO: maybe already in linear-space
        $$ if env_combine_mode == 'MULTIPLY'
            physical_color = mix(physical_color, physical_color * env_color.xyz, specular_strength * reflectivity);
        $$ elif env_combine_mode == 'MIX'
            physical_color = mix(physical_color, env_color.xyz, specular_strength * reflectivity);
        $$ elif env_combine_mode == 'ADD'
            physical_color = physical_color + env_color.xyz * specular_strength * reflectivity;
        $$ endif
    $$ endif

    $$ if wireframe
        $$ if indexer == 3
        let distance_from_edge = min(varyings.wireframe_coords.x, min(varyings.wireframe_coords.y, varyings.wireframe_coords.z));
        $$ else
        let distance_from_edge = min(varyings.wireframe_coords.x, min(varyings.wireframe_coords.y, min(varyings.wireframe_coords.z, varyings.wireframe_coords.a)));
        $$ endif
        if (distance_from_edge > 0.5 * u_material.wireframe) {
            discard;
        }
    $$ endif

    let out_color = vec4<f32>(physical_color, diffuse_color.a);

    var out: FragmentOutput;
    out.color = out_color;

    $$ if write_pick
    // The wobject-id must be 20 bits. In total it must not exceed 64 bits.
    out.pick = (
        pick_pack(varyings.pick_id, 20) +
        pick_pack(varyings.pick_idx, 26) +
        pick_pack(u32(varyings.pick_coords.x * 63.0), 6) +
        pick_pack(u32(varyings.pick_coords.y * 63.0), 6) +
        pick_pack(u32(varyings.pick_coords.z * 63.0), 6)
    );
    $$ endif

    return out;
}
