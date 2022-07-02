import wgpu  # only for flags/enums
from . import register_wgpu_render_function
from ._shadercomposer import Binding, WorldObjectShader
from ...objects import Mesh, InstancedMesh
from ...materials import MeshStandardMaterial
from ...resources import Buffer
from ...utils import normals_from_vertices
from .shaderlibs import (
    mesh_vertex_shader,
    lights,
    bsdfs,
    bsdfs_physical,
    physical_lighting,
    shadow,
)


@register_wgpu_render_function(Mesh, MeshStandardMaterial)
def mesh_renderer(render_info):
    """Render function capable of rendering meshes."""
    wobject = render_info.wobject
    geometry = wobject.geometry
    material = wobject.material  # noqa

    # Initialize
    topology = (
        wgpu.PrimitiveTopology.line_list
        if material.wireframe
        else wgpu.PrimitiveTopology.triangle_list
    )
    shader = MeshStandardShader(
        render_info,
        colormap_format="f32",
        instanced=False,
        wireframe=material.wireframe,
        vertex_color_channels=0,
    )

    # We're assuming the presence of an index buffer for now
    assert getattr(geometry, "indices", None)
    n = geometry.indices.data.size

    # Normals. Usually it'd be given. If not, we'll calculate it from the vertices.
    if getattr(geometry, "normals", None) is not None:
        normal_buffer = geometry.normals
    else:
        normal_data = normals_from_vertices(
            geometry.positions.data, geometry.indices.data
        )
        normal_buffer = Buffer(normal_data)

    # We're using storage buffers for everything; no vertex nor index buffers.
    vertex_buffers = []
    index_buffer = geometry.indices
    shader["vertex_attributes"] = []

    vertex_buffers.append(geometry.positions)
    # TODO: auto get type from buffer
    shader["vertex_attributes"].append(("position", "vec3<f32>"))

    vertex_buffers.append(normal_buffer)
    shader["vertex_attributes"].append(("normal", "vec3<f32>"))

    # Init bindings
    bindings = [
        Binding("u_stdinfo", "buffer/uniform", render_info.stdinfo_uniform),
        Binding("u_wobject", "buffer/uniform", wobject.uniform_buffer),
        Binding("u_material", "buffer/uniform", material.uniform_buffer),
    ]

    bindings1 = []  # non-auto-generated bindings

    # Per-vertex color, colormap, or a plane color?
    shader["color_mode"] = "uniform"
    if material.vertex_colors:
        shader["color_mode"] = "vertex"
        shader["vertex_color_channels"] = nchannels = geometry.colors.data.shape[1]
        if nchannels not in (1, 2, 3, 4):
            raise ValueError(f"Geometry.colors needs 1-4 columns, not {nchannels}")

        vertex_buffers.append(geometry.colors)
        shader["vertex_attributes"].append(("color", f"vec{nchannels}<f32>"))

    # We need uv to use the pbr maps, so if uv not exist, ignore all maps
    if geometry.texcoords is not None:
        shader["has_uv"] = True

        vertex_buffers.append(geometry.texcoords)
        shader["vertex_attributes"].append(("texcoord", "vec2<f32>"))

        if material.map is not None:
            shader["use_color_map"] = True
            bindings.append(
                Binding(f"s_color_map", "sampler/filtering", material.map, "FRAGMENT")
            )
            bindings.append(
                Binding(f"t_color_map", "texture/auto", material.map, "FRAGMENT")
            )

        if material.env_map is not None:
            shader["use_env_map"] = True
            bindings.append(
                Binding(f"s_env_map", "sampler/filtering", material.env_map, "FRAGMENT")
            )
            bindings.append(
                Binding(f"t_env_map", "texture/auto", material.env_map, "FRAGMENT")
            )

        if material.normal_map is not None:
            shader["use_normal_map"] = True
            bindings.append(
                Binding(
                    f"s_normal_map",
                    "sampler/filtering",
                    material.normal_map,
                    "FRAGMENT",
                )
            )
            bindings.append(
                Binding(
                    f"t_normal_map", "texture/auto", material.normal_map, "FRAGMENT"
                )
            )

        if material.roughness_map is not None:
            shader["use_roughness_map"] = True
            bindings.append(
                Binding(
                    f"s_roughness_map",
                    "sampler/filtering",
                    material.roughness_map,
                    "FRAGMENT",
                )
            )
            bindings.append(
                Binding(
                    f"t_roughness_map",
                    "texture/auto",
                    material.roughness_map,
                    "FRAGMENT",
                )
            )

        if material.metalness_map is not None:
            shader["use_metalness_map"] = True
            bindings.append(
                Binding(
                    f"s_metalness_map",
                    "sampler/filtering",
                    material.metalness_map,
                    "FRAGMENT",
                )
            )
            bindings.append(
                Binding(
                    f"t_metalness_map",
                    "texture/auto",
                    material.metalness_map,
                    "FRAGMENT",
                )
            )

        if material.emissive_map is not None:
            shader["use_emissive_map"] = True
            bindings.append(
                Binding(
                    f"s_emissive_map",
                    "sampler/filtering",
                    material.emissive_map,
                    "FRAGMENT",
                )
            )
            bindings.append(
                Binding(
                    f"t_emissive_map", "texture/auto", material.emissive_map, "FRAGMENT"
                )
            )

        if material.ao_map is not None:
            shader["use_ao_map"] = True
            bindings.append(
                Binding(f"s_ao_map", "sampler/filtering", material.ao_map, "FRAGMENT")
            )
            bindings.append(
                Binding(f"t_ao_map", "texture/auto", material.ao_map, "FRAGMENT")
            )

    # Lights states
    shader["num_dir_lights"] = len(render_info.state.directional_lights)
    shader["num_point_lights"] = len(render_info.state.point_lights)
    shader["num_spot_lights"] = len(render_info.state.spot_lights)

    ambient_lights_buffer = render_info.state.ambient_lights_uniform_buffer
    if ambient_lights_buffer:
        bindings.append(
            Binding(
                f"u_ambient_light",
                "buffer/uniform",
                ambient_lights_buffer,
                structname="AmbientLight",
            ),
        )

    directional_lights_buffer = render_info.state.directional_lights_uniform_buffer
    if directional_lights_buffer:
        bindings.append(
            Binding(
                f"u_directional_lights",
                "buffer/uniform",
                directional_lights_buffer,
                structname="DirectionalLight",
            ),
        )

    point_lights_buffer = render_info.state.point_lights_uniform_buffer
    if point_lights_buffer:
        bindings.append(
            Binding(
                f"u_point_lights",
                "buffer/uniform",
                point_lights_buffer,
                structname="PointLight",
            ),
        )

    spot_lights_buffer = render_info.state.spot_lights_uniform_buffer
    if spot_lights_buffer:
        bindings.append(
            Binding(
                f"u_spot_lights",
                "buffer/uniform",
                spot_lights_buffer,
                structname="SpotLight",
            ),
        )

    shader["has_shadow"] = False
    if wobject.receive_shadow and (
        len(render_info.state.spot_lights)
        + len(render_info.state.point_lights)
        + len(render_info.state.directional_lights)
        > 0
    ):
        shader["has_shadow"] = True
        bindings.append(
            Binding(
                f"u_shadow_sampler",
                "shadow_sampler/comparison",
                render_info.state.shadow_sampler,
            )
        )

        if len(render_info.state.directional_lights) > 0:

            bindings.append(
                Binding(
                    f"u_shadow_map_dir_light",
                    "shadow_texture/2d-array",
                    render_info.state.directional_lights_shadow_texture.create_view(
                        dimension="2d-array"
                    ),
                )
            )

            bindings.append(
                Binding(
                    f"u_shadow_dir_light",
                    "buffer/uniform",
                    render_info.state.directional_shadows_uniform_buffer,
                )
            )

        if len(render_info.state.spot_lights) > 0:

            bindings.append(
                Binding(
                    f"u_shadow_map_spot_light",
                    "shadow_texture/2d-array",
                    render_info.state.spot_lights_shadow_texture.create_view(
                        dimension="2d-array"
                    ),
                )
            )

            bindings.append(
                Binding(
                    f"u_shadow_spot_light",
                    "buffer/uniform",
                    render_info.state.spot_shadows_uniform_buffer,
                )
            )

        if len(render_info.state.point_lights) > 0:

            bindings.append(
                Binding(
                    f"u_shadow_map_point_light",
                    "shadow_texture/cube-array",
                    render_info.state.point_lights_shadow_texture.create_view(
                        dimension="cube-array"
                    ),
                )
            )

            bindings.append(
                Binding(
                    f"u_shadow_point_light",
                    "buffer/uniform",
                    render_info.state.point_shadows_uniform_buffer,
                )
            )

    # Instanced meshes have an extra storage buffer that we add manually
    instance_buffer = None
    n_instances = 1
    if isinstance(wobject, InstancedMesh):
        shader["instanced"] = True
        instance_buffer = wobject.instance_infos
        # bindings1.append(
        #     Binding(
        #         "s_instance_infos",
        #         "buffer/read_only_storage",
        #         wobject.instance_infos,
        #         "VERTEX",
        #     )
        # )
        n_instances = wobject.instance_infos.nitems

    # Determine culling
    if material.side == "FRONT":
        cull_mode = wgpu.CullMode.back
    elif material.side == "BACK":
        cull_mode = wgpu.CullMode.front
    else:  # material.side == "BOTH"
        cull_mode = wgpu.CullMode.none

    # Let the shader generate code for our bindings
    for i, binding in enumerate(bindings):
        shader.define_binding(0, i, binding)

    # Determine in what render passes this objects must be rendered
    suggested_render_mask = 3
    if material.opacity < 1:
        suggested_render_mask = 2
    elif shader["color_mode"] == "vertex":
        if shader["vertex_color_channels"] in (1, 3):
            suggested_render_mask = 1
    elif shader["color_mode"] == "map":
        if shader["colormap_nchannels"] in (1, 3):
            suggested_render_mask = 1
    elif shader["color_mode"] == "normal":
        suggested_render_mask = 1
    elif shader["color_mode"] == "uniform":
        suggested_render_mask = 1 if material.color[3] >= 1 else 2
    else:
        raise RuntimeError(f"Unexpected color mode {shader['color_mode']}")

    # Put it together!
    return [
        {
            "suggested_render_mask": suggested_render_mask,
            "render_shader": shader,
            "primitive_topology": topology,
            "cull_mode": cull_mode,
            "indices": (range(n), range(n_instances)),
            "index_buffer": index_buffer,
            "vertex_buffers": vertex_buffers,
            "bindings0": bindings,
            "bindings1": bindings1,
            "instance_buffers": instance_buffer,
        }
    ]


class MeshStandardShader(WorldObjectShader):
    def get_code(self):
        return (
            self.get_definitions()
            + self.common_functions()
            + mesh_vertex_shader
            + lights
            + bsdfs
            + bsdfs_physical
            + physical_lighting
            + shadow
            + self.fragment_shader()
        )

    def fragment_shader(self):
        return """

        @stage(fragment)
        fn fs_main(varyings: Varyings, @builtin(front_facing) is_front: bool) -> FragmentOutput {
            $$ if color_mode == 'vertex'
                var color_value = varyings.color;
            $$ elif color_mode == 'normal'
                var color_value = vec4<f32>((normalize(varyings.normal.xyz) * 0.5 + 0.5), 1.0);
            $$ else
                var color_value = u_material.color;
            $$ endif

            $$ if use_color_map is defined
                color_value = color_value * textureSample( t_color_map, s_color_map, varyings.texcoord );
            $$ endif

            let albeido = color_value.rgb;

            var metalness_factor: f32 = u_material.metalness;

            $$ if use_metalness_map is defined
                let texel_metalness = textureSample( t_metalness_map, s_metalness_map, varyings.texcoord );
                metalness_factor *= texel_metalness.b;
            $$ endif

            var roughness_factor: f32 = u_material.roughness;

            $$ if use_roughness_map is defined
                let texel_roughness = textureSample( t_roughness_map, s_roughness_map, varyings.texcoord );
                roughness_factor *= texel_roughness.g;
            $$ endif


            let dxy = max( abs( dpdx( varyings.geometry_normal ) ), abs( dpdy( varyings.geometry_normal ) ) );
            let geometry_roughness = max( max( dxy.x, dxy.y ), dxy.z );
            var roughness = max( roughness_factor, 0.0525 );
            roughness += geometry_roughness;
            roughness = min( roughness, 1.0 );

            var material: PhysicalMaterial;
            material.diffuse_color = albeido * ( 1.0 - metalness_factor );
            material.specular_color = mix( vec3<f32>( 0.04 ), albeido.rgb, metalness_factor );
            material.roughness = roughness;
            material.specular_f90 = 1.0;

            var reflected_light: ReflectedLight = ReflectedLight(vec3<f32>(0.0), vec3<f32>(0.0), vec3<f32>(0.0), vec3<f32>(0.0));


            let view_dir = select(
                normalize(u_stdinfo.cam_transform_inv[3].xyz - varyings.world_pos),
                ( u_stdinfo.cam_transform_inv * vec4<f32>(0.0, 0.0, 1.0, 0.0) ).xyz,
                is_orthographic()
            );

            var normal = varyings.normal;

            if (u_material.flat_shading != 0 ) {
                let u = dpdx(varyings.world_pos);
                let v = dpdy(varyings.world_pos);
                normal = normalize(cross(u, v));
                normal = select(normal, -normal, (select(0, 1, is_front) + u_stdinfo.flipped_winding) == 1);  //?
            }

            normal = select(-normal, normal, is_front);

            $$ if use_normal_map is defined
                var normal_map = textureSample( t_normal_map, s_normal_map, varyings.texcoord );
                normal_map = normal_map * 2.0 - 1.0;
                let normal_map_scale = vec3<f32>( normal_map.xy * u_material.normal_scale, normal_map.z );   // normal_scale
                normal = perturbNormal2Arb(view_dir, normal, normal_map_scale, varyings.texcoord, is_front);
            $$ endif

            var geometry: GeometricContext;
            geometry.position = varyings.world_pos;
            geometry.normal = normal;
            geometry.view_dir = view_dir;


            //direct

            var i = 0;
            $$ if num_point_lights > 0
                loop {
                    if (i >= {{ num_point_lights }}) { break; }

                    let point_light = u_point_lights[i];

                    var light = getPointLightInfo(point_light, geometry);

                    if (! light.visible) { continue; }

                    $$ if has_shadow
                    let shadow = get_cube_shadow(u_shadow_map_point_light, u_shadow_sampler, i, u_shadow_point_light[i].light_view_proj_matrix, geometry.position, light.direction, u_shadow_point_light[i].bias);
                    light.color *= shadow;
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

                    $$ if has_shadow
                    let coords = u_shadow_spot_light[i].light_view_proj_matrix * vec4<f32>(geometry.position,1.0);
                    let shadow = get_shadow(u_shadow_map_spot_light, u_shadow_sampler, i, coords, u_shadow_spot_light[i].bias);
                    light.color *= shadow;
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

                    $$ if has_shadow
                    let coords = u_shadow_dir_light[i].light_view_proj_matrix * vec4<f32>(geometry.position,1.0);
                    let shadow = get_shadow(u_shadow_map_dir_light, u_shadow_sampler, i, coords, u_shadow_dir_light[i].bias);
                    light.color *= shadow;
                    $$ endif

                    reflected_light = RE_Direct_Physical( light, geometry, material, reflected_light );

                    continuing {
                        i += 1;
                    }
                }
            $$ endif

            // indirect diffuse

            let ambient_color = u_ambient_light.color.rgb;
            var irradiance = getAmbientLightIrradiance( ambient_color );

             // TODO: lightmap
            $$ if use_light_map is defined
            let lightMapIrradiance = ( textureSample( t_light_map, s_light_map, varyings.texcoord )).rgb * u_material.light_map_intensity;
            irradiance += lightMapIrradiance;
            $$ endif

            reflected_light = RE_IndirectDiffuse_Physical( irradiance, geometry, material, reflected_light );

            // TODO: IBL
            // indirect specular

            $$ if use_env_map is defined
            //material.roughness = 0.0;
            let mip_level_r = getMipLevel(u_material.env_map_max_mip_level, material.roughness);
            let radiance = getIBLRadiance( geometry.view_dir, geometry.normal, material.roughness, t_env_map, s_env_map, mip_level_r );
            let mip_level_i = getMipLevel(u_material.env_map_max_mip_level, 1.0);
            let iblIrradiance = getIBLIrradiance( geometry.normal, t_env_map, s_env_map, mip_level_i );
            reflected_light = RE_IndirectSpecular_Physical( radiance.rgb, iblIrradiance.rgb, geometry, material, reflected_light );
            $$ endif

            // ao

            $$ if use_ao_map is defined
            let ao_map_intensity = u_material.ao_map_intensity;
            let ambientOcclusion = ( textureSample( t_ao_map, s_ao_map, varyings.texcoord ).r - 1.0 ) * ao_map_intensity + 1.0;
            reflected_light = RE_AmbientOcclusion_Physical(ambientOcclusion, geometry, material, reflected_light);
            $$ endif


            var lit_color = reflected_light.direct_diffuse + reflected_light.direct_specular + reflected_light.indirect_diffuse + reflected_light.indirect_specular;
            var emissive_color = u_material.emissive_color.rgb * u_material.emissive_intensity;

            $$ if use_emissive_map is defined
            emissive_color *= textureSample( t_emissive_map, s_emissive_map, varyings.texcoord).rgb;
            $$ endif

            lit_color += emissive_color;

            let final_color = vec4<f32>(lit_color, color_value.a * u_material.opacity);

            // Wrap up

            apply_clipping_planes(varyings.world_pos);
            var out = get_fragment_output(varyings.position.z, final_color);

            $$ if write_pick
            // The wobject-id must be 20 bits. In total it must not exceed 64 bits.
            out.pick = (
                pick_pack(varyings.pick_id, 20) +
                pick_pack(varyings.pick_idx, 26) +
                pick_pack(u32(varyings.pick_coords.x * 64.0), 6) +
                pick_pack(u32(varyings.pick_coords.y * 64.0), 6) +
                pick_pack(u32(varyings.pick_coords.z * 64.0), 6)
            );
            $$ endif

            return out;
        }

        """
