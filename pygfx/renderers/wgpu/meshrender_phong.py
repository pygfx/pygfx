import wgpu  # only for flags/enums

from . import register_wgpu_render_function
from ._shadercomposer import Binding, WorldObjectShader
from .pointsrender import handle_colormap
from ...objects import Mesh, InstancedMesh
from ...materials import MeshPhongMaterial
from ...resources import Buffer
from ...utils import normals_from_vertices
from .shaderlibs import (
    mesh_vertex_shader,
    lights,
    bsdfs,
    bsdfs_blinn_phong,
    blinn_phong,
    shadow,
)


@register_wgpu_render_function(Mesh, MeshPhongMaterial)
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
    shader = MeshPhongShader(
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
    elif material.map is not None:
        shader["color_mode"] = "map"
        map_bindings = handle_colormap(geometry, material, shader)

        # TODO: this is a hack, we don't need the texcoords buffer
        # Code refactoring required
        map_bindings = map_bindings[:-1]
        bindings.extend(map_bindings)

        vertex_buffers.append(geometry.texcoords)

        colormap_dim = shader["colormap_dim"][:-1]
        atype = "f32" if colormap_dim == "1" else f"vec{colormap_dim}<f32>"
        shader["vertex_attributes"].append(("texcoord", atype))
        shader["has_uv"] = True

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


class MeshPhongShader(WorldObjectShader):
    def get_code(self):
        return (
            self.get_definitions()
            + self.common_functions()
            + mesh_vertex_shader
            + lights
            + bsdfs
            + bsdfs_blinn_phong
            + blinn_phong
            + shadow
            + self.fragment_shader()
        )

    def fragment_shader(self):
        return """

        @stage(fragment)
        fn fs_main(varyings: Varyings, @builtin(front_facing) is_front: bool) -> FragmentOutput {
            $$ if color_mode == 'vertex'
                let color_value = varyings.color;
                let albeido = color_value.rgb;
            $$ elif color_mode == 'map'
                let color_value = sample_colormap(varyings.texcoord);
                let albeido = color_value.rgb;  // no more colormap
            $$ elif color_mode == 'normal'
                let albeido = normalize(varyings.normal.xyz) * 0.5 + 0.5;
                let color_value = vec4<f32>(albeido, 1.0);
            $$ else
                let color_value = u_material.color;
                let albeido = color_value.rgb;
            $$ endif

            var material: BlinnPhongMaterial;

            material.diffuse_color = albeido;
            material.specular_color = u_material.specular_color.rgb;
            material.specular_shininess = u_material.shininess;

            material.specular_strength = 1.0;  //TODO: Use specular_map if exists

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

                    $$ if has_shadow
                    let shadow = get_cube_shadow(u_shadow_map_point_light, u_shadow_sampler, i, u_shadow_point_light[i].light_view_proj_matrix, geometry.position, light.direction, u_shadow_point_light[i].bias);
                    light.color *= shadow;
                    $$ endif

                    reflected_light = RE_Direct_BlinnPhong( light, geometry, material, reflected_light );

                    i += 1;
                }
            $$ endif

            $$ if num_spot_lights > 0
                i = 0;
                loop {
                    if (i >= {{ num_spot_lights }}) { break; }

                    let spot_light = u_spot_lights[i];

                    var light = getSpotLightInfo(spot_light, geometry);

                    $$ if has_shadow
                    let coords = u_shadow_spot_light[i].light_view_proj_matrix * vec4<f32>(geometry.position,1.0);
                    let shadow = get_shadow(u_shadow_map_spot_light, u_shadow_sampler, i, coords, u_shadow_spot_light[i].bias);
                    light.color *= shadow;
                    $$ endif

                    reflected_light = RE_Direct_BlinnPhong( light, geometry, material, reflected_light );

                    i += 1;
                }
            $$ endif

            $$ if num_dir_lights > 0
                i = 0;
                loop {
                    if (i >= {{ num_dir_lights }}) { break; }

                    let dir_light = u_directional_lights[i];

                    var light = getDirectionalLightInfo(dir_light, geometry);

                    $$ if has_shadow
                    let coords = u_shadow_dir_light[i].light_view_proj_matrix * vec4<f32>(geometry.position,1.0);
                    let shadow = get_shadow(u_shadow_map_dir_light, u_shadow_sampler, i, coords, u_shadow_dir_light[i].bias);
                    light.color *= shadow;
                    $$ endif

                    reflected_light = RE_Direct_BlinnPhong( light, geometry, material, reflected_light );

                    i += 1;
                }
            $$ endif


            let ambient_color = u_ambient_light.color.rgb;
            let irradiance = getAmbientLightIrradiance( ambient_color );
            reflected_light = RE_IndirectDiffuse_BlinnPhong( irradiance, geometry, material, reflected_light );

            let lit_color = reflected_light.direct_diffuse + reflected_light.direct_specular + reflected_light.indirect_diffuse + reflected_light.indirect_specular + u_material.emissive_color.rgb;

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
