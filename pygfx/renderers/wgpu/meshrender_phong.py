import wgpu  # only for flags/enums

from . import register_wgpu_render_function
from ._shadercomposer import Binding, WorldObjectShader
from .pointsrender import handle_colormap
from ...objects import Mesh, InstancedMesh
from ...materials import MeshPhongMaterial
from ...resources import Buffer
from ...utils import normals_from_vertices
from .shaderlibs import mesh_vertex_shader, lights, bsdfs, blinn_phong


@register_wgpu_render_function(Mesh, MeshPhongMaterial)
def mesh_renderer(render_info):
    """Render function capable of rendering meshes."""
    wobject = render_info.wobject
    geometry = wobject.geometry
    material = wobject.material  # noqa

    # Initialize
    topology = wgpu.PrimitiveTopology.triangle_list
    shader = MeshPhongShader(
        render_info,
        lighting="",
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
    vertex_buffers = {}
    index_buffer = None

    # Init bindings
    bindings = [
        Binding("u_stdinfo", "buffer/uniform", render_info.stdinfo_uniform),
        Binding("u_wobject", "buffer/uniform", wobject.uniform_buffer),
        Binding("u_material", "buffer/uniform", material.uniform_buffer),
        Binding("s_indices", "buffer/read_only_storage", geometry.indices, "VERTEX"),
        Binding(
            "s_positions", "buffer/read_only_storage", geometry.positions, "VERTEX"
        ),
        Binding("s_normals", "buffer/read_only_storage", normal_buffer, "VERTEX"),
    ]

    bindings1 = []  # non-auto-generated bindings

    # Per-vertex color, colormap, or a plane color?
    shader["color_mode"] = "uniform"
    if material.vertex_colors:
        shader["color_mode"] = "vertex"
        shader["vertex_color_channels"] = nchannels = geometry.colors.data.shape[1]
        if nchannels not in (1, 2, 3, 4):
            raise ValueError(f"Geometry.colors needs 1-4 columns, not {nchannels}")
        bindings.append(
            Binding("s_colors", "buffer/read_only_storage", geometry.colors, "VERTEX")
        )
    elif material.map is not None:
        shader["color_mode"] = "map"
        bindings.extend(handle_colormap(geometry, material, shader))

    # Lights states
    shader["num_dir_lights"] = len(render_info.state.directional_lights)
    shader["num_point_lights"] = len(render_info.state.point_lights)

    state_buffers = render_info.state.uniform_buffers
    ambient_lights_buffer = state_buffers["ambient_lights"]

    if ambient_lights_buffer:
        bindings.append(
            Binding(
                f"u_ambient_light",
                "buffer/uniform",
                ambient_lights_buffer,
                structname="AmbientLight",
            ),
        )

    directional_lights_buffer = state_buffers["directional_lights"]
    if directional_lights_buffer:
        bindings.append(
            Binding(
                f"u_directional_lights",
                "buffer/uniform",
                directional_lights_buffer,
                structname="DirectionalLight",
            ),
        )

    point_lights_buffer = state_buffers["point_lights"]
    if point_lights_buffer:
        bindings.append(
            Binding(
                f"u_point_lights",
                "buffer/uniform",
                point_lights_buffer,
                structname="PointLight",
            ),
        )

    # Instanced meshes have an extra storage buffer that we add manually
    n_instances = 1
    if isinstance(wobject, InstancedMesh):
        shader["instanced"] = True
        bindings1.append(
            Binding(
                "s_instance_infos",
                "buffer/read_only_storage",
                wobject.instance_infos,
                "VERTEX",
            )
        )
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
            + blinn_phong
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

            var i: i32 = 0;
            var material: BlinnPhongMaterial;

            material.diffuse_color = albeido;
            material.specular_color = u_material.specular_color.rgb;
            material.specular_shininess = u_material.shininess;

            material.specular_strength = 1.0;  //TODO: Use specular_map if exists

            var reflected_light: ReflectedLight = ReflectedLight(vec3<f32>(0.0), vec3<f32>(0.0), vec3<f32>(0.0), vec3<f32>(0.0));


            $$ if num_dir_lights > 0 or num_point_lights > 0
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

                // normal = select(-normal, normal, is_front);  // do we really need this?

            $$ endif

            $$ if num_dir_lights > 0
                loop {
                    if (i >= {{ num_dir_lights }}) { break; }

                    let dir_light = u_directional_lights[i];

                    var light: IncidentLight;

                    light.color = dir_light.color.rgb;
                    light.direction = -dir_light.direction.xyz;
                    light.visible = true;

                    reflected_light = RE_Direct_BlinnPhong( light, reflected_light, view_dir, normal, material );

                    i += 1;
                }
            $$ endif

            $$ if num_point_lights > 0
                i = 0;
                loop {
                    if (i >= {{ num_point_lights }}) { break; }

                    let point_light = u_point_lights[i];

                    var light: IncidentLight;

                    let i_vector = point_light.world_transform[3].xyz - varyings.world_pos;

                    light.direction = normalize(i_vector);
                    let light_distance = length(i_vector);

                    light.color = point_light.color.rgb;
                    light.color *= getDistanceAttenuation( light_distance, point_light.distance, point_light.decay );
                    light.visible = (light.color.r != 0.0) || (light.color.g != 0.0) || (light.color.b != 0.0);

                    reflected_light = RE_Direct_BlinnPhong( light, reflected_light, view_dir, normal, material );

                    i += 1;
                }
            $$ endif


            let ambient_color = u_ambient_light.color.rgb;
            let irradiance = getAmbientLightIrradiance( ambient_color );
            reflected_light = RE_IndirectDiffuse_BlinnPhong( irradiance, reflected_light, material );

            let lit_color = reflected_light.direct_diffuse + reflected_light.direct_specular + reflected_light.indirect_diffuse + reflected_light.indirect_specular + u_material.emissive_color.rgb;


            $$ if wireframe
                let distance_from_edge = min(varyings.wireframe_coords.x, min(varyings.wireframe_coords.y, varyings.wireframe_coords.z));
                if (distance_from_edge > 0.5 * u_material.wireframe) {
                    discard;
                }
            $$ endif

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
