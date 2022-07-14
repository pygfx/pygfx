import wgpu  # only for flags/enums

from . import register_wgpu_render_function, Binding
from .meshshader import MeshShader
from ...objects import Mesh
from ...materials import MeshPhongMaterial
from ...resources import Buffer
from ...utils import normals_from_vertices
from ._shaderlibs import (
    mesh_vertex_shader,
    lights,
    bsdfs,
    bsdfs_blinn_phong,
    blinn_phong_lighting,
    shadow,
)


@register_wgpu_render_function(Mesh, MeshPhongMaterial)
class MeshPhongShader(MeshShader):
    def __init__(self, wobject):
        super().__init__(wobject)

    def get_resources(self, wobject, shared):

        geometry = wobject.geometry
        material = wobject.material

        # indexbuffer
        # vertex_buffers
        # list of list of dicts

        # We're assuming the presence of an index buffer for now
        assert getattr(geometry, "indices", None)

        index_buffer = geometry.indices

        vertex_attributes = {}

        vertex_attributes["position"] = geometry.positions

        # Normals. Usually it'd be given. If not, we'll calculate it from the vertices.
        if getattr(geometry, "normals", None) is not None:
            normal_buffer = geometry.normals
        else:
            normal_data = normals_from_vertices(
                geometry.positions.data, geometry.indices.data
            )
            normal_buffer = Buffer(normal_data)

        vertex_attributes["normal"] = normal_buffer

        # Init bindings
        bindings = [
            Binding("u_stdinfo", "buffer/uniform", shared.uniform_buffer),
            Binding("u_wobject", "buffer/uniform", wobject.uniform_buffer),
            Binding("u_material", "buffer/uniform", material.uniform_buffer),
        ]

        if material.vertex_colors:
            self["use_vertex_colors"] = True
            vertex_attributes["color"] = geometry.colors

        # We need uv to use the maps, so if uv not exist, ignore all maps
        if geometry.texcoords is not None:
            self["has_uv"] = True

            vertex_attributes["texcoord"] = geometry.texcoords

            # for legacy meshphong compatibility, uv_size can be 1, 2 or 3 for different map channel numbers

            if len(geometry.texcoords.data.shape) == 1:
                uv_size = 1
            else:
                uv_size = geometry.texcoords.data.shape[1]

            self["uv_size"] = uv_size

            # TODO: Our colormap supports various formats, not just float32.
            # And it is transmitted to GPU according to the original data format.
            # we should automatically convert it to float32 format before transfer to GPU
            # Also see MeshStandardMaterial shader
            if material.map is not None and not material.vertex_colors:
                self["use_color_map"] = True

                bindings.append(
                    Binding(
                        f"s_color_map", "sampler/filtering", material.map, "FRAGMENT"
                    )
                )
                bindings.append(
                    Binding(f"t_color_map", "texture/auto", material.map, "FRAGMENT")
                )

        self["use_light"] = True

        # TODO: Whether it should be defined as uniform so that it can be changed without recompiling?
        self["receive_shadow"] = wobject.receive_shadow

        # Define shader code for vertex buffer

        self.define_vertex_buffer(vertex_attributes, instanced=self["instanced"])

        # Define shader code for binding
        bindings = {i: binding for i, binding in enumerate(bindings)}
        self.define_bindings(0, bindings)

        return {
            "index_buffer": index_buffer,
            "vertex_buffers": list(vertex_attributes.values()),
            "instance_buffer": wobject.instance_infos if self["instanced"] else None,
            "bindings": {0: bindings},
        }

    def get_pipeline_info(self, wobject, shared):
        material = wobject.material

        topology = (
            wgpu.PrimitiveTopology.line_list
            if material.wireframe
            else wgpu.PrimitiveTopology.triangle_list
        )

        if material.side == "FRONT":
            cull_mode = wgpu.CullMode.back
        elif material.side == "BACK":
            cull_mode = wgpu.CullMode.front
        else:  # material.side == "BOTH"
            cull_mode = wgpu.CullMode.none

        return {
            "primitive_topology": topology,
            "cull_mode": cull_mode,
        }

    def get_code(self):
        return (
            self.code_definitions()
            + self.code_common()
            + mesh_vertex_shader
            + lights
            + bsdfs
            + bsdfs_blinn_phong
            + blinn_phong_lighting
            + shadow
            + self.code_fragment()
        )

    def code_common(self):
        """Get the WGSL functions builtin by PyGfx."""

        # Just a placeholder
        blending_code = """
        let alpha_compare_epsilon : f32 = 1e-6;
        {{ blending_code }}
        """

        return (
            self._code_is_orthographic()
            + self._code_lighting()
            + self._code_clipping_planes()
            + self._code_picking()
            + self._code_misc()
            + blending_code
        )

    def code_fragment(self):
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
            normal = select(-normal, normal, is_front);  // do we really need this?
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
