import wgpu  # only for flags/enums

from . import (
    register_wgpu_render_function,
    WorldObjectShader,
    Binding,
    RenderMask,
    shaderlib,
)
from ._utils import to_texture_format, GfxSampler, GfxTextureView
from ...objects import Mesh, InstancedMesh
from ...materials import (
    MeshBasicMaterial,
    MeshPhongMaterial,
    MeshNormalMaterial,
    MeshNormalLinesMaterial,
    MeshSliceMaterial,
    MeshStandardMaterial,
)
from ...resources import Buffer, Texture
from ...utils import normals_from_vertices


@register_wgpu_render_function(Mesh, MeshBasicMaterial)
class MeshShader(WorldObjectShader):
    type = "render"

    def __init__(self, wobject):
        super().__init__(wobject)

        material = wobject.material
        geometry = wobject.geometry

        # Is this an instanced mesh?
        self["instanced"] = isinstance(wobject, InstancedMesh)

        # Is this a wireframe mesh?
        self["wireframe"] = getattr(material, "wireframe", False)
        self["flat_shading"] = getattr(material, "flat_shading", False)

        # Lighting off in the base class
        self["lighting"] = ""
        self["receive_shadow"] = wobject.receive_shadow

        # Per-vertex color, colormap, or a plane color?
        self["colorspace"] = "srgb"

        color_mode = str(material.color_mode).split(".")[-1]
        if color_mode == "auto":
            if material.map is not None:
                self["color_mode"] = "vertex_map"
                self["color_buffer_channels"] = 0
                self["colorspace"] = material.map.colorspace
            else:
                self["color_mode"] = "uniform"
                self["color_buffer_channels"] = 0
        elif color_mode == "uniform":
            self["color_mode"] = "uniform"
            self["color_buffer_channels"] = 0
        elif color_mode == "vertex":
            self["color_mode"] = "vertex"
            self["color_buffer_channels"] = nchannels = geometry.colors.data.shape[1]
            if nchannels not in (1, 2, 3, 4):
                raise ValueError(f"Geometry.colors needs 1-4 columns, not {nchannels}")
        elif color_mode == "face":
            self["color_mode"] = "face"
            self["color_buffer_channels"] = nchannels = geometry.colors.data.shape[1]
            if nchannels not in (1, 2, 3, 4):
                raise ValueError(f"Geometry.colors needs 1-4 columns, not {nchannels}")
        elif color_mode == "vertex_map":
            self["color_mode"] = "vertex_map"
            self["color_buffer_channels"] = 0
            self["colorspace"] = material.map.colorspace
            if material.map is None:
                raise ValueError(f"Cannot apply colormap is no material.map is set.")
        elif color_mode == "face_map":
            self["color_mode"] = "face_map"
            self["color_buffer_channels"] = 0
            self["colorspace"] = material.map.colorspace
            if material.map is None:
                raise ValueError(f"Cannot apply colormap is no material.map is set.")
        else:
            raise RuntimeError(f"Unknown color_mode: '{color_mode}'")

    def get_bindings(self, wobject, shared):
        geometry = wobject.geometry
        material = wobject.material

        # We're assuming the presence of an index buffer for now
        assert getattr(geometry, "indices", None)

        # Triangles or quads?
        if geometry.indices.data is not None and geometry.indices.data.shape[-1] == 4:
            self["indexer"] = 6
        else:
            self["indexer"] = 3

        # Normals. Usually it'd be given. If not, we'll calculate it from the vertices.
        if getattr(geometry, "normals", None) is not None:
            normal_buffer = geometry.normals
        else:
            if geometry.indices.data.shape[-1] == 4:
                n1 = normals_from_vertices(
                    geometry.positions.data, geometry.indices.data[..., :3]
                )
                n2 = normals_from_vertices(
                    geometry.positions.data, geometry.indices.data[..., [0, 2, 3]]
                )
                normal_data = (n1 + n2) / 2
            else:
                normal_data = normals_from_vertices(
                    geometry.positions.data, geometry.indices.data
                )
            normal_buffer = Buffer(normal_data)

        # Init bindings
        rbuffer = "buffer/read_only_storage"
        bindings = [
            Binding("u_stdinfo", "buffer/uniform", shared.uniform_buffer),
            Binding("u_wobject", "buffer/uniform", wobject.uniform_buffer),
            Binding("u_material", "buffer/uniform", material.uniform_buffer),
            Binding("s_indices", rbuffer, geometry.indices, "VERTEX"),
            Binding("s_positions", rbuffer, geometry.positions, "VERTEX"),
            Binding("s_normals", rbuffer, normal_buffer, "VERTEX"),
        ]

        if hasattr(geometry, "texcoords1") and geometry.texcoords1 is not None:
            bindings.append(
                Binding(
                    "s_texcoords1",
                    "buffer/read_only_storage",
                    geometry.texcoords1,
                    "VERTEX",
                )
            )
            self["use_texcoords1"] = True

        if self["color_mode"] in ("vertex", "face"):
            bindings.append(Binding("s_colors", rbuffer, geometry.colors, "VERTEX"))
        elif self["color_mode"] in ("vertex_map", "face_map"):
            bindings.extend(
                self.define_texcoords_and_colormap(
                    material.map, geometry.texcoords, material.map_interpolation
                )
            )

        F = "FRAGMENT"  # noqa: N806
        sampling = "sampler/filtering"
        sampler = GfxSampler(material.map_interpolation, "repeat")
        texturing = "texture/auto"

        # set envmap configs
        if getattr(material, "env_map", None):
            self._check_texture(
                material.env_map, 6
            )  # TODO: envmap not only cube, but also equirect (hdr format)
            view = GfxTextureView(material.env_map, view_dim="cube")
            bindings.append(Binding("s_env_map", sampling, sampler, F))
            bindings.append(Binding("t_env_map", texturing, view, F))

            if isinstance(material, MeshStandardMaterial):
                self["use_IBL"] = True
            elif isinstance(material, MeshBasicMaterial):
                self["use_env_map"] = True
                self["env_combine_mode"] = getattr(
                    material, "env_combine_mode", "MULTIPLY"
                )

            self["env_mapping_mode"] = getattr(
                material, "env_mapping_mode", "CUBE-REFLECTION"
            )

        # set lightmap configs
        if getattr(material, "light_map", None):
            if "use_texcoords1" not in self.kwargs:
                raise ValueError(
                    "Light map requires a second set of texture coordinates (geometry.texcoords1), but it is not present."
                )
            else:
                self._check_texture(material.light_map)
                view = GfxTextureView(material.light_map, view_dim="2d")
                self["use_light_map"] = True
                bindings.append(Binding("s_light_map", sampling, sampler, F))
                bindings.append(Binding("t_light_map", texturing, view, F))

        # set aomap configs
        if getattr(material, "ao_map", None):
            if "use_texcoords1" not in self.kwargs:
                raise ValueError(
                    "AoMap requires a second set of texture coordinates (geometry.texcoords1), but it is not present."
                )
            else:
                self._check_texture(material.ao_map)
                view = GfxTextureView(material.ao_map, view_dim="2d")
                self["use_ao_map"] = True
                bindings.append(Binding("s_ao_map", sampling, sampler, F))
                bindings.append(Binding("t_ao_map", texturing, view, F))

        # Define shader code for binding
        bindings = {i: binding for i, binding in enumerate(bindings)}
        self.define_bindings(0, bindings)

        # Instanced meshes have an extra storage buffer that we add manually
        bindings1 = {}  # non-auto-generated bindings
        if self["instanced"]:
            bindings1[0] = Binding(
                "s_instance_infos", rbuffer, wobject.instance_buffer, "VERTEX"
            )

        return {
            0: bindings,
            1: bindings1,
        }

    def get_pipeline_info(self, wobject, shared):
        material = wobject.material

        topology = wgpu.PrimitiveTopology.triangle_list

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

    def get_render_info(self, wobject, shared):
        geometry = wobject.geometry
        material = wobject.material

        if geometry.indices.data is not None and geometry.indices.data.shape[-1] == 4:
            offset, size = geometry.indices.draw_range
            offset, size = 6 * offset, 6 * size
        else:
            offset, size = geometry.indices.draw_range
            offset, size = 3 * offset, 3 * size

        n_instances = 1
        if self["instanced"]:
            n_instances = wobject.instance_buffer.nitems

        render_mask = wobject.render_mask
        if not render_mask:
            render_mask = RenderMask.all
            if material.is_transparent:
                render_mask = RenderMask.transparent
            elif self["color_mode"] == "uniform":
                if material.color_is_transparent:
                    render_mask = RenderMask.transparent
                else:
                    render_mask = RenderMask.opaque
            elif self["color_mode"] in ("vertex", "face"):
                if self["color_buffer_channels"] in (1, 3):
                    render_mask = RenderMask.opaque
            elif self["color_mode"] in ("vertex_map", "face_map"):
                if self["colormap_nchannels"] in (1, 3):
                    render_mask = RenderMask.opaque
            elif self["color_mode"] == "normal":
                render_mask = RenderMask.opaque
            else:
                raise RuntimeError(f"Unexpected color mode {self['color_mode']}")

        return {
            "indices": (size, n_instances, offset, 0),
            "render_mask": render_mask,
        }

    def get_code(self):
        return (
            self.code_definitions()
            + self.code_common()
            + self.code_lighting()
            + self.code_vertex()
            + self.code_fragment()
        )

    def code_vertex(self):
        return """

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

            // Get vertex position
            let raw_pos = load_s_positions(i0);
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
            $$ if color_mode == 'face' or color_mode == 'vertex'
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

            // Set texture coords
            $$ if colormap_dim == '1d'
            varyings.texcoord = f32(load_s_texcoords(tex_coord_index));
            $$ elif colormap_dim == '2d'
            varyings.texcoord = vec2<f32>(load_s_texcoords(tex_coord_index));
            $$ elif colormap_dim == '3d'
            varyings.texcoord = vec3<f32>(load_s_texcoords(tex_coord_index));
            $$ endif

            $$ if use_texcoords1 is defined
            varyings.texcoord1 = vec2<f32>(load_s_texcoords1(tex_coord_index));
            $$ endif

            // Set the normal
            let raw_normal = load_s_normals(i0);
            // Transform the normal to world space
            // Note that the world transform matrix cannot be directly applied to the normal
            let normal_matrix = transpose(u_wobject.world_transform_inv);
            let world_normal = normalize((normal_matrix * vec4<f32>(raw_normal, 0.0)).xyz);

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

    """

    def code_fragment(self):
        return """

        @fragment
        fn fs_main(varyings: Varyings, @builtin(front_facing) is_front: bool) -> FragmentOutput {

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

            $$ if color_mode == 'vertex' or color_mode == 'face'
                let color_value = varyings.color;
                let albeido = color_value.rgb;
            $$ elif color_mode == 'vertex_map' or color_mode == 'face_map'
                let color_value = sample_colormap(varyings.texcoord);
                let albeido = color_value.rgb;  // no more colormap
            $$ elif color_mode == 'normal'
                let albeido = normalize(surface_normal) * 0.5 + 0.5;
                let color_value = vec4<f32>(albeido, 1.0);
            $$ else
                let color_value = u_material.color;
                let albeido = color_value.rgb;
            $$ endif

            // Move to physical colorspace (linear photon count) so we can do math
            $$ if colorspace == 'srgb'
                let physical_albeido = srgb2physical(albeido);
            $$ else
                let physical_albeido = albeido;
            $$ endif
            let opacity = color_value.a * u_material.opacity;

            // Get normal used to calculate lighting or reflection
            $$ if lighting or use_env_map is defined
                // Get view direction
                let view = select(
                    normalize(u_stdinfo.cam_transform_inv[3].xyz - varyings.world_pos),
                    ( u_stdinfo.cam_transform_inv * vec4<f32>(0.0, 0.0, 1.0, 0.0) ).xyz,
                    is_orthographic()
                );
                // Get normal used to calculate lighting
                var normal = select(-surface_normal, surface_normal, is_front);
                $$ if use_normal_map is defined
                    let normal_map = textureSample( t_normal_map, s_normal_map, varyings.texcoord ) * 2.0 - 1.0;
                    let normal_map_scale = vec3<f32>( normal_map.xy * u_material.normal_scale, normal_map.z );
                    normal = perturbNormal2Arb(view, normal, normal_map_scale, varyings.texcoord, is_front);
                $$ endif
            $$ endif

            // Lighting
            $$ if lighting
                // Do the math
                var physical_color = lighting_{{ lighting }}(varyings, normal, view, physical_albeido);
            $$ else
                var physical_color = physical_albeido;

                // Light map (pre-baked lighting)
                $$ if use_light_map is defined
                    let light_map_color = srgb2physical( textureSample( t_light_map, s_light_map, varyings.texcoord1 ).rgb );
                    physical_color *= light_map_color * u_material.light_map_intensity * RECIPROCAL_PI;
                $$ endif

                // Ambient occlusion
                $$ if use_ao_map is defined
                    let ao_map_intensity = u_material.ao_map_intensity;
                    let ambientOcclusion = ( textureSample( t_ao_map, s_ao_map, varyings.texcoord1 ).r - 1.0 ) * ao_map_intensity + 1.0;
                    physical_color *= ambientOcclusion;
                $$ endif
            $$ endif

            // Environment mapping
            $$ if use_env_map is defined
                let reflectivity = u_material.reflectivity;
                let specular_strength = 1.0; // TODO: support specular_map
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

            let out_color = vec4<f32>(physical_color, opacity);

            // Wrap up

            apply_clipping_planes(varyings.world_pos);
            var out = get_fragment_output(varyings.position.z, out_color);

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

        """

    def code_lighting(self):
        return ""

    def _check_texture(self, t, size2=1):
        assert isinstance(t, Texture)
        assert t.size[2] == size2
        fmt = to_texture_format(t.format)
        assert "norm" in fmt or "float" in fmt


@register_wgpu_render_function(Mesh, MeshNormalMaterial)
class MeshNormalShader(MeshShader):
    def __init__(self, wobject):
        super().__init__(wobject)
        self["color_mode"] = "normal"
        self["colormap_dim"] = ""  # disable texture if there happens to be one


@register_wgpu_render_function(Mesh, MeshPhongMaterial)
class MeshPhongShader(MeshShader):
    def __init__(self, wobject):
        super().__init__(wobject)
        self["lighting"] = "phong"

    def code_lighting(self):
        # return shaderlib.lighting_phong_simple()  # the quick 'n dirty way
        code = ""
        if self["receive_shadow"]:
            code += shaderlib.shadow()
        code += shaderlib.lighting_phong()
        return code


@register_wgpu_render_function(Mesh, MeshStandardMaterial)
class MeshStandardShader(MeshShader):
    def __init__(self, wobject):
        super().__init__(wobject)
        self["lighting"] = "pbr"

    def get_bindings(self, wobject, shared):
        result = super().get_bindings(wobject, shared)

        geometry = wobject.geometry
        material = wobject.material

        bindings = []

        F = "FRAGMENT"  # noqa: N806
        sampling = "sampler/filtering"
        sampler = GfxSampler(material.map_interpolation, "repeat")
        texturing = "texture/auto"

        # We need uv to use the maps, so if uv not exist, ignore all maps
        if hasattr(geometry, "texcoords") and geometry.texcoords is not None:
            # Texcoords must always be nx2 since it used for all texture maps.
            if not (
                geometry.texcoords.data.ndim == 2
                and geometry.texcoords.data.shape[1] == 2
            ):
                raise ValueError("For standard material, the texcoords must be Nx2")

            if material.normal_map is not None:
                self._check_texture(material.normal_map)
                view = GfxTextureView(material.normal_map, view_dim="2d")
                self["use_normal_map"] = True
                bindings.append(Binding("s_normal_map", sampling, sampler, F))
                bindings.append(Binding("t_normal_map", texturing, view, F))

            if material.roughness_map is not None:
                self._check_texture(material.roughness_map)
                view = GfxTextureView(material.roughness_map, view_dim="2d")
                self["use_roughness_map"] = True
                bindings.append(Binding("s_roughness_map", sampling, sampler, F))
                bindings.append(Binding("t_roughness_map", texturing, view, F))

            if material.metalness_map is not None:
                self._check_texture(material.metalness_map)
                view = GfxTextureView(material.metalness_map, view_dim="2d")
                self["use_metalness_map"] = True
                bindings.append(Binding("s_metalness_map", sampling, sampler, F))
                bindings.append(Binding("t_metalness_map", texturing, view, F))

            if material.emissive_map is not None:
                self._check_texture(material.emissive_map)
                view = GfxTextureView(material.emissive_map, view_dim="2d")
                self["use_emissive_map"] = True
                bindings.append(Binding("s_emissive_map", sampling, sampler, F))
                bindings.append(Binding("t_emissive_map", texturing, view, F))

        # Define shader code for binding
        bindings = {i: binding for i, binding in enumerate(bindings)}
        self.define_bindings(2, bindings)

        # Update result
        result[2] = bindings
        return result

    def code_lighting(self):
        code = ""
        if self["receive_shadow"]:
            code += shaderlib.shadow()
        code += shaderlib.lighting_pbr()
        return code


@register_wgpu_render_function(Mesh, MeshNormalLinesMaterial)
class MeshNormalLinesShader(MeshShader):
    def __init__(self, wobject):
        super().__init__(wobject)
        self["color_mode"] = "uniform"
        self["lighting"] = ""
        self["wireframe"] = False

    def get_pipeline_info(self, wobject, shared):
        d = super().get_pipeline_info(wobject, shared)
        d["primitive_topology"] = wgpu.PrimitiveTopology.line_list
        return d

    def get_render_info(self, wobject, shared):
        # We directly look at the vertex data, so geometry.indices.draw_range is ignored.
        d = super().get_render_info(wobject, shared)
        d["indices"] = wobject.geometry.positions.nitems * 2, d["indices"][1]
        return d

    def code_vertex(self):
        return """

        struct VertexInput {
            @builtin(vertex_index) vertex_index : u32,
        };


        @vertex
        fn vs_main(in: VertexInput) -> Varyings {
            let index = i32(in.vertex_index);
            let r = index % 2;
            let i0 = index / 2;

            // Get regular position
            let raw_pos = load_s_positions(i0);
            var world_pos = u_wobject.world_transform * vec4<f32>(raw_pos, 1.0);

            // Get the normal, expressed in world coords. Use the normal-matrix
            // to take anisotropic scaling into account.
            let normal_matrix = transpose(u_wobject.world_transform_inv);
            let raw_normal = load_s_normals(i0);
            let world_normal = normalize((normal_matrix * vec4<f32>(raw_normal, 0.0)).xyz);

            // Calculate the two end-pieces of the line that we want to show.
            let pos1 = world_pos.xyz / world_pos.w;
            let pos2 = pos1 + world_normal * u_material.line_length;

            // Select either end of the line and make this the world pos
            let pos3 = pos1 * f32(r) + pos2 * (1.0 - f32(r));
            world_pos = vec4<f32>(pos3 * world_pos.w, world_pos.w,);

            // To NDC
            let ndc_pos = u_stdinfo.projection_transform * u_stdinfo.cam_transform * world_pos;

            var varyings: Varyings;
            varyings.world_pos = vec3<f32>(world_pos.xyz / world_pos.w);
            varyings.position = vec4<f32>(ndc_pos);

            // Stub varyings, because the mesh varyings are based on face index
            varyings.normal = vec3<f32>(world_normal);
            varyings.pick_id = u32(u_wobject.id);
            varyings.pick_idx = u32(0);
            varyings.pick_coords = vec3<f32>(0.0);

            return varyings;
        }
        """


@register_wgpu_render_function(Mesh, MeshSliceMaterial)
class MeshSliceShader(WorldObjectShader):
    """Shader for rendering mesh slices."""

    type = "render"

    def __init__(self, wobject):
        super().__init__(wobject)

        material = wobject.material
        geometry = wobject.geometry

        color_mode = str(material.color_mode).split(".")[-1]
        if color_mode == "auto":
            if material.map is not None:
                self["color_mode"] = "vertex_map"
                self["color_buffer_channels"] = 0
            else:
                self["color_mode"] = "uniform"
                self["color_buffer_channels"] = 0
        elif color_mode == "uniform":
            self["color_mode"] = "uniform"
            self["color_buffer_channels"] = 0
        elif color_mode == "vertex":
            self["color_mode"] = "vertex"
            self["color_buffer_channels"] = nchannels = geometry.colors.data.shape[1]
            if nchannels not in (1, 2, 3, 4):
                raise ValueError(f"Geometry.colors needs 1-4 columns, not {nchannels}")
        elif color_mode == "face":
            self["color_mode"] = "face"
            self["color_buffer_channels"] = nchannels = geometry.colors.data.shape[1]
            if nchannels not in (1, 2, 3, 4):
                raise ValueError(f"Geometry.colors needs 1-4 columns, not {nchannels}")
        elif color_mode == "vertex_map":
            self["color_mode"] = "vertex_map"
            self["color_buffer_channels"] = 0
            if material.map is None:
                raise ValueError(f"Cannot apply colormap is no material.map is set.")
        elif color_mode == "face_map":
            self["color_mode"] = "face_map"
            self["color_buffer_channels"] = 0
            if material.map is None:
                raise ValueError(f"Cannot apply colormap is no material.map is set.")
        else:
            raise RuntimeError(f"Unknown color_mode: '{color_mode}'")

    def get_bindings(self, wobject, shared):
        # It would technically be possible to implement colormapping or
        # per-vertex colors, but its a tricky dance to get the per-vertex
        # data (e.g. texcoords) into a varying. And because the visual
        # result is a line, its likely that in most use-cases a uniform
        # color is preferred anyway. So for now we don't implement that.

        geometry = wobject.geometry
        material = wobject.material

        # Init uniform bindings
        bindings = [
            Binding("u_stdinfo", "buffer/uniform", shared.uniform_buffer),
            Binding("u_wobject", "buffer/uniform", wobject.uniform_buffer),
            Binding("u_material", "buffer/uniform", material.uniform_buffer),
        ]

        # We're assuming the presence of an index buffer for now
        assert getattr(geometry, "indices", None)

        # Init storage buffer bindings
        rbuffer = "buffer/read_only_storage"
        bindings.append(Binding("s_indices", rbuffer, geometry.indices, "VERTEX"))
        bindings.append(Binding("s_positions", rbuffer, geometry.positions, "VERTEX"))

        # Bindings for color
        if self["color_mode"] in ("vertex", "face"):
            bindings.append(Binding("s_colors", rbuffer, geometry.colors, "VERTEX"))
        elif self["color_mode"] in ("vertex_map", "face_map"):
            bindings.extend(
                self.define_texcoords_and_colormap(
                    material.map, geometry.texcoords, material.map_interpolation
                )
            )

        # Let the shader generate code for our bindings
        bindings = {i: binding for i, binding in enumerate(bindings)}
        self.define_bindings(0, bindings)

        return {
            0: bindings,
        }

    def get_pipeline_info(self, wobject, shared):
        return {
            "primitive_topology": wgpu.PrimitiveTopology.triangle_list,
            "cull_mode": wgpu.CullMode.none,
        }

    def get_render_info(self, wobject, shared):
        material = wobject.material  # noqa
        geometry = wobject.geometry

        offset, size = geometry.indices.draw_range
        offset, size = offset * 6, size * 6

        # As long as we don't use alpha for aa in the frag shader, we can use a render_mask of 1 or 2.
        render_mask = wobject.render_mask
        if not render_mask:
            if material.is_transparent:
                render_mask = RenderMask.transparent
            elif self["color_mode"] == "uniform":
                if material.color_is_transparent:
                    render_mask = RenderMask.transparent
                else:
                    render_mask = RenderMask.opaque
            elif self["color_mode"] in ("vertex", "face"):
                if self["color_buffer_channels"] in (1, 3):
                    render_mask = RenderMask.opaque
                else:
                    render_mask = RenderMask.all
            elif self["color_mode"] in ("vertex_map", "face_map"):
                if self["colormap_nchannels"] in (1, 3):
                    render_mask = RenderMask.opaque
                else:
                    render_mask = RenderMask.all
            else:
                raise RuntimeError(f"Unexpected color mode {self['color_mode']}")

        return {
            "indices": (size, 1, offset, 0),
            "render_mask": render_mask,
        }

    def get_code(self):
        return (
            self.code_definitions()
            + self.code_common()
            + self.code_vertex()
            + self.code_fragment()
        )

    def code_vertex(self):
        return """
        struct VertexInput {
            @builtin(vertex_index) vertex_index : u32,
        };
        @vertex
        fn vs_main(in: VertexInput) -> Varyings {
            // This vertex shader uses VertexId and storage buffers instead of
            // vertex buffers. It creates 6 vertices for each face in the mesh,
            // drawn with triangle-list. For the faces that cross the plane, we
            // draw a (thick) line segment with round caps (we need 6 verts for that).
            // Other faces become degenerate triangles.
            let screen_factor = u_stdinfo.logical_size.xy / 2.0;
            let l2p = u_stdinfo.physical_size.x / u_stdinfo.logical_size.x;
            let thickness = u_material.thickness;  // in logical pixels

            // Get the face index, and sample the vertex indices
            let index = i32(in.vertex_index);
            let segment_index = index % 6;
            let face_index = (index - segment_index) / 6;
            let vii = vec3<i32>(load_s_indices(face_index));

            // Vertex positions of this face, in local object coordinates
            let pos1a = load_s_positions(vii[0]);
            let pos2a = load_s_positions(vii[1]);
            let pos3a = load_s_positions(vii[2]);
            let pos1b = u_wobject.world_transform * vec4<f32>(pos1a, 1.0);
            let pos2b = u_wobject.world_transform * vec4<f32>(pos2a, 1.0);
            let pos3b = u_wobject.world_transform * vec4<f32>(pos3a, 1.0);
            let pos1 = pos1b.xyz / pos1b.w;
            let pos2 = pos2b.xyz / pos2b.w;
            let pos3 = pos3b.xyz / pos3b.w;
            // Get the plane definition
            let plane = u_material.plane.xyzw;  // ax + by + cz + d == 0
            let n = plane.xyz;  // not necessarily a unit vector
            // Intersect the plane with pos 1 and 2
            var p: vec3<f32>;
            p = pos1.xyz;
            let denom1 = dot(n,  pos2.xyz - pos1.xyz);
            let t1 = -(plane.x * p.x + plane.y * p.y + plane.z * p.z + plane.w) / denom1;
            // Intersect the plane with pos 2 and 3
            p = pos2.xyz;
            let denom2 = dot(n, pos3.xyz - pos2.xyz);
            let t2 = -(plane.x * p.x + plane.y * p.y + plane.z * p.z + plane.w) / denom2;
            // Intersect the plane with pos 3 and 1
            p = pos3.xyz;
            let denom3 = dot(n, pos1.xyz - pos3.xyz);
            let t3 = -(plane.x * p.x + plane.y * p.y + plane.z * p.z + plane.w) / denom3;
            // Selectors (the denom check seems not needed, but it feels safer to do, in case e.g. the precision changes)
            let b1 = select(0, 4, (t1 >= 0.0) && (t1 <= 1.0) && (denom1 != 0.0));
            let b2 = select(0, 2, (t2 >= 0.0) && (t2 <= 1.0) && (denom2 != 0.0));
            let b3 = select(0, 1, (t3 >= 0.0) && (t3 <= 1.0) && (denom3 != 0.0));
            var pos_index: i32;
            pos_index = b1 + b2 + b3;

            // The big triage
            var the_pos: vec4<f32>;
            var the_coord: vec2<f32>;
            var segment_length: f32;
            var pick_idx = u32(0u);
            var pick_coords = vec3<f32>(0.0);
            if (pos_index < 3 || pos_index == 4) {
                // Just return the same vertex, resulting in degenerate triangles
                the_pos = u_stdinfo.projection_transform * u_stdinfo.cam_transform * vec4<f32>(pos1, 1.0);
                the_coord = vec2<f32>(0.0, 0.0);
                segment_length = 0.0;
            } else {
                if (pos_index == 7) {
                    // For each edge we check whether it looks like it is selected. The value
                    // of the other t does not matter because it's always on the edge too.
                    // I would expect that comparing with 0.0 and 1.0 would work, but
                    // apparently the t-values can also be 0.5.
                    if (t1 < 0.5 && t2 >= 0.5) { pos_index = 6; }
                    else if (t2 < 0.5 && t3 >= 0.5) { pos_index = 3; }
                    else if (t3 < 0.5 && t1 >= 0.5) { pos_index = 5; }
                }
                // Get the positions where the frame intersects the plane
                let pos00: vec3<f32> = pos1;
                let pos12: vec3<f32> = mix(pos1, pos2, vec3<f32>(t1, t1, t1));
                let pos23: vec3<f32> = mix(pos2, pos3, vec3<f32>(t2, t2, t2));
                let pos31: vec3<f32> = mix(pos3, pos1, vec3<f32>(t3, t3, t3));
                // b1+b2+b3     000    001    010    011    100    101    110    111
                var positions_a = array<vec3<f32>, 8>(pos00, pos00, pos00, pos23, pos00, pos12, pos12, pos00);
                var positions_b = array<vec3<f32>, 8>(pos00, pos00, pos00, pos31, pos00, pos31, pos23, pos00);
                // Select the two positions that define the line segment
                let pos_a = positions_a[pos_index];
                let pos_b = positions_b[pos_index];
                // Same for face weights
                let fw00 = vec3<f32>(0.5, 0.5, 0.5);
                let fw12 = mix(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(t1, t1, t1));
                let fw23 = mix(vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(t2, t2, t2));
                let fw31 = mix(vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(t3, t3, t3));
                var fws_a = array<vec3<f32>, 8>(fw00, fw00, fw00, fw23, fw00, fw12, fw12, fw00);
                var fws_b = array<vec3<f32>, 8>(fw00, fw00, fw00, fw31, fw00, fw31, fw23, fw00);
                let fw_a = fws_a[pos_index];
                let fw_b = fws_b[pos_index];
                // Go from local coordinates to NDC
                var npos_a: vec4<f32> = u_stdinfo.projection_transform * u_stdinfo.cam_transform * vec4<f32>(pos_a, 1.0);
                var npos_b: vec4<f32> = u_stdinfo.projection_transform * u_stdinfo.cam_transform * vec4<f32>(pos_b, 1.0);
                // Don't forget to "normalize"!
                // todo: omitting this step diminish the thickness with distance, but it that the way?
                npos_a = npos_a / npos_a.w;
                npos_b = npos_b / npos_b.w;
                // And to logical pixel coordinates (don't worry about offset)
                let ppos_a = npos_a.xy * screen_factor;
                let ppos_b = npos_b.xy * screen_factor;
                // Get the segment vector, its length, and how much it scales because of thickness
                let v0 = ppos_b - ppos_a;
                segment_length = length(v0);  // in logical pixels;
                let segment_factor = (segment_length + thickness) / segment_length;
                // Get the (orthogonal) unit vectors that span the segment
                let v1 = normalize(v0);
                let v2 = vec2<f32>(v1.y, -v1.x);
                // Get the vector, in local logical pixels for the segment's square
                let pvec_local = 0.5 * vec2<f32>(segment_length + thickness, thickness);
                // Select one of the four corners of the segment rectangle
                var vecs = array<vec2<f32>, 6>(
                    vec2<f32>(-1.0, -1.0),
                    vec2<f32>( 1.0,  1.0),
                    vec2<f32>(-1.0,  1.0),
                    vec2<f32>( 1.0,  1.0),
                    vec2<f32>(-1.0, -1.0),
                    vec2<f32>( 1.0, -1.0),
                );
                let the_vec = vecs[segment_index];
                // Construct the position, also make sure zw scales correctly
                let pvec = the_vec.x * pvec_local.x * v1 + the_vec.y * pvec_local.y * v2;
                let z_range = (npos_b.z - npos_a.z) * segment_factor;
                let the_pos_p = 0.5 * (ppos_a + ppos_b) + pvec;
                let the_pos_z = 0.5 * (npos_a.z + npos_b.z) + the_vec.x * z_range * 0.5;
                let depth_offset = -0.0001;  // to put the mesh slice atop a mesh
                the_pos = vec4<f32>(the_pos_p / screen_factor, the_pos_z + depth_offset, 1.0);
                // Define the local coordinate in physical pixels
                the_coord = the_vec * pvec_local;
                // Picking info
                pick_idx = u32(face_index);
                let mixval = the_vec.x * 0.5 + 0.5;
                pick_coords = vec3<f32>(mix(fw_a, fw_b, vec3<f32>(mixval, mixval, mixval)));
            }

            // Shader output
            var varyings: Varyings;
            varyings.position = vec4<f32>(the_pos);
            varyings.world_pos = vec3<f32>(ndc_to_world_pos(the_pos));
            varyings.dist2center = vec2<f32>(the_coord * l2p);
            varyings.segment_length = f32(segment_length * l2p);
            varyings.segment_width = f32(thickness * l2p);
            varyings.pick_idx = u32(pick_idx);
            varyings.pick_coords = vec3<f32>(pick_coords);

            // per-vertex or per-face coloring
            $$ if color_mode == 'face' or color_mode == 'vertex'
                $$ if color_mode == 'face'
                    let cvalue = load_s_colors(face_index);
                $$ else
                    let cvalue = (  load_s_colors(vii[0]) * pick_coords[0] +
                                    load_s_colors(vii[1]) * pick_coords[1] +
                                    load_s_colors(vii[2]) * pick_coords[2] );
                $$ endif
                $$ if color_buffer_channels == 1
                    varyings.color = vec4<f32>(cvalue, cvalue, cvalue, 1.0);
                $$ elif color_buffer_channels == 2
                    varyings.color = vec4<f32>(cvalue.r, cvalue.r, cvalue.r, cvalue.g);
                $$ elif color_buffer_channels == 3
                    varyings.color = vec4<f32>(cvalue, 1.0);
                $$ elif color_buffer_channels == 4
                    varyings.color = vec4<f32>(cvalue);
                $$ endif
            $$ endif

            // Set texture coords
            $$ if color_mode == 'face_map' or color_mode == 'vertex_map'
                $$ if color_mode == 'face_map'
                    let cvalue = load_s_texcoords(face_index);
                $$ else
                    let cvalue = (  load_s_texcoords(vii[0]) * pick_coords[0] +
                                    load_s_texcoords(vii[1]) * pick_coords[1] +
                                    load_s_texcoords(vii[2]) * pick_coords[2] );
                $$ endif
                $$ if colormap_dim == '1d'
                    varyings.texcoord = f32(cvalue);
                $$ elif colormap_dim == '2d'
                    varyings.texcoord = vec2<f32>(cvalue);
                $$ elif colormap_dim == '3d'
                    varyings.texcoord = vec3<f32>(cvalue);
                $$ endif
            $$ endif

            return varyings;
        }
        """

    def code_fragment(self):
        return """
        @fragment
        fn fs_main(varyings: Varyings) -> FragmentOutput {
            // Discart fragments that are too far from the centerline. This makes round caps.
            // Note that we operate in physical pixels here.
            let distx = max(0.0, abs(varyings.dist2center.x) - 0.5 * varyings.segment_length);
            let dist = length(vec2<f32>(distx, varyings.dist2center.y));
            if (dist > varyings.segment_width * 0.5) {
                discard;
            }

            $$ if color_mode == 'vertex' or color_mode == 'face'
                let color_value = varyings.color;
                let albeido = color_value.rgb;
            $$ elif color_mode == 'vertex_map' or color_mode == 'face_map'
                let color_value = sample_colormap(varyings.texcoord);
                let albeido = color_value.rgb;  // no more colormap
            $$ else
                let color_value = u_material.color;
                let albeido = color_value.rgb;
            $$ endif

            // No aa. This is something we need to decide on. See line renderer.
            // Making this < 1 would affect the suggested_render_mask.
            let alpha = 1.0;
            // Set color
            let physical_color = srgb2physical(albeido);
            let opacity = min(1.0, color_value.a) * alpha;
            let out_color = vec4<f32>(physical_color, opacity);

            // Wrap up
            apply_clipping_planes(varyings.world_pos);
            var out = get_fragment_output(varyings.position.z, out_color);
            $$ if write_pick
            // The wobject-id must be 20 bits. In total it must not exceed 64 bits.
            out.pick = (
                pick_pack(u32(u_wobject.id), 20) +
                pick_pack(varyings.pick_idx, 26) +
                pick_pack(u32(varyings.pick_coords.x * 63.0), 6) +
                pick_pack(u32(varyings.pick_coords.y * 63.0), 6) +
                pick_pack(u32(varyings.pick_coords.z * 63.0), 6)
            );
            $$ endif

            // We curve the line away to the background, so that line pieces overlap each-other
            // in a better way, especially avoiding the joins from overlapping the next line piece.
            out.depth = varyings.position.z + 0.0001 * dist;
            return out;
        }
        """
