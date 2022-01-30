import wgpu  # only for flags/enums

from . import register_wgpu_render_function
from ._shadercomposer import Binding, WorldObjectShader
from .pointsrender import handle_colormap
from ...objects import Mesh, InstancedMesh
from ...materials import (
    MeshBasicMaterial,
    MeshPhongMaterial,
    MeshFlatMaterial,
    MeshNormalMaterial,
    MeshNormalLinesMaterial,
    MeshSliceMaterial,
)
from ...resources import Buffer
from ...utils import normals_from_vertices


@register_wgpu_render_function(Mesh, MeshBasicMaterial)
def mesh_renderer(render_info):
    """Render function capable of rendering meshes."""
    wobject = render_info.wobject
    geometry = wobject.geometry
    material = wobject.material  # noqa

    # Initialize
    topology = wgpu.PrimitiveTopology.triangle_list
    shader = MeshShader(
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

    # Triage based on material
    if isinstance(material, MeshNormalMaterial):
        # Special simple fragment shader
        shader["color_mode"] = "normal"
        shader["colormap_dim"] = ""  # disable texture if there happens to be one
    elif isinstance(material, MeshNormalLinesMaterial):
        # Special simple vertex shader with plain fragment shader
        topology = wgpu.PrimitiveTopology.line_list
        shader.vertex_shader = shader.vertex_shader_normal_lines
        index_buffer = None
        n = geometry.positions.nitems * 2
        shader["color_mode"] = "uniform"
        shader["lighting"] = ""
        shader["wireframe"] = False
    elif isinstance(material, MeshFlatMaterial):
        shader["lighting"] = "flat"
    elif isinstance(material, MeshPhongMaterial):
        shader["lighting"] = "phong"
    else:
        pass  # simple lighting

    # Instanced meshes have an extra storage buffer that we add manually
    n_instances = 1
    if isinstance(wobject, InstancedMesh):
        shader["instanced"] = True
        bindings.append(
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
        }
    ]


class MeshShader(WorldObjectShader):
    def get_code(self):
        return (
            self.get_definitions()
            + self.common_functions()
            + self.helpers()
            + self.vertex_shader()
            + self.fragment_shader()
        )

    def vertex_shader(self):
        return """

        struct VertexInput {
            [[builtin(vertex_index)]] vertex_index : u32;
            $$ if instanced
            [[builtin(instance_index)]] instance_index : u32;
            $$ endif
        };

        $$ if instanced
        struct InstanceInfo {
            transform: mat4x4<f32>;
            id: u32;
        };
        struct InstanceInfos {
            data: [[stride(80)]] array<InstanceInfo>;
        };
        [[group(2), binding(0)]]
        var<storage,read> s_instance_infos: InstanceInfos;
        $$ endif


        [[stage(vertex)]]
        fn vs_main(in: VertexInput) -> Varyings {

            // Select what face we're at
            let index = i32(in.vertex_index);
            let face_index = index / 3;
            var sub_index = index % 3;

            // If the camera flips a dimension, it flips the face winding.
            // We can correct for this by adjusting the order (sub_index) here.
            sub_index = select(sub_index, -1 * (sub_index - 1) + 1, u_stdinfo.flipped_winding > 0);

            // Sample
            let ii = load_s_indices(face_index);
            let i0 = i32(ii[sub_index]);

            // Get world transform
            $$ if instanced
                let instance_info = s_instance_infos.data[in.instance_index];
                let world_transform = u_wobject.world_transform * instance_info.transform;
            $$ else
                let world_transform = u_wobject.world_transform;
            $$ endif

            // Get vertex position
            let raw_pos = load_s_positions(i0);
            let world_pos = world_transform * vec4<f32>(raw_pos, 1.0);
            var ndc_pos = u_stdinfo.projection_transform * u_stdinfo.cam_transform * world_pos;

            // For the wireframe we also need the ndc_pos of the other vertices of this face
            $$ if wireframe
                $$ for i in (1, 2, 3)
                    let raw_pos{{ i }} = load_s_positions(i32(ii[{{ i - 1 }}]));
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

            // Per-vertex colors
            $$ if vertex_color_channels == 1
            let cvalue = load_s_colors(i0);
            varyings.color = vec4<f32>(cvalue, cvalue, cvalue, 1.0);
            $$ elif vertex_color_channels == 2
            let cvalue = load_s_colors(i0);
            varyings.color = vec4<f32>(cvalue.r, cvalue.r, cvalue.r, cvalue.g);
            $$ elif vertex_color_channels == 3
            varyings.color = vec4<f32>(load_s_colors(i0), 1.0);
            $$ elif vertex_color_channels == 4
            varyings.color = vec4<f32>(load_s_colors(i0));
            $$ endif

            // Set texture coords
            $$ if colormap_dim == '1d'
            varyings.texcoord = f32(load_s_texcoords(i0));
            $$ elif colormap_dim == '2d'
            varyings.texcoord = vec2<f32>(load_s_texcoords(i0));
            $$ elif colormap_dim == '3d'
            varyings.texcoord = vec3<f32>(load_s_texcoords(i0));
            $$ endif

            // Set the normal
            let raw_normal = load_s_normals(i0);
            let world_pos_n = world_transform * vec4<f32>(raw_pos + raw_normal, 1.0);
            let world_normal = normalize(world_pos_n - world_pos).xyz;
            varyings.normal = vec3<f32>(world_normal);

            // Vectors for lighting, all in world coordinates
            let view_vec = normalize(ndc_to_world_pos(vec4<f32>(0.0, 0.0, 1.0, 1.0)));
            varyings.view = vec3<f32>(view_vec);
            varyings.light = vec3<f32>(view_vec);

            // Set wireframe barycentric-like coordinates
            $$ if wireframe
                $$ for i in (1, 2, 3)
                    let p{{ i }} = (ndc_pos{{ i }}.xy / ndc_pos{{ i }}.w) * u_stdinfo.logical_size * 0.5;
                $$ endfor
                let dist1 = abs((p3.x - p2.x) * (p2.y - p1.y) - (p2.x - p1.x) * (p3.y - p2.y)) / distance(p2, p3);
                let dist2 = abs((p3.x - p1.x) * (p1.y - p2.y) - (p1.x - p2.x) * (p3.y - p1.y)) / distance(p1, p3);
                let dist3 = abs((p1.x - p2.x) * (p2.y - p3.y) - (p2.x - p3.x) * (p1.y - p2.y)) / distance(p2, p1);
                var arr_wireframe_coords = array<vec3<f32>, 3>(
                    vec3<f32>(dist1, 0.0, 0.0), vec3<f32>(0.0, dist2, 0.0), vec3<f32>(0.0, 0.0, dist3)
                );
                varyings.wireframe_coords = vec3<f32>(arr_wireframe_coords[sub_index]);  // in logical pixels
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
            varyings.pick_idx = u32(face_index);
            var arr_pick_coords = array<vec3<f32>, 3>(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(0.0, 0.0, 1.0));
            varyings.pick_coords = vec3<f32>(arr_pick_coords[sub_index]);

            return varyings;
        }

    """

    def vertex_shader_normal_lines(self):
        return """

        struct VertexInput {
            [[builtin(vertex_index)]] vertex_index : u32;
        };


        [[stage(vertex)]]
        fn vs_main(in: VertexInput) -> Varyings {
            let index = i32(in.vertex_index);
            let r = index % 2;
            let i0 = (index - r) / 2;

            let raw_pos = load_s_positions(i0);
            let raw_normal = load_s_normals(i0);

            let world_pos1 = u_wobject.world_transform * vec4<f32>(raw_pos, 1.0);
            let world_pos2 = u_wobject.world_transform * vec4<f32>(raw_pos + raw_normal, 1.0);

            // The normal is sized in world coordinates
            let world_normal = normalize(world_pos2 - world_pos1);

            let amplitude = 1.0;
            let world_pos = world_pos1 + f32(r) * world_normal * amplitude;
            let ndc_pos = u_stdinfo.projection_transform * u_stdinfo.cam_transform * world_pos;

            var varyings: Varyings;
            varyings.world_pos = vec3<f32>(world_pos.xyz / world_pos.w);
            varyings.position = vec4<f32>(ndc_pos);

            // Stub varyings, because the mesh varyings are based on face index
            varyings.pick_id = u32(u_wobject.id);
            varyings.pick_idx = u32(0);
            varyings.pick_coords = vec3<f32>(0.0);

            return varyings;
        }
        """

    def fragment_shader(self):
        return """

        [[stage(fragment)]]
        fn fs_main(varyings: Varyings, [[builtin(front_facing)]] is_front: bool) -> FragmentOutput {

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

            // Lighting
            $$ if lighting
                let world_pos = varyings.world_pos;
                let lit_color = lighting_{{ lighting }}(is_front, varyings.world_pos, varyings.normal, varyings.light, varyings.view, albeido);
            $$ else
                let lit_color = albeido;
            $$ endif

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

    def helpers(self):
        return """

        $$ if lighting
        fn lighting_phong(
            is_front: bool,
            world_pos: vec3<f32>,
            normal: vec3<f32>,
            light: vec3<f32>,
            view: vec3<f32>,
            albeido: vec3<f32>,
        ) -> vec3<f32> {
            let light_color = vec3<f32>(1.0, 1.0, 1.0);

            // Light parameters
            let ambient_factor = 0.1;
            let diffuse_factor = 0.7;
            let specular_factor = 0.3;
            let shininess = u_material.shininess;

            // Base vectors
            var normal: vec3<f32> = normalize(normal);
            let view = normalize(view);
            let light = normalize(light);

            // Maybe flip the normal - otherwise backfacing faces are not lit
            // See pygfx/issues/#105 for details
            normal = select(normal, -normal, is_front);

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
            let emissive_color = u_material.emissive_color.rgb;

            // Put together
            return albeido * (ambient_color + diffuse_color) + specular_color + emissive_color;
        }

        fn lighting_flat(
            is_front: bool,
            world_pos: vec3<f32>,
            normal: vec3<f32>,
            light: vec3<f32>,
            view: vec3<f32>,
            albeido: vec3<f32>,
        ) -> vec3<f32> {

            let u = dpdx(world_pos);
            let v = dpdy(world_pos);
            var normal = normalize(cross(u, v));

            // The normal calculated above may not be oriented correctly.
            // We have two flags: is_front and u_stdinfo.flipped_winding.
            // Note that lighting_phong() also applies the is_front flag.
            // Below code means: flip the normal if the XOR of these two flags is true.
            normal = select(normal, -normal, (select(0, 1, is_front) + u_stdinfo.flipped_winding) == 1);

            // The rest is the same as phong
            return lighting_phong(is_front, world_pos, normal, light, view, albeido);
        }
        $$ endif

        """


@register_wgpu_render_function(Mesh, MeshSliceMaterial)
def meshslice_renderer(render_info):
    """Render function capable of rendering mesh slices."""

    # It would technically be possible to implement colormapping or
    # per-vertex colors, but its a tricky dance to get the per-vertex
    # data (e.g. texcoords) into a varying. And because the visual
    # result is a line, its likely that in most use-cases a uniform
    # color is preferred anyway. So for now we don't implement that.

    wobject = render_info.wobject
    geometry = wobject.geometry
    material = wobject.material  # noqa

    # Initialize
    topology = wgpu.PrimitiveTopology.triangle_list
    shader = MeshSliceShader(render_info)

    # We're assuming the presence of an index buffer for now
    assert getattr(geometry, "indices", None)
    n = (geometry.indices.data.size // 3) * 6
    n_instances = 1

    bindings = {}

    # Init uniform bindings
    bindings[0] = Binding("u_stdinfo", "buffer/uniform", render_info.stdinfo_uniform)
    bindings[1] = Binding("u_wobject", "buffer/uniform", wobject.uniform_buffer)
    bindings[2] = Binding("u_material", "buffer/uniform", material.uniform_buffer)

    # Init storage buffer bindings
    bindings[3] = Binding(
        "s_indices", "buffer/read_only_storage", geometry.indices, "VERTEX"
    )
    bindings[4] = Binding(
        "s_positions", "buffer/read_only_storage", geometry.positions, "VERTEX"
    )

    # Let the shader generate code for our bindings
    for i, binding in bindings.items():
        shader.define_binding(0, i, binding)

    # As long as we don't use alpha for aa in the frag shader, we can use a render_mask of 1 or 2.
    only_color_and_opacity_determine_fragment_alpha = True
    suggested_render_mask = 3
    if only_color_and_opacity_determine_fragment_alpha:
        is_opaque = material.opacity >= 1 and material.color[3] >= 1
        suggested_render_mask = 1 if is_opaque else 2

    # Put it together!
    return [
        {
            "suggested_render_mask": suggested_render_mask,
            "render_shader": shader,
            "primitive_topology": topology,
            "indices": (range(n), range(n_instances)),
            "index_buffer": None,
            "vertex_buffers": {},
            "bindings0": bindings,
        }
    ]


class MeshSliceShader(WorldObjectShader):
    def get_code(self):
        return (
            self.get_definitions()
            + self.common_functions()
            + self.vertex_shader()
            + self.fragment_shader()
        )

    def vertex_shader(self):
        return """

        struct VertexInput {
            [[builtin(vertex_index)]] vertex_index : u32;
        };


        [[stage(vertex)]]
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
            let ii = vec3<i32>(load_s_indices(face_index));

            // Vertex positions of this face, in local object coordinates
            let pos1a = load_s_positions(ii[0]);
            let pos2a = load_s_positions(ii[1]);
            let pos3a = load_s_positions(ii[2]);
            let pos1b = u_wobject.world_transform * vec4<f32>(pos1a, 1.0);
            let pos2b = u_wobject.world_transform * vec4<f32>(pos2a, 1.0);
            let pos3b = u_wobject.world_transform * vec4<f32>(pos3a, 1.0);
            let pos1 = pos1b.xyz / pos1b.w;
            let pos2 = pos2b.xyz / pos2b.w;
            let pos3 = pos3b.xyz / pos3b.w;

            // Get the plane definition
            let plane = u_material.plane.xyzw;  // ax + by + cz + d
            let n = plane.xyz;  // not necessarily a unit vector

            // Intersect the plane with pos 1 and 2
            var p: vec3<f32>;
            var u: vec3<f32>;
            p = pos1.xyz;
            u = pos2.xyz - pos1.xyz;
            let t1 = -(plane.x * p.x + plane.y * p.y + plane.z * p.z + plane.w) / dot(n, u);
            // Intersect the plane with pos 2 and 3
            p = pos2.xyz;
            u = pos3.xyz - pos2.xyz;
            let t2 = -(plane.x * p.x + plane.y * p.y + plane.z * p.z + plane.w) / dot(n, u);
            // Intersect the plane with pos 3 and 1
            p = pos3.xyz;
            u = pos1.xyz - pos3.xyz;
            let t3 = -(plane.x * p.x + plane.y * p.y + plane.z * p.z + plane.w) / dot(n, u);

            // Selectors

            let b1 = select(0, 4, (t1 > 0.0) && (t1 < 1.0));
            let b2 = select(0, 2, (t2 > 0.0) && (t2 < 1.0));
            let b3 = select(0, 1, (t3 > 0.0) && (t3 < 1.0));
            let pos_index = b1 + b2 + b3;

            // The big triage
            var the_pos: vec4<f32>;
            var the_coord: vec2<f32>;
            var segment_length: f32;
            var pick_idx = u32(0u);
            var pick_coords = vec3<f32>(0.0);

            if (pos_index < 3) {//   (pos_index < 3) {  // or dot(n, u) == 0.0
                // Just return the same vertex, resulting in degenerate triangles
                the_pos = u_stdinfo.projection_transform * u_stdinfo.cam_transform * vec4<f32>(pos1, 1.0);
                the_coord = vec2<f32>(0.0, 0.0);
                segment_length = 0.0;

            } else {
                // Get the positions where the frame intersects the plane
                let pos00: vec3<f32> = pos1;
                let pos12: vec3<f32> = mix(pos1, pos2, vec3<f32>(t1, t1, t1));
                let pos23: vec3<f32> = mix(pos2, pos3, vec3<f32>(t2, t2, t2));
                let pos31: vec3<f32> = mix(pos3, pos1, vec3<f32>(t3, t3, t3));
                // b1+b2+b3     000    001    010    011    100    101    110    111
                var positions_a = array<vec3<f32>, 8>(pos00, pos00, pos00, pos23, pos00, pos12, pos12, pos12);
                var positions_b = array<vec3<f32>, 8>(pos00, pos00, pos00, pos31, pos00, pos31, pos23, pos23);
                // Select the two positions that define the line segment
                let pos_a = positions_a[pos_index];
                let pos_b = positions_b[pos_index];

                // Same for face weights
                let fw00 = vec3<f32>(0.5, 0.5, 0.5);
                let fw12 = mix(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(t1, t1, t1));
                let fw23 = mix(vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(t2, t2, t2));
                let fw31 = mix(vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(t3, t3, t3));
                var fws_a = array<vec3<f32>, 8>(fw00, fw00, fw00, fw23, fw00, fw12, fw12, fw12);
                var fws_b = array<vec3<f32>, 8>(fw00, fw00, fw00, fw31, fw00, fw31, fw23, fw23);
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
            return varyings;
        }
        """

    def fragment_shader(self):
        return """

        [[stage(fragment)]]
        fn fs_main(varyings: Varyings) -> FragmentOutput {
            var out: FragmentOutput;

            // Discart fragments that are too far from the centerline. This makes round caps.
            // Note that we operate in physical pixels here.
            let distx = max(0.0, abs(varyings.dist2center.x) - 0.5 * varyings.segment_length);
            let dist = length(vec2<f32>(distx, varyings.dist2center.y));
            if (dist > varyings.segment_width * 0.5) {
                discard;
            }

            // No aa. This is something we need to decide on. See line renderer.
            // Making this < 1 would affect the suggested_render_mask.
            let alpha = 1.0;

            // Set color
            let color = u_material.color;
            let final_color = vec4<f32>(color.rgb, min(1.0, color.a) * alpha);

            // Wrap up

            apply_clipping_planes(varyings.world_pos);
            var out = get_fragment_output(varyings.position.z, final_color);

            $$ if write_pick
            // The wobject-id must be 20 bits. In total it must not exceed 64 bits.
            out.pick = (
                pick_pack(u32(u_wobject.id), 20) +
                pick_pack(varyings.pick_idx, 26) +
                pick_pack(u32(varyings.pick_coords.x * 64.0), 6) +
                pick_pack(u32(varyings.pick_coords.y * 64.0), 6) +
                pick_pack(u32(varyings.pick_coords.z * 64.0), 6)
            );
            $$ endif

            return out;
        }
        """
