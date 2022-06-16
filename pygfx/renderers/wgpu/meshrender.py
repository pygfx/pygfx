import wgpu  # only for flags/enums

from . import register_wgpu_render_function
from ._shadercomposer import WorldObjectShader
from ._pipelinebuilder import Binding
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
class MeshShader(WorldObjectShader):

    type = "render"

    def __init__(self, build_args):
        super().__init__(build_args)

        wobject = build_args.wobject
        material = wobject.material
        geometry = wobject.geometry

        # Is this an instanced mesh?
        self["instanced"] = isinstance(wobject, InstancedMesh)

        # Is this a wireframe mesh?
        self["wireframe"] = material.wireframe

        # Lighting off in the base class
        self["lighting"] = ""

        # Per-vertex color, colormap, or a plane color?
        if material.vertex_colors:
            self["color_mode"] = "vertex"
            self["vertex_color_channels"] = nchannels = geometry.colors.data.shape[1]
            if nchannels not in (1, 2, 3, 4):
                raise ValueError(f"Geometry.colors needs 1-4 columns, not {nchannels}")
        elif material.map is not None:
            self["color_mode"] = "map"
            self["vertex_color_channels"] = 0
        else:
            self["color_mode"] = "uniform"
            self["vertex_color_channels"] = 0

    def get_resources(self, build_args):

        wobject = build_args.wobject
        shared = build_args.shared
        geometry = wobject.geometry
        material = wobject.material

        # indexbuffer
        # vertex_buffers
        # list of list of dicts

        # We're assuming the presence of an index buffer for now
        assert getattr(geometry, "indices", None)

        # Normals. Usually it'd be given. If not, we'll calculate it from the vertices.
        if getattr(geometry, "normals", None) is not None:
            normal_buffer = geometry.normals
        else:
            normal_data = normals_from_vertices(
                geometry.positions.data, geometry.indices.data
            )
            normal_buffer = Buffer(normal_data)

        # Init bindings
        bindings = [
            Binding("u_stdinfo", "buffer/uniform", shared.uniform_buffer),
            Binding("u_wobject", "buffer/uniform", wobject.uniform_buffer),
            Binding("u_material", "buffer/uniform", material.uniform_buffer),
            Binding(
                "s_indices", "buffer/read_only_storage", geometry.indices, "VERTEX"
            ),
            Binding(
                "s_positions", "buffer/read_only_storage", geometry.positions, "VERTEX"
            ),
            Binding("s_normals", "buffer/read_only_storage", normal_buffer, "VERTEX"),
        ]

        if self["color_mode"] == "vertex":
            bindings.append(
                Binding(
                    "s_colors", "buffer/read_only_storage", geometry.colors, "VERTEX"
                )
            )
        if self["color_mode"] == "map":
            bindings.extend(handle_colormap(geometry, material, self))

        bindings1 = {}  # non-auto-generated bindings

        # Instanced meshes have an extra storage buffer that we add manually
        if self["instanced"]:
            bindings1[0] = Binding(
                "s_instance_infos",
                "buffer/read_only_storage",
                wobject.instance_buffer,
                "VERTEX",
            )

        # Define shader code for binding
        for i, binding in enumerate(bindings):
            self.define_binding(0, i, binding)

        return {
            "index_buffer": None,
            "vertex_buffers": {},
            "bindings": {
                0: {i: b for i, b in enumerate(bindings)},
                1: bindings1,
            },
        }

    def get_pipeline_info(self, build_args):
        wobject = build_args.wobject
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

    def get_render_info(self, build_args):
        wobject = build_args.wobject
        geometry = wobject.geometry
        material = wobject.material

        n = geometry.indices.data.size
        n_instances = 1
        if self["instanced"]:
            n_instances = wobject.instance_buffer.nitems

        m = {"auto": 0, "opaque": 1, "transparent": 2, "all": 3}

        render_mask = m[wobject.render_mask]
        if not render_mask:
            if material.opacity < 1:
                render_mask = 2
            elif self["color_mode"] == "vertex":
                if self["vertex_color_channels"] in (1, 3):
                    render_mask = 1
            elif self["color_mode"] == "map":
                if self["colormap_nchannels"] in (1, 3):
                    render_mask = 1
            elif self["color_mode"] == "normal":
                render_mask = 1
            elif self["color_mode"] == "uniform":
                render_mask = 1 if material.color[3] >= 1 else 2
            else:
                raise RuntimeError(f"Unexpected color mode {self['color_mode']}")

        return {
            "indices": (n, n_instances),
            "render_mask": render_mask,
        }

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


        @stage(vertex)
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
                let instance_info = s_instance_infos[in.instance_index];
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


@register_wgpu_render_function(Mesh, MeshFlatMaterial)
class MeshFlatShader(MeshShader):
    def __init__(self, build_args):
        super().__init__(build_args)
        self["lighting"] = "flat"


@register_wgpu_render_function(Mesh, MeshNormalMaterial)
class MeshNormalShader(MeshShader):
    def __init__(self, build_args):
        super().__init__(build_args)
        self["color_mode"] = "normal"
        self["colormap_dim"] = ""  # disable texture if there happens to be one


@register_wgpu_render_function(Mesh, MeshPhongMaterial)
class MeshPhongShader(MeshShader):
    def __init__(self, build_args):
        super().__init__(build_args)
        self["lighting"] = "phong"


@register_wgpu_render_function(Mesh, MeshNormalLinesMaterial)
class MeshNormalLinesShader(MeshShader):
    def __init__(self, build_args):
        super().__init__(build_args)
        self["color_mode"] = "uniform"
        self["lighting"] = ""
        self["wireframe"] = False

    def get_pipeline_info(self, build_args):
        d = super().get_pipeline_info(build_args)
        d["primitive_topology"] = wgpu.PrimitiveTopology.line_list
        return d

    def get_render_info(self, build_args):
        d = super().get_render_info(build_args)
        d["indices"] = build_args.wobject.geometry.positions.nitems * 2, d["indices"][1]
        return d

    def vertex_shader(self):
        return """

        struct VertexInput {
            @builtin(vertex_index) vertex_index : u32,
        };


        @stage(vertex)
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


@register_wgpu_render_function(Mesh, MeshSliceMaterial)
class MeshSliceShader(WorldObjectShader):
    """TODO: Restore this (I accidentally deleted the code)."""
