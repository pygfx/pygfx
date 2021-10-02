import wgpu  # only for flags/enums

from . import register_wgpu_render_function
from ._shadercomposer import WorldObjectShader
from ...objects import Mesh, InstancedMesh
from ...materials import (
    MeshBasicMaterial,
    MeshPhongMaterial,
    MeshFlatMaterial,
    MeshNormalMaterial,
    MeshNormalLinesMaterial,
    MeshSliceMaterial,
)
from ...resources import Buffer, Texture, TextureView
from ...utils import normals_from_vertices


@register_wgpu_render_function(Mesh, MeshBasicMaterial)
def mesh_renderer(wobject, render_info):
    """Render function capable of rendering meshes."""
    geometry = wobject.geometry
    material = wobject.material  # noqa

    # Initialize
    topology = wgpu.PrimitiveTopology.triangle_list
    shader = MeshShader(
        wobject,
        lighting="",
        need_normals=False,
        texture_dim="",
        texture_format="f32",
        instanced=False,
        climcorrection=None,
        wireframe=material.wireframe,
    )
    vs_entry_point = "vs_main"
    fs_entry_point = "fs_main"

    # We're assuming the presence of an index buffer for now
    assert getattr(geometry, "index", None)
    n = geometry.index.data.size

    # Normals. Usually it'd be given. If not, we'll calculate it from the vertices.
    if getattr(geometry, "normals", None) is not None:
        normal_buffer = geometry.normals
    else:
        normal_data = normals_from_vertices(
            geometry.positions.data, geometry.index.data
        )
        normal_buffer = Buffer(normal_data, usage="vertex|storage")

    # Init bindings 0: uniforms
    bindings0 = {
        0: ("buffer/uniform", render_info.stdinfo_uniform),
        1: ("buffer/uniform", wobject.uniform_buffer),
        2: ("buffer/uniform", material.uniform_buffer),
    }

    # todo: kunnen deze definities tegerlijk met die hierboven gedaan worden? (voor wgpu en shader) zodat ze gegarandeerd sync zijn?
    # todo: ook zoiets voor storage buffer bindings
    # todo: and then ... maybe the renderer function and shader class can be combined?
    shader.define_uniform(0, 0, "u_stdinfo", render_info.stdinfo_uniform.data.dtype)
    shader.define_uniform(0, 1, "u_wobject", wobject.uniform_buffer.data.dtype)
    shader.define_uniform(0, 2, "u_material", material.uniform_buffer.data.dtype)

    # We're using storage buffers for everything; no vertex nor index buffers.
    vertex_buffers = {}
    index_buffer = None

    # Init bindings 1: storage buffers, textures, and samplers
    bindings1 = {}
    bindings1[0] = "buffer/read_only_storage", geometry.index
    bindings1[1] = "buffer/read_only_storage", geometry.positions
    bindings1[2] = "buffer/read_only_storage", normal_buffer
    if getattr(geometry, "texcoords", None) is not None:
        bindings1[3] = "buffer/read_only_storage", geometry.texcoords

    # If a texture is applied ...
    if material.map is not None:
        if isinstance(material.map, Texture):
            raise TypeError("material.map is a Texture, but must be a TextureView")
        elif not isinstance(material.map, TextureView):
            raise TypeError("material.map must be a TextureView")
        elif getattr(geometry, "texcoords", None) is None:
            raise ValueError("material.map is present, but geometry has no texcoords")
        bindings1[4] = "sampler/filtering", material.map
        bindings1[5] = "texture/auto", material.map
        # Dimensionality
        view_dim = material.map.view_dim
        if view_dim == "1d":
            shader["texture_dim"] = "1d"
        elif view_dim == "2d":
            shader["texture_dim"] = "2d"
        elif view_dim == "3d":
            shader["texture_dim"] = "3d"
        else:
            raise ValueError("Unexpected texture dimension")
        # Texture dim matches texcoords
        if view_dim == "1d" and "x" not in geometry.texcoords.format:
            pass
        elif not geometry.texcoords.format.endswith("x" + view_dim[0]):
            raise ValueError(
                f"geometry.texcoords {geometry.texcoords.format} does not match material.map {view_dim}"
            )
        # Sampling type
        if "norm" in material.map.format or "float" in material.map.format:
            shader["texture_format"] = "f32"
            if "unorm" in material.map.format:
                shader["climcorrection"] = " * 255.0"
            elif "snorm" in material.map.format:
                shader["climcorrection"] = " * 255.0 - 128.0"
        elif "uint" in material.map.format:
            shader["texture_format"] = "u32"
        else:
            shader["texture_format"] = "i32"
        # Channels
        shader["texture_nchannels"] = material.map.texture.nchannels

    # Collect texture and sampler
    if isinstance(material, MeshNormalMaterial):
        # Special simple fragment shader
        fs_entry_point = "fs_normal_color"
        shader["texture_dim"] = ""  # disable texture if there happens to be one
        shader["need_normals"] = True
    elif isinstance(material, MeshNormalLinesMaterial):
        # Special simple vertex shader with plain fragment shader
        topology = wgpu.PrimitiveTopology.line_list
        vs_entry_point = "vs_normal_lines"
        shader["texture_dim"] = ""  # disable texture if there happens to be one
        index_buffer = None
        n = geometry.positions.nitems * 2
        shader["need_normals"] = True
    elif isinstance(material, MeshFlatMaterial):
        shader["lighting"] = "flat"
        shader["need_normals"] = True
    elif isinstance(material, MeshPhongMaterial):
        shader["lighting"] = "phong"
        shader["need_normals"] = True
    else:
        pass  # simple lighting

    # Instanced meshes have their own vertex shader
    n_instances = 1
    if isinstance(wobject, InstancedMesh):
        if vs_entry_point != "vs_main":
            raise TypeError(f"Instanced mesh does not work with {material}")
        shader["instanced"] = True
        bindings1[6] = "buffer/read_only_storage", wobject.matrices
        n_instances = wobject.matrices.nitems

    # Determine culling
    if material.side == "FRONT":
        cull_mode = wgpu.CullMode.back
    elif material.side == "BACK":
        cull_mode = wgpu.CullMode.front
    else:  # material.side == "BOTH"
        cull_mode = wgpu.CullMode.none

    # Put it together!
    wgsl = shader.generate_wgsl()
    return [
        {
            "vertex_shader": (wgsl, vs_entry_point),
            "fragment_shader": (wgsl, fs_entry_point),
            "primitive_topology": topology,
            "cull_mode": cull_mode,
            "indices": (range(n), range(n_instances)),
            "index_buffer": index_buffer,
            "vertex_buffers": vertex_buffers,
            "bindings0": bindings0,
            "bindings1": bindings1,
        }
    ]


class MeshShader(WorldObjectShader):
    def get_code(self):
        return (
            self.get_definitions()
            + self.more_definitions()
            + self.common_functions()
            + self.helpers()
            + self.vertex_shader()
            + self.fragment_shader()
        )

    def more_definitions(self):
        return """

        struct VertexInput {
            [[builtin(vertex_index)]] vertex_index : u32;
            $$ if instanced
            [[builtin(instance_index)]] instance_index : u32;
            $$ endif
        };
        struct VertexOutput {
            [[location(0)]] texcoord: vec3<f32>;
            [[location(1)]] normal: vec3<f32>;
            [[location(2)]] view: vec3<f32>;
            [[location(3)]] light: vec3<f32>;
            [[location(4)]] face_idx: vec4<f32>;
            [[location(5)]] face_coords: vec3<f32>;
            [[location(6)]] world_pos: vec3<f32>;
            $$ if wireframe
            [[location(7)]] wireframe_coords: vec3<f32>;
            $$ endif
            [[builtin(position)]] ndc_pos: vec4<f32>;
        };

        struct FragmentOutput {
            [[location(0)]] color: vec4<f32>;
            [[location(1)]] pick: vec4<i32>;
        };

        [[block]]
        struct BufferI32 {
            data: [[stride(4)]] array<i32>;
        };

        [[block]]
        struct BufferF32 {
            data: [[stride(4)]] array<f32>;
        };


        [[group(1), binding(0)]]
        var<storage,read> s_indices: BufferI32;

        [[group(1), binding(1)]]
        var<storage,read> s_pos: BufferF32;

        [[group(1), binding(2)]]
        var<storage,read> s_normal: BufferF32;

        [[group(1), binding(3)]]
        var<storage,read> s_texcoord: BufferF32;

        $$ if texture_dim
        [[group(1), binding(4)]]
        var r_sampler: sampler;
        [[group(1), binding(5)]]
        var r_tex: texture_{{ texture_dim }}<{{ texture_format }}>;
        $$ endif

        $$ if instanced
        [[block]]
        struct BufferMat4 {
            data: [[stride(64)]] array<mat4x4<f32>>;
        };
        [[group(1), binding(6)]]
        var<storage,read> s_submatrices: BufferMat4;
        $$ endif
        """

    def vertex_shader(self):
        return """

        [[stage(vertex)]]
        fn vs_main(in: VertexInput) -> VertexOutput {

            // Select what face we're at
            let index = i32(in.vertex_index);
            let face_index = index / 3;
            var sub_index = index % 3;

            // If the camera flips a dimension, it flips the face winding.
            // We can correct for this by adjusting the order (sub_index) here.
            sub_index = select(sub_index, -1 * (sub_index - 1) + 1, u_stdinfo.flipped_winding > 0);

            // Sample
            let i1 = s_indices.data[face_index * 3 + 0];
            let i2 = s_indices.data[face_index * 3 + 1];
            let i3 = s_indices.data[face_index * 3 + 2];
            var arr_i0 = array<i32, 3>(i1, i2, i3);
            let i0 = arr_i0[sub_index];

            // Get world transform
            $$ if instanced
                let submatrix: mat4x4<f32> = s_submatrices.data[in.instance_index];
                let world_transform = u_wobject.world_transform * submatrix;
            $$ else
                let world_transform = u_wobject.world_transform;
            $$ endif

            // Get vertex position
            let raw_pos = vec3<f32>(s_pos.data[i0 * 3 + 0], s_pos.data[i0 * 3 + 1], s_pos.data[i0 * 3 + 2]);
            let world_pos = world_transform * vec4<f32>(raw_pos, 1.0);
            var ndc_pos = u_stdinfo.projection_transform * u_stdinfo.cam_transform * world_pos;

            // For the wireframe we also need the ndc_pos of the other vertices of this face
            $$ if wireframe
                $$ for i in (1, 2, 3)
                    let raw_pos{{ i }} = vec3<f32>(s_pos.data[i{{ i }} * 3 + 0], s_pos.data[i{{ i }} * 3 + 1], s_pos.data[i{{ i }} * 3 + 2]);
                    let world_pos{{ i }} = world_transform * vec4<f32>(raw_pos{{ i }}, 1.0);
                    let ndc_pos{{ i }} = u_stdinfo.projection_transform * u_stdinfo.cam_transform * world_pos{{ i }};
                $$ endfor
                let depth_offset = -0.0001;  // to put the mesh slice atop a mesh
                ndc_pos.z = ndc_pos.z + depth_offset;
            $$ endif

            // Prepare output
            var out: VertexOutput;

            // Set position
            out.world_pos =world_pos.xyz / world_pos.w;
            out.ndc_pos = vec4<f32>(ndc_pos.xyz, ndc_pos.w);

            // Set texture coords
            $$ if texture_dim == '1d'
            out.texcoord =vec3<f32>(s_texcoord.data[i0], 0.0, 0.0);
            $$ elif texture_dim == '2d'
            out.texcoord = vec3<f32>(s_texcoord.data[i0 * 2 + 0], s_texcoord.data[i0 * 2 + 1], 0.0);
            $$ elif texture_dim == '3d'
            out.texcoord = vec3<f32>(s_texcoord.data[i0 * 3 + 0], s_texcoord.data[i0 * 3 + 1], s_texcoord.data[i0 * 3 + 2]);
            $$ endif

            // Set the normal
            $$ if need_normals
                let raw_normal = vec3<f32>(s_normal.data[i0 * 3 + 0], s_normal.data[i0 * 3 + 1], s_normal.data[i0 * 3 + 2]);
                let world_pos_n = world_transform * vec4<f32>(raw_pos + raw_normal, 1.0);
                let world_normal = normalize(world_pos_n - world_pos).xyz;
                out.normal = world_normal;
            $$ endif

            // Vectors for lighting, all in world coordinates
            $$ if lighting
                let view_vec = normalize(ndc_to_world_pos(vec4<f32>(0.0, 0.0, 1.0, 1.0)));
                out.view = view_vec;
                out.light = view_vec;
            $$ endif

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
                out.wireframe_coords = arr_wireframe_coords[sub_index];  // in logical pixels
            $$ endif

            // Set varyings for picking. We store the face_index, and 3 weights
            // that indicate how close the fragment is to each vertex (barycentric
            // coordinates). This allows the selection of the nearest vertex or
            // edge. Note that integers larger than about 4M loose too much
            // precision when passed as a varyings (on my machine). We therefore
            // encode them in two values.
            let d = 10000;
            $$ if instanced
                let inst_index = i32(in.instance_index);
            $$ else
                let inst_index = 0;
            $$ endif
            out.face_idx = vec4<f32>(
                f32(inst_index / d), f32(inst_index % d), f32(face_index / d), f32(face_index % d)
            );
            var arr_face_coords = array<vec3<f32>, 3>(
                vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(0.0, 0.0, 1.0)
            );
            out.face_coords = arr_face_coords[sub_index];

            return out;
        }


        [[stage(vertex)]]
        fn vs_normal_lines(in: VertexInput) -> VertexOutput {
            let index = i32(in.vertex_index);
            let r = index % 2;
            let i0 = (index - r) / 2;

            let raw_pos = vec3<f32>(s_pos.data[i0 * 3 + 0], s_pos.data[i0 * 3 + 1], s_pos.data[i0 * 3 + 2]);
            let raw_normal = vec3<f32>(
                s_normal.data[i0 * 3 + 0], s_normal.data[i0 * 3 + 1], s_normal.data[i0 * 3 + 2]
            );

            let world_pos1 = u_wobject.world_transform * vec4<f32>(raw_pos, 1.0);
            let world_pos2 = u_wobject.world_transform * vec4<f32>(raw_pos + raw_normal, 1.0);

            // The normal is sized in world coordinates
            let world_normal = normalize(world_pos2 - world_pos1);

            let amplitude = 1.0;
            let world_pos = world_pos1 + f32(r) * world_normal * amplitude;
            let ndc_pos = u_stdinfo.projection_transform * u_stdinfo.cam_transform * world_pos;

            var out: VertexOutput;
            out.world_pos = vec3<f32>(world_pos.xyz / world_pos.w);
            out.ndc_pos = ndc_pos;
            return out;
        }
        """

    def fragment_shader(self):
        return """

        [[stage(fragment)]]
        fn fs_main(in: VertexOutput, [[builtin(front_facing)]] is_front: bool) -> FragmentOutput {
            var out: FragmentOutput;
            var color_value: vec4<f32>;

            $$ if texture_dim
                $$ if texture_dim == '1d'
                    $$ if texture_format == 'f32'
                        color_value = textureSample(r_tex, r_sampler, in.texcoord.x);
                    $$ else
                        let texcoords_dim = f32(textureDimensions(r_tex);
                        let texcoords_u = i32(in.texcoord.x * texcoords_dim % texcoords_dim);
                        color_value = vec4<f32>(textureLoad(r_tex, texcoords_u, 0));
                    $$ endif
                $$ elif texture_dim == '2d'
                    $$ if texture_format == 'f32'
                        color_value = textureSample(r_tex, r_sampler, in.texcoord.xy);
                    $$ else
                        let texcoords_dim = vec2<f32>(textureDimensions(r_tex));
                        let texcoords_u = vec2<i32>(in.texcoord.xy * texcoords_dim % texcoords_dim);
                        color_value = vec4<f32>(textureLoad(r_tex, texcoords_u, 0));
                    $$ endif
                $$ elif texture_dim == '3d'
                    $$ if texture_format == 'f32'
                        color_value = textureSample(r_tex, r_sampler, in.texcoord.xyz);
                    $$ else
                        let texcoords_dim = vec3<f32>(textureDimensions(r_tex));
                        let texcoords_u = vec3<i32>(in.texcoord.xyz * texcoords_dim % texcoords_dim);
                        color_value = vec4<f32>(textureLoad(r_tex, texcoords_u, 0));
                    $$ endif
                $$ endif

                $$ if climcorrection
                    color_value = vec4<f32>(color_value.rgb {{ climcorrection }}, color_value.a);
                $$ endif
                $$ if texture_nchannels == 1
                    color_value = vec4<f32>(color_value.rrr, 1.0);
                $$ elif texture_nchannels == 2
                    color_value = vec4<f32>(color_value.rrr, color_value.g);
                $$ endif
                let albeido = (color_value.rgb - u_material.clim[0]) / (u_material.clim[1] - u_material.clim[0]);
            $$ else
                // Just a simple color
                color_value = u_material.color;
                let albeido: vec3<f32> = color_value.rgb;
            $$ endif

            // Lighting
            $$ if lighting
                let lit_color = lighting_{{ lighting }}(is_front, in.world_pos, in.normal, in.light, in.view, albeido);
            $$ else
                let lit_color = albeido;
            $$ endif

            // Final color
            out.color = vec4<f32>(lit_color, color_value.a);

            // Picking
            let face_id = vec2<i32>(in.face_idx.xz * 10000.0 + in.face_idx.yw + 0.5);  // inst+face
            let w8 = vec3<i32>(in.face_coords.xyz * 255.0 + 0.5);
            out.pick = vec4<i32>(u_wobject.id, face_id, w8.x * 65536 + w8.y * 256 + w8.z);

            out.color.a = out.color.a * u_material.opacity;

            $$ if wireframe
                let distance_from_edge = min(in.wireframe_coords.x, min(in.wireframe_coords.y, in.wireframe_coords.z));
                if (distance_from_edge > 0.5 * u_material.wireframe) {
                    discard;
                }
            $$ endif

            apply_clipping_planes(in.world_pos);
            return out;
        }


        [[stage(fragment)]]
        fn fs_normal_color(in: VertexOutput) -> FragmentOutput {
            var out: FragmentOutput;

            // Color
            let v = normalize(in.normal) * 0.5 + 0.5;
            out.color = vec4<f32>(v, 1.0);

            // Picking
            let face_id = vec2<i32>(in.face_idx.xz * 10000.0 + in.face_idx.yw + 0.5);  // inst+face
            let w8 = vec3<i32>(in.face_coords.xyz * 255.0 + 0.5);
            out.pick = vec4<i32>(u_wobject.id, face_id, w8.x * 65536 + w8.y * 256 + w8.z);

            apply_clipping_planes(in.world_pos);
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
            // todo: allow configuring material specularity
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
def meshslice_renderer(wobject, render_info):
    """Render function capable of rendering mesh slices."""

    geometry = wobject.geometry
    material = wobject.material  # noqa

    # Initialize
    topology = wgpu.PrimitiveTopology.triangle_list
    shader = MeshSliceShader(wobject)
    vs_entry_point = "vs_main"
    fs_entry_point = "fs_main"

    # We're assuming the presence of an index buffer for now
    assert getattr(geometry, "index", None)
    n = (geometry.index.data.size // 3) * 6
    n_instances = 1

    # Init bindings 0: uniforms
    bindings0 = {
        0: ("buffer/uniform", render_info.stdinfo_uniform),
        1: ("buffer/uniform", wobject.uniform_buffer),
        2: ("buffer/uniform", material.uniform_buffer),
    }

    shader.define_uniform(0, 0, "u_stdinfo", render_info.stdinfo_uniform.data.dtype)
    shader.define_uniform(0, 1, "u_wobject", wobject.uniform_buffer.data.dtype)
    shader.define_uniform(0, 2, "u_material", material.uniform_buffer.data.dtype)

    # Init bindings 1: storage buffers, textures, and samplers
    bindings1 = {}
    bindings1[0] = "buffer/read_only_storage", geometry.index
    bindings1[1] = "buffer/read_only_storage", geometry.positions

    # Put it together!
    wgsl = shader.generate_wgsl()
    return [
        {
            "vertex_shader": (wgsl, vs_entry_point),
            "fragment_shader": (wgsl, fs_entry_point),
            "primitive_topology": topology,
            "indices": (range(n), range(n_instances)),
            "index_buffer": None,
            "vertex_buffers": {},
            "bindings0": bindings0,
            "bindings1": bindings1,
        }
    ]


class MeshSliceShader(WorldObjectShader):
    def get_code(self):
        return (
            self.get_definitions()
            + self.more_definitions()
            + self.common_functions()
            + self.vertex_shader()
            + self.fragment_shader()
        )

    def more_definitions(self):
        return """

        struct VertexInput {
            [[builtin(vertex_index)]] vertex_index : u32;
        };
        struct VertexOutput {
            [[location(0)]] dist2center: vec2<f32>;
            [[location(1)]] segment_length: f32;
            [[location(2)]] segment_width: f32;
            [[location(3)]] face_idx: vec4<f32>;
            [[location(4)]] face_coords: vec3<f32>;
            [[location(5)]] world_pos: vec3<f32>;
            [[builtin(position)]] ndc_pos: vec4<f32>;
        };

        struct FragmentOutput {
            [[location(0)]] color: vec4<f32>;
            [[location(1)]] pick: vec4<i32>;
        };

        [[block]]
        struct BufferI32 {
            data: [[stride(4)]] array<i32>;
        };

        [[block]]
        struct BufferF32 {
            data: [[stride(4)]] array<f32>;
        };

        [[group(1), binding(0)]]
        var<storage,read> s_indices: BufferI32;

        [[group(1), binding(1)]]
        var<storage,read> s_pos: BufferF32;
        """

    def vertex_shader(self):
        return """
        [[stage(vertex)]]
        fn vs_main(in: VertexInput) -> VertexOutput {

            // This vertex shader uses VertexId and storage buffers instead of
            // vertex buffers. It creates 6 vertices for each face in the mesh,
            // drawn with triangle-list. For the faces that cross the plane, we
            // draw a (thick) line segment with round caps (we need 6 verts for that).
            // Other faces become degenerate triangles.

            var out: VertexOutput;

            let screen_factor = u_stdinfo.logical_size.xy / 2.0;
            let l2p = u_stdinfo.physical_size.x / u_stdinfo.logical_size.x;
            let line_width = u_material.thickness;  // in logical pixels

            // Get the face index, and sample the vertex indices
            let index = i32(in.vertex_index);
            let segment_index = index % 6;
            let face_index = (index - segment_index) / 6;
            let i1 = s_indices.data[face_index * 3 + 0];
            let i2 = s_indices.data[face_index * 3 + 1];
            let i3 = s_indices.data[face_index * 3 + 2];

            // Vertex positions of this face, in local object coordinates
            let pos1a = vec3<f32>(s_pos.data[i1 * 3 + 0], s_pos.data[i1 * 3 + 1], s_pos.data[i1 * 3 + 2]);
            let pos2a = vec3<f32>(s_pos.data[i2 * 3 + 0], s_pos.data[i2 * 3 + 1], s_pos.data[i2 * 3 + 2]);
            let pos3a = vec3<f32>(s_pos.data[i3 * 3 + 0], s_pos.data[i3 * 3 + 1], s_pos.data[i3 * 3 + 2]);
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
                // todo: omitting this step diminish the line width with distance, but it that the way?
                npos_a = npos_a / npos_a.w;
                npos_b = npos_b / npos_b.w;

                // And to logical pixel coordinates (don't worry about offset)
                let ppos_a = npos_a.xy * screen_factor;
                let ppos_b = npos_b.xy * screen_factor;

                // Get the segment vector, its length, and how much it scales because of line width
                let v0 = ppos_b - ppos_a;
                segment_length = length(v0);  // in logical pixels;
                let segment_factor = (segment_length + line_width) / segment_length;

                // Get the (orthogonal) unit vectors that span the segment
                let v1 = normalize(v0);
                let v2 = vec2<f32>(v1.y, -v1.x);

                // Get the vector, in local logical pixels for the segment's square
                let pvec_local = 0.5 * vec2<f32>(segment_length + line_width, line_width);

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
                out.face_idx = vec4<f32>(0.0, 0.0, f32(face_index / 10000), f32(face_index % 10000));
                let mixval = the_vec.x * 0.5 + 0.5;
                out.face_coords = mix(fw_a, fw_b, vec3<f32>(mixval, mixval, mixval));
            }

            // Shader output
            out.world_pos = ndc_to_world_pos(the_pos);
            out.ndc_pos = the_pos;
            out.dist2center = the_coord * l2p;
            out.segment_length = segment_length * l2p;
            out.segment_width = line_width * l2p;
            return out;
        }
        """

    def fragment_shader(self):
        return """

        [[stage(fragment)]]
        fn fs_main(in: VertexOutput) -> FragmentOutput {
            var out: FragmentOutput;

            // Discart fragments that are too far from the centerline. This makes round caps.
            // Note that we operate in physical pixels here.
            let distx = max(0.0, abs(in.dist2center.x) - 0.5 * in.segment_length);
            let dist = length(vec2<f32>(distx, in.dist2center.y));
            if (dist > in.segment_width * 0.5) {
                discard;
            }

            // No aa. This is something we need to decide on. See line renderer.
            let alpha = 1.0;

            // Set color
            let color = u_material.color;
            out.color = vec4<f32>(color.rgb, min(1.0, color.a) * alpha);

            // Picking
            let face_id = vec2<i32>(in.face_idx.xz * 10000.0 + in.face_idx.yw + 0.5);  // inst+face
            let w8 = vec3<i32>(in.face_coords.xyz * 255.0 + 0.5);
            out.pick = vec4<i32>(u_wobject.id, face_id, w8.x * 65536 + w8.y * 256 + w8.z);

            out.color.a = out.color.a * u_material.opacity;
            apply_clipping_planes(in.world_pos);
            return out;
        }
        """
