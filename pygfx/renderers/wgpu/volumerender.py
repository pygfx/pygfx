import wgpu  # only for flags/enums

from . import register_wgpu_render_function
from ._shadercomposer import BaseShader
from ...objects import Volume
from ...materials import VolumeSliceMaterial
from ...resources import Texture, TextureView


@register_wgpu_render_function(Volume, VolumeSliceMaterial)
def volume_slice_renderer(wobject, render_info):
    """Render function capable of rendering volumes."""

    geometry = wobject.geometry
    material = wobject.material  # noqa
    shader = VolumeSliceShader(climcorrection=False)

    shader.define_uniform(0, 0, "u_stdinfo", render_info.stdinfo_uniform.data.dtype)
    shader.define_uniform(0, 1, "u_wobject", wobject.uniform_buffer.data.dtype)
    shader.define_uniform(0, 2, "u_material", material.uniform_buffer.data.dtype)

    bindings0 = {
        0: ("buffer/uniform", render_info.stdinfo_uniform),
        1: ("buffer/uniform", wobject.uniform_buffer),
        2: ("buffer/uniform", material.uniform_buffer),
    }

    topology = wgpu.PrimitiveTopology.triangle_list
    n = 12

    # Collect texture and sampler
    if material.map is None:
        raise ValueError("VolumeSliceMaterial must have a texture map.")
    else:
        if isinstance(material.map, TextureView):
            view = material.map
        elif isinstance(material.map, Texture):
            view = material.map.get_view(filter="linear")
        else:
            raise TypeError("material.map must be a TextureView")
        if view.view_dim.lower() != "3d":
            raise TypeError("material.map must a 3D texture (view)")
        elif getattr(geometry, "texcoords", None) is None:
            raise ValueError("With VolumeSliceMaterial, geometry needs texcoords")
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
        if material.map.format.startswith("rgb"):  # rgb maps to rgba
            shader["texture_color"] = True
        elif material.map.format.startswith("r"):
            shader["texture_color"] = False
        else:
            raise ValueError("Unexpected texture format")

    bindings1 = {
        0: ("sampler/filtering", view),
        1: ("texture/auto", view),
        2: ("buffer/read_only_storage", geometry.positions),
        3: ("buffer/read_only_storage", geometry.texcoords),
    }

    # Put it together!
    wgsl = shader.generate_wgsl()
    return [
        {
            "vertex_shader": (wgsl, "vs_main"),
            "fragment_shader": (wgsl, "fs_main"),
            "primitive_topology": topology,
            "indices": (range(n), range(1)),
            "vertex_buffers": {},
            "bindings0": bindings0,
            "bindings1": bindings1,
        }
    ]


class VolumeSliceShader(BaseShader):
    def get_code(self):
        return (
            self.get_definitions()
            + self.more_definitions()
            + self.vertex_shader()
            + self.fragment_shader()
        )

    def more_definitions(self):
        return """

        struct VertexInput {
            [[builtin(vertex_index)]] index : u32;
        };
        struct VertexOutput {
            [[location(0)]] texcoord: vec3<f32>;
            [[builtin(position)]] pos: vec4<f32>;
        };

        struct FragmentOutput {
            [[location(0)]] color: vec4<f32>;
            [[location(1)]] pick: vec4<i32>;
        };

        [[block]]
        struct BufferF32 {
            data: [[stride(4)]] array<f32>;
        };


        [[group(1), binding(0)]]
        var r_sampler: sampler;

        [[group(1), binding(1)]]
        var r_tex: texture_3d<{{ texture_format }}>;

        [[group(1), binding(2)]]
        var<storage,read> s_positions: BufferF32;

        [[group(1), binding(3)]]
        var<storage,read> s_texcoords: BufferF32;
        """

    def vertex_shader(self):
        return """

        [[stage(vertex)]]
        fn vs_main(in: VertexInput) -> VertexOutput {
            var out: VertexOutput;

            // We're assuming a box geometry, using the same layout as a simple
            // ThreeJS BoxBufferGeometry. And we're only using the first eight
            // vertices. These are laid out like this:
            //
            //   Vertices       Planes (right, left, back, front, top, bottom)
            //                            0      1    2      3     4     5
            //
            //    5----0        0: 0231        +----+
            //   /|   /|        1: 7546       /|24 /|
            //  7----2 |        2: 5014      +----+ |0
            //  | 4--|-1        3: 2763     1| +--|-+
            //  |/   |/         4: 0572      |/35 |/
            //  6----3          5: 3641      +----+


            let plane = u_material.plane.xyzw;  // ax + by + cz + d
            let n = plane.xyz;

            // Define edges (using vertex indices), and their matching plane
            // indices (each edge touches two planes). Note that these need to
            // match the above figure, and that needs to match with the actual
            // BoxGeometry implementation!
            var edges = array<vec2<i32>,12>(
                vec2<i32>(0, 2), vec2<i32>(2, 3), vec2<i32>(3, 1), vec2<i32>(1, 0),
                vec2<i32>(4, 6), vec2<i32>(6, 7), vec2<i32>(7, 5), vec2<i32>(5, 4),
                vec2<i32>(5, 0), vec2<i32>(1, 4), vec2<i32>(2, 7), vec2<i32>(6, 3),
            );
            var ed2pl = array<vec2<i32>,12>(
                vec2<i32>(0, 4), vec2<i32>(0, 3), vec2<i32>(0, 5), vec2<i32>(0, 2),
                vec2<i32>(1, 5), vec2<i32>(1, 3), vec2<i32>(1, 4), vec2<i32>(1, 2),
                vec2<i32>(2, 4), vec2<i32>(2, 5), vec2<i32>(3, 4), vec2<i32>(3, 5),
            );

            // Init intersection info
            var intersect_flags = array<i32,12>(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
            var intersect_positions = array<vec3<f32>,12>(
                vec3<f32>(0.0, 0.0, 0.0), vec3<f32>(0.0, 0.0, 0.0), vec3<f32>(0.0, 0.0, 0.0),
                vec3<f32>(0.0, 0.0, 0.0), vec3<f32>(0.0, 0.0, 0.0), vec3<f32>(0.0, 0.0, 0.0),
                vec3<f32>(0.0, 0.0, 0.0), vec3<f32>(0.0, 0.0, 0.0), vec3<f32>(0.0, 0.0, 0.0),
                vec3<f32>(0.0, 0.0, 0.0), vec3<f32>(0.0, 0.0, 0.0), vec3<f32>(0.0, 0.0, 0.0),
            );
            var intersect_texcoords = array<vec3<f32>,12>(
                vec3<f32>(0.0, 0.0, 0.0), vec3<f32>(0.0, 0.0, 0.0), vec3<f32>(0.0, 0.0, 0.0),
                vec3<f32>(0.0, 0.0, 0.0), vec3<f32>(0.0, 0.0, 0.0), vec3<f32>(0.0, 0.0, 0.0),
                vec3<f32>(0.0, 0.0, 0.0), vec3<f32>(0.0, 0.0, 0.0), vec3<f32>(0.0, 0.0, 0.0),
                vec3<f32>(0.0, 0.0, 0.0), vec3<f32>(0.0, 0.0, 0.0), vec3<f32>(0.0, 0.0, 0.0),
            );

            // Intersect the 12 edges
            for (var i:i32=0; i<12; i=i+1) {
                let edge = edges[i];
                let p1_raw = vec3<f32>(
                    s_positions.data[edge[0] * 3],
                    s_positions.data[edge[0] * 3 + 1],
                    s_positions.data[edge[0] * 3 + 2],
                );
                let p2_raw = vec3<f32>(
                    s_positions.data[edge[1] * 3],
                    s_positions.data[edge[1] * 3 + 1],
                    s_positions.data[edge[1] * 3 + 2],
                );
                let p1_ = u_wobject.world_transform * vec4<f32>(p1_raw, 1.0);
                let p2_ = u_wobject.world_transform * vec4<f32>(p2_raw, 1.0);
                let p1 = p1_.xyz / p1_.w;
                let p2 = p2_.xyz / p2_.w;
                let tc1 = vec3<f32>(
                    s_texcoords.data[edge[0] * 3],
                    s_texcoords.data[edge[0] * 3 + 1],
                    s_texcoords.data[edge[0] * 3 + 2],
                );
                let tc2 = vec3<f32>(
                    s_texcoords.data[edge[1] * 3],
                    s_texcoords.data[edge[1] * 3 + 1],
                    s_texcoords.data[edge[1] * 3 + 2],
                );
                let u = p2 - p1;
                let t = -(plane.x * p1.x + plane.y * p1.y + plane.z * p1.z + plane.w) / dot(n, u);
                let intersects:bool = t > 0.0 && t < 1.0;
                intersect_flags[i] = select(0, 1, intersects);
                intersect_positions[i] = mix(p1, p2, vec3<f32>(t, t, t));
                intersect_texcoords[i] = mix(tc1, tc2, vec3<f32>(t, t, t));
            }

            // Init six vertices
            var vertices = array<vec3<f32>,6>(
                vec3<f32>(0.0, 0.0, 0.0),
                vec3<f32>(0.0, 0.0, 0.0),
                vec3<f32>(0.0, 0.0, 0.0),
                vec3<f32>(0.0, 0.0, 0.0),
                vec3<f32>(0.0, 0.0, 0.0),
                vec3<f32>(0.0, 0.0, 0.0),
            );
            var texcoords = array<vec3<f32>,6>(
                vec3<f32>(0.0, 0.0, 0.0),
                vec3<f32>(0.0, 0.0, 0.0),
                vec3<f32>(0.0, 0.0, 0.0),
                vec3<f32>(0.0, 0.0, 0.0),
                vec3<f32>(0.0, 0.0, 0.0),
                vec3<f32>(0.0, 0.0, 0.0),
            );

            // Find first intersection point. This can be any valid intersection.
            // In ed2pl[i][0], the 0 could also be a one. It would mean that we'd
            // move around the box in the other direction.
            var plane_index: i32 = 0;
            for (var i:i32=0; i<12; i=i+1) {
                if (intersect_flags[i] == 1) {
                    plane_index = ed2pl[i][0];
                    vertices[0] = intersect_positions[i];
                    texcoords[0] = intersect_texcoords[i];
                    break;
                }
            }

            // From there take (at most) 5 steps
            let i_start: i32 = i;
            var i_last: i32 = i;
            var max_iter: i32 = 6;
            for (var iter:i32=1; iter<max_iter; iter=iter+1) {
                for (var i:i32=0; i<12; i=i+1) {
                    if (i != i_last && intersect_flags[i] == 1) {
                        if (ed2pl[i][0] == plane_index) {
                            vertices[iter] = intersect_positions[i];
                            texcoords[iter] = intersect_texcoords[i];
                            plane_index = ed2pl[i][1];
                            i_last = i;
                            break;
                        } elseif (ed2pl[i][1] == plane_index) {
                            vertices[iter] = intersect_positions[i];
                            texcoords[iter] = intersect_texcoords[i];
                            plane_index = ed2pl[i][0];
                            i_last = i;
                            break;
                        }
                    }
                }
                if (i_last == i_start) {
                    max_iter = iter;
                    break;
                }
            }

            // Make the rest degenerate triangles
            for (var i:i32=max_iter; i<6; i=i+1) {
                vertices[i] = vertices[0];
            }

            // Now select the current vertex. We mimic a triangle fan with a triangle list.
            // This works whether the number of vertices/intersections is 3, 4, 5, and 6.
            let index = i32(in.index);
            var indexmap = array<i32,12>(0, 1, 2, 0, 2, 3, 0, 3, 4, 0, 4, 5);
            let world_pos = vertices[ indexmap[index] ];
            let ndc_pos = u_stdinfo.projection_transform * u_stdinfo.cam_transform * vec4<f32>(world_pos, 1.0);
            out.pos = ndc_pos;
            out.texcoord = texcoords[ indexmap[index] ];

            return out;
        }
        """

    def fragment_shader(self):
        return """

        [[stage(fragment)]]
        fn fs_main(in: VertexOutput) -> FragmentOutput {
            var out: FragmentOutput;
            var color_value: vec4<f32>;

            $$ if texture_format == 'f32'
                color_value = textureSample(r_tex, r_sampler, in.texcoord.xyz);
            $$ else
                let texcoords_dim = vec3<f32>(textureDimensions(r_tex));
                let texcoords_u = vec3<i32>(in.texcoord.xyz * texcoords_dim % texcoords_dim);
                color_value = vec4<f32>(textureLoad(r_tex, texcoords_u, 0));
            $$ endif

            $$ if climcorrection
                color_value = vec4<f32>(color_value.rgb {{ climcorrection }}, color_value.a);
            $$ endif
            $$ if not texture_color
                color_value = color_value.rrra;
            $$ endif
            let albeido = (color_value.rgb - u_material.clim[0]) / (u_material.clim[1] - u_material.clim[0]);

            out.color = vec4<f32>(albeido, color_value.a);
            out.pick = vec4<i32>(u_wobject.id, vec3<i32>(in.texcoord * 1048576.0 + 0.5));
            return out;
        }
        """
