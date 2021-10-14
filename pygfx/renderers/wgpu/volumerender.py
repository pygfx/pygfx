import wgpu  # only for flags/enums

from . import register_wgpu_render_function
from ._shadercomposer import Binding, WorldObjectShader
from ._conv import to_texture_format
from ...objects import Volume
from ...materials import VolumeSliceMaterial, VolumeRayMaterial, VolumeMipMaterial
from ...resources import Texture, TextureView


@register_wgpu_render_function(Volume, VolumeSliceMaterial)
def volume_slice_renderer(wobject, render_info):
    """Render function capable of rendering volumes."""

    geometry = wobject.geometry
    material = wobject.material  # noqa
    shader = VolumeSliceShader(wobject, climcorrection=False)

    bindings = {}

    bindings[0] = Binding("u_stdinfo", "buffer/uniform", render_info.stdinfo_uniform)
    bindings[1] = Binding("u_wobject", "buffer/uniform", wobject.uniform_buffer)
    bindings[2] = Binding("u_material", "buffer/uniform", material.uniform_buffer)

    topology = wgpu.PrimitiveTopology.triangle_list
    n = 12

    # Collect texture and sampler
    if geometry.grid is None:
        raise ValueError("Volume.geometry must have a grid (texture).")
    else:
        if isinstance(geometry.grid, TextureView):
            view = geometry.grid
        elif isinstance(geometry.grid, Texture):
            view = geometry.grid.get_view(filter="linear")
        else:
            raise TypeError("Volume.geometry.grid must be a Texture or TextureView")
        if view.view_dim.lower() != "3d":
            raise TypeError("Volume.geometry.grid must a 3D texture (view)")
        elif getattr(geometry, "texcoords", None) is None:
            raise ValueError("Volume.geometry needs texcoords")
        # Sampling type
        fmt = to_texture_format(geometry.grid.format)
        if "norm" in fmt or "float" in fmt:
            shader["texture_format"] = "f32"
            if "unorm" in fmt:
                shader["climcorrection"] = " * 255.0"
            elif "snorm" in fmt:
                shader["climcorrection"] = " * 255.0 - 128.0"
        elif "uint" in fmt:
            shader["texture_format"] = "u32"
        else:
            shader["texture_format"] = "i32"
        # Channels
        shader["texture_nchannels"] = len(fmt) - len(fmt.lstrip("rgba"))

    bindings[3] = Binding(
        "s_positions", "buffer/read_only_storage", geometry.positions, "VERTEX"
    )
    bindings[4] = Binding(
        "s_texcoords", "buffer/read_only_storage", geometry.texcoords, "VERTEX"
    )
    bindings[5] = Binding("r_sampler", "sampler/filtering", view, "FRAGMENT")
    bindings[6] = Binding("r_tex", "texture/auto", view, "FRAGMENT")

    # Let the shader generate code for our bindings
    for i, binding in bindings.items():
        shader.define_binding(0, i, binding)

    # Put it together!
    wgsl = shader.generate_wgsl()
    return [
        {
            "vertex_shader": (wgsl, "vs_main"),
            "fragment_shader": (wgsl, "fs_main"),
            "primitive_topology": topology,
            "indices": (range(n), range(1)),
            "vertex_buffers": {},
            "bindings0": bindings,
        }
    ]


class VolumeSliceShader(WorldObjectShader):
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
            [[location(0)]] texcoord: vec3<f32>;
            [[location(1)]] world_pos: vec3<f32>;
            [[builtin(position)]] ndc_pos: vec4<f32>;
        };

        struct FragmentOutput {
            [[location(0)]] color: vec4<f32>;
            [[location(1)]] pick: vec4<i32>;
        };
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
                let p1_p = u_wobject.world_transform * vec4<f32>(p1_raw, 1.0);
                let p2_p = u_wobject.world_transform * vec4<f32>(p2_raw, 1.0);
                let p1 = p1_p.xyz / p1_p.w;
                let p2 = p2_p.xyz / p2_p.w;
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
            let index = i32(in.vertex_index);
            var indexmap = array<i32,12>(0, 1, 2, 0, 2, 3, 0, 3, 4, 0, 4, 5);
            let world_pos = vertices[ indexmap[index] ];
            let ndc_pos = u_stdinfo.projection_transform * u_stdinfo.cam_transform * vec4<f32>(world_pos, 1.0);
            out.world_pos = world_pos;
            out.ndc_pos = ndc_pos;
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
            $$ if texture_nchannels == 1
                color_value = vec4<f32>(color_value.rrr, 1.0);
            $$ elif texture_nchannels == 2
                color_value = vec4<f32>(color_value.rrr, color_value.g);
            $$ endif
            let albeido = (color_value.rgb - u_material.clim[0]) / (u_material.clim[1] - u_material.clim[0]);

            out.color = vec4<f32>(albeido, color_value.a);
            out.pick = vec4<i32>(u_wobject.id, vec3<i32>(in.texcoord * 1048576.0 + 0.5));

            out.color.a = out.color.a * u_material.opacity;
            apply_clipping_planes(in.world_pos);
            return out;
        }
        """


@register_wgpu_render_function(Volume, VolumeRayMaterial)
def volume_ray_renderer(wobject, render_info):
    """Render function capable of rendering volumes."""

    geometry = wobject.geometry
    material = wobject.material  # noqa
    shader = VolumeRayShader(wobject, climcorrection=False)

    bindings = {}

    bindings[0] = Binding("u_stdinfo", "buffer/uniform", render_info.stdinfo_uniform)
    bindings[1] = Binding("u_wobject", "buffer/uniform", wobject.uniform_buffer)
    bindings[2] = Binding("u_material", "buffer/uniform", material.uniform_buffer)

    assert getattr(geometry, "index", None)
    bindings[3] = Binding(
        "s_indices", "buffer/read_only_storage", geometry.index, "VERTEX"
    )
    topology = wgpu.PrimitiveTopology.triangle_list
    n = geometry.index.data.size

    # Collect texture and sampler
    if geometry.grid is None:
        raise ValueError("Volume.geometry must have a grid (texture).")
    else:
        if isinstance(geometry.grid, TextureView):
            view = geometry.grid
        elif isinstance(geometry.grid, Texture):
            view = geometry.grid.get_view(filter="linear")
        else:
            raise TypeError("Volume.geometry.grid must be a Texture or TextureView")
        if view.view_dim.lower() != "3d":
            raise TypeError("Volume.geometry.grid must a 3D texture (view)")
        elif getattr(geometry, "texcoords", None) is None:
            raise ValueError("Volume.geometry needs texcoords")
        # Sampling type
        fmt = to_texture_format(geometry.grid.format)
        if "norm" in fmt or "float" in fmt:
            shader["texture_format"] = "f32"
            if "unorm" in fmt:
                shader["climcorrection"] = " * 255.0"
            elif "snorm" in fmt:
                shader["climcorrection"] = " * 255.0 - 128.0"
        elif "uint" in fmt:
            shader["texture_format"] = "u32"
        else:
            shader["texture_format"] = "i32"
        # Channels
        shader["texture_nchannels"] = len(fmt) - len(fmt.lstrip("rgba"))

    bindings[4] = Binding(
        "s_positions", "buffer/read_only_storage", geometry.positions, "VERTEX"
    )
    bindings[5] = Binding(
        "s_texcoords", "buffer/read_only_storage", geometry.texcoords, "VERTEX"
    )
    bindings[6] = Binding("r_sampler", "sampler/filtering", view, "FRAGMENT")
    bindings[7] = Binding("r_tex", "texture/auto", view, "FRAGMENT")

    # Let the shader generate code for our bindings
    for i, binding in bindings.items():
        shader.define_binding(0, i, binding)

    # Put it together!
    wgsl = shader.generate_wgsl()
    return [
        {
            "vertex_shader": (wgsl, "vs_main"),
            "fragment_shader": (wgsl, "fs_main"),
            "primitive_topology": topology,
            "indices": (range(n), range(1)),
            "vertex_buffers": {},
            "bindings0": bindings,
        }
    ]


class VolumeRayShader(WorldObjectShader):
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
            [[location(0)]] texcoord: vec3<f32>;
            [[location(1)]] world_pos: vec4<f32>;
            [[location(2)]] near_pos: vec4<f32>;
            [[location(3)]] far_pos: vec4<f32>;
            [[builtin(position)]] position: vec4<f32>;
        };

        struct FragmentOutput {
            [[location(0)]] color: vec4<f32>;
            [[location(1)]] pick: vec4<i32>;
        };
        """

    def vertex_shader(self):
        return """

        [[stage(vertex)]]
        fn vs_main(in: VertexInput) -> VertexOutput {
            var out: VertexOutput;

            // Select what face we're at
            let index = i32(in.vertex_index);
            let face_index = index / 3;
            var sub_index = index % 3;

            // Sample index
            let i1 = i32(s_indices.data[face_index * 3 + 0]);
            let i2 = i32(s_indices.data[face_index * 3 + 1]);
            let i3 = i32(s_indices.data[face_index * 3 + 2]);
            var arr_i0 = array<i32, 3>(i1, i2, i3);
            let i0 = arr_i0[sub_index];

            // Get world pos
            let raw_pos = vec3<f32>(s_positions.data[i0 * 3 + 0], s_positions.data[i0 * 3 + 1], s_positions.data[i0 * 3 + 2]);
            let world_pos = u_wobject.world_transform * vec4<f32>(raw_pos, 1.0);

            // Turn to NDC
            let ndc_pos = u_stdinfo.projection_transform * u_stdinfo.cam_transform * world_pos;

            // Step forward and backward, then map back
            let ndc_pos1 = vec4<f32>(ndc_pos.xy, -ndc_pos.w, ndc_pos.w);
            let ndc_pos2 = vec4<f32>(ndc_pos.xy, ndc_pos.w, ndc_pos.w);

            let ndc_to_world = u_stdinfo.cam_transform_inv * u_stdinfo.projection_transform_inv;

            out.world_pos = world_pos;
            out.position = ndc_pos;
            out.texcoord = vec3<f32>(s_texcoords.data[i0 * 3 + 0], s_texcoords.data[i0 * 3 + 1], s_texcoords.data[i0 * 3 + 2]);

            out.near_pos = ndc_to_world * ndc_pos1;
            out.far_pos = ndc_to_world * ndc_pos2;
            //out.near_pos = out.near_pos / out.near_pos.w;
            //out.far_pos = out.far_pos / out.far_pos.w;

            return out;
        }
        """

    def fragment_shader(self):
        return """

        fn sample(texcoord: vec3<f32>) -> vec4<f32> {
            var color_value: vec4<f32>;

            $$ if texture_format == 'f32'
                color_value = textureSample(r_tex, r_sampler, texcoord.xyz);
            $$ else
                let texcoords_dim = vec3<f32>(textureDimensions(r_tex));
                let texcoords_u = vec3<i32>(texcoord.xyz * texcoords_dim % texcoords_dim);
                color_value = vec4<f32>(textureLoad(r_tex, texcoords_u, 0));
            $$ endif

            $$ if climcorrection
                color_value = vec4<f32>(color_value.rgb {{ climcorrection }}, color_value.a);
            $$ endif
            $$ if texture_nchannels == 1
                color_value = vec4<f32>(color_value.rrr, 1.0);
            $$ elif texture_nchannels == 2
                color_value = vec4<f32>(color_value.rrr, color_value.g);
            $$ endif
            return color_value;
        }

        [[stage(fragment)]]
        fn fs_main(in: VertexOutput) -> FragmentOutput {

            let relative_step_size = 1.0;

            let farpos = in.far_pos.xyz / in.far_pos.w;
            let nearpos = in.near_pos.xyz / in.near_pos.w;

            // Calculate unit vector pointing in the view direction through this fragment.
            let view_ray = normalize(farpos.xyz - nearpos.xyz);

            // ==== Raycasting setup
            let pos = in.world_pos.xyz / in.world_pos.w;
            let sizef = vec3<f32>(u_wobject.size.xyz);
            var distance = dot(nearpos - pos, view_ray);
            distance = max(distance, min((-0.5 - pos.x) / view_ray.x, (sizef.x - 0.5 - pos.x) / view_ray.x));
            distance = max(distance, min((-0.5 - pos.y) / view_ray.y, (sizef.y - 0.5 - pos.y) / view_ray.y));
            distance = max(distance, min((-0.5 - pos.z) / view_ray.z, (sizef.z - 0.5 - pos.z) / view_ray.z));
            // Now we have the starting position on the front surface
            let front = pos + view_ray * distance;

            // Decide how many steps to take
            let nsteps = i32(-distance / relative_step_size + 0.5);
            if( nsteps < 1 ) { discard; }
            // Get starting location and step vector in texture coordinates
            let step = ((pos - front) / sizef) / f32(nsteps);
            let start_loc = front / sizef;

            // ==== Before loop
            var max_val = -999999.0;
            var max_iter = -1;
            var loc = start_loc;

            // ==== In loop
            for (var iter=0; iter<nsteps; iter=iter+1) {
                let color = sample(loc);
                let val = color.r;
                if (val > max_val) {
                    max_val = val;
                    max_iter = iter;
                }
                // Next!
                loc = loc + step;
            }

            // ==== Refine
            var the_color = vec4<f32>(0.0, 0.0, 0.0, 0.0);
            var the_loc = start_loc;
            if ( max_iter > -1 ) {
                loc = start_loc + step * (f32(max_iter) - 0.5);
                max_val = max_val - 1.0;
                for (var i=0; i<10; i=i+1) {
                    let color = sample(loc);
                    let val = color.r;
                    if (val > max_val) {
                        max_val = val;
                        the_loc = loc;
                        the_color = color;
                    }
                    loc = loc + step * 0.1;
                }
            }

            // ==== Colormapping etc.
            let albeido = (the_color.rgb - u_material.clim[0]) / (u_material.clim[1] - u_material.clim[0]);
            var out: FragmentOutput;
            out.color = vec4<f32>(albeido, the_color.a * u_material.opacity);
            out.pick = vec4<i32>(u_wobject.id, vec3<i32>(the_loc * 1048576.0 + 0.5));
            //out.color = vec4<f32>(start_loc, 1.0);
            return out;
        }
        """
