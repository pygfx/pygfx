import wgpu  # only for flags/enums

from . import register_wgpu_render_function
from ._shadercomposer import Binding, WorldObjectShader
from .imagerender import sampled_value_to_color, handle_colormap
from ._utils import to_texture_format
from ...objects import Volume
from ...materials import VolumeBasicMaterial, VolumeSliceMaterial, VolumeRayMaterial
from ...resources import Texture, TextureView


vertex_and_fragment = wgpu.ShaderStage.VERTEX | wgpu.ShaderStage.FRAGMENT


@register_wgpu_render_function(Volume, VolumeBasicMaterial)
def volume_renderer(render_info):
    """Render function capable of rendering volumes."""

    wobject = render_info.wobject
    geometry = wobject.geometry
    material = wobject.material  # noqa

    # The shaders for the volume slice and ray are very different, but their
    # setup is very similar. This triage captures all diferences.
    if isinstance(material, VolumeSliceMaterial):
        shader = VolumeSliceShader(render_info, climcorrection="")
        topology = wgpu.PrimitiveTopology.triangle_list
        n = 12
        cull_mode = wgpu.CullMode.none
    elif isinstance(material, VolumeRayMaterial):
        shader = VolumeRayShader(
            render_info,
            climcorrection="",
            mode=material.render_mode,
        )
        topology = wgpu.PrimitiveTopology.triangle_list
        n = 36
        cull_mode = wgpu.CullMode.front  # the back planes are the ref
    else:
        raise RuntimeError(f"Unexpected volume material: {material}")

    bindings = [
        Binding("u_stdinfo", "buffer/uniform", render_info.stdinfo_uniform),
        Binding("u_wobject", "buffer/uniform", wobject.uniform_buffer),
        Binding("u_material", "buffer/uniform", material.uniform_buffer),
    ]

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
        # Sampling type
        fmt = to_texture_format(geometry.grid.format)
        if "norm" in fmt or "float" in fmt:
            shader["img_format"] = "f32"
            if "unorm" in fmt:
                shader["climcorrection"] = " * 255.0"
            elif "snorm" in fmt:
                shader["climcorrection"] = " * 255.0 - 128.0"
        elif "uint" in fmt:
            shader["img_format"] = "u32"
        else:
            shader["img_format"] = "i32"
        # Channels
        shader["img_nchannels"] = len(fmt) - len(fmt.lstrip("rgba"))

    bindings.append(Binding("s_img", "sampler/filtering", view, "FRAGMENT"))
    bindings.append(Binding("t_img", "texture/auto", view, vertex_and_fragment))

    # If a colormap is applied ...
    if material.map is not None:
        bindings.extend(handle_colormap(geometry, material, shader))

    # Let the shader generate code for our bindings
    for i, binding in enumerate(bindings):
        shader.define_binding(0, i, binding)

    # Get in what passes this needs rendering
    suggested_render_mask = 3
    if material.opacity >= 1 and shader["img_nchannels"] in (1, 3):
        suggested_render_mask = 1
    elif material.opacity < 1:
        suggested_render_mask = 2

    # Put it together!
    return [
        {
            "suggested_render_mask": suggested_render_mask,
            "render_shader": shader,
            "primitive_topology": topology,
            "cull_mode": cull_mode,
            "indices": (range(n), range(1)),
            "vertex_buffers": {},
            "bindings0": bindings,
        }
    ]


class BaseVolumeShader(WorldObjectShader):
    def volume_helpers(self):
        return (
            sampled_value_to_color
            + """
        struct VolGeometry {
            indices: array<i32,36>;
            positions: array<vec3<f32>,8>;
            texcoords: array<vec3<f32>,8>;
        };

        fn get_vol_geometry() -> VolGeometry {
            let size = textureDimensions(t_img);
            var geo: VolGeometry;

            geo.indices = array<i32,36>(
                0, 1, 2,   3, 2, 1,   4, 5, 6,   7, 6, 5,   6, 7, 3,   2, 3, 7,
                1, 0, 4,   5, 4, 0,   5, 0, 7,   2, 7, 0,   1, 4, 3,   6, 3, 4,
            );

            let pos1 = vec3<f32>(-0.5);
            let pos2 = vec3<f32>(size) + pos1;
            geo.positions = array<vec3<f32>,8>(
                vec3<f32>(pos2.x, pos1.y, pos2.z),
                vec3<f32>(pos2.x, pos1.y, pos1.z),
                vec3<f32>(pos2.x, pos2.y, pos2.z),
                vec3<f32>(pos2.x, pos2.y, pos1.z),
                vec3<f32>(pos1.x, pos1.y, pos1.z),
                vec3<f32>(pos1.x, pos1.y, pos2.z),
                vec3<f32>(pos1.x, pos2.y, pos1.z),
                vec3<f32>(pos1.x, pos2.y, pos2.z),
            );

            geo.texcoords = array<vec3<f32>,8>(
                vec3<f32>(1.0, 0.0, 1.0),
                vec3<f32>(1.0, 0.0, 0.0),
                vec3<f32>(1.0, 1.0, 1.0),
                vec3<f32>(1.0, 1.0, 0.0),
                vec3<f32>(0.0, 0.0, 0.0),
                vec3<f32>(0.0, 0.0, 1.0),
                vec3<f32>(0.0, 1.0, 0.0),
                vec3<f32>(0.0, 1.0, 1.0),
            );

            return geo;
        }

        fn sample_vol(texcoord: vec3<f32>, sizef: vec3<f32>) -> vec4<f32> {
            $$ if img_format == 'f32'
                return textureSample(t_img, s_img, texcoord.xyz);
            $$ else
                let texcoords_u = vec3<i32>(texcoord.xyz * sizef);
                return vec4<f32>(textureLoad(t_img, texcoords_u, 0));
            $$ endif
        }

    """
        )


class VolumeSliceShader(BaseVolumeShader):
    def get_code(self):
        return (
            self.get_definitions()
            + self.common_functions()
            + self.volume_helpers()
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

            // Our geometry is implicitly defined by the volume dimensions.
            var geo = get_vol_geometry();

            // This layout is like this:
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
                let p1_raw = geo.positions[ edge[0] ];
                let p2_raw = geo.positions[ edge[1] ];
                let p1_p = u_wobject.world_transform * vec4<f32>(p1_raw, 1.0);
                let p2_p = u_wobject.world_transform * vec4<f32>(p2_raw, 1.0);
                let p1 = p1_p.xyz / p1_p.w;
                let p2 = p2_p.xyz / p2_p.w;
                let tc1 = geo.texcoords[ edge[0] ];
                let tc2 = geo.texcoords[ edge[1] ];
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

            var varyings : Varyings;
            varyings.position = vec4<f32>(ndc_pos);
            varyings.world_pos = vec3<f32>(world_pos);
            varyings.texcoord = vec3<f32>(texcoords[ indexmap[index] ]);
            return varyings;
        }
        """

    def fragment_shader(self):
        return """

        [[stage(fragment)]]
        fn fs_main(varyings: Varyings) -> FragmentOutput {
            let sizef = vec3<f32>(textureDimensions(t_img));
            let value = sample_vol(varyings.texcoord.xyz, sizef);
            let color = sampled_value_to_color(value);
            let albeido = color.rgb;
            let final_color = vec4<f32>(albeido, color.a * u_material.opacity);

            // Wrap up
            apply_clipping_planes(varyings.world_pos);
            var out = get_fragment_output(varyings.position.z, final_color);

            $$ if write_pick
            // The wobject-id must be 20 bits. In total it must not exceed 64 bits.
            out.pick = (
                pick_pack(u32(u_wobject.id), 20) +
                pick_pack(u32(varyings.texcoord.x * 16384.0), 14) +
                pick_pack(u32(varyings.texcoord.y * 16384.0), 14) +
                pick_pack(u32(varyings.texcoord.z * 16384.0), 14)
            );
            $$ endif

            return out;
        }

        """


class VolumeRayShader(BaseVolumeShader):
    def get_code(self):
        return (
            self.get_definitions()
            + self.common_functions()
            + self.volume_helpers()
            + self.render_function()
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

            // Our geometry is implicitly defined by the volume dimensions.
            var geo = get_vol_geometry();

            // Select what face we're at
            let index = i32(in.vertex_index);
            let i0 = geo.indices[index];

            // Sample position, and convert to world pos, and then to ndc
            let data_pos = vec4<f32>(geo.positions[i0], 1.0);
            let world_pos = u_wobject.world_transform * data_pos;
            let ndc_pos = u_stdinfo.projection_transform * u_stdinfo.cam_transform * world_pos;

            // Prepare inverse matrix
            let ndc_to_data = u_wobject.world_transform_inv * u_stdinfo.cam_transform_inv * u_stdinfo.projection_transform_inv;

            var varyings: Varyings;

            // Store values for fragment shader
            varyings.position = vec4<f32>(ndc_pos);
            varyings.world_pos = vec3<f32>(world_pos.xyz);

            // The position on the face of the cube. We can say that it's the back face,
            // because we cull the front faces.
            // These positions are in data positions (voxels) rather than texcoords (0..1),
            // because distances make more sense in this space. In the fragment shader we
            // can consider it an isotropic volume, because any position, rotation,
            // and scaling of the volume is part of the world transform.
            varyings.data_back_pos = vec4<f32>(data_pos);

            // We calculate the NDC positions for the near and front clipping planes,
            // and transform these back to data coordinates. From these positions
            // we can construct the view vector in the fragment shader, which is then
            // resistant to perspective transforms. It also makes that if the camera
            // is inside the volume, only the part in front in rendered.
            // Note that the w component for these positions should be left intact.
            let ndc_pos1 = vec4<f32>(ndc_pos.xy, -ndc_pos.w, ndc_pos.w);
            let ndc_pos2 = vec4<f32>(ndc_pos.xy, ndc_pos.w, ndc_pos.w);
            varyings.data_near_pos = vec4<f32>(ndc_to_data * ndc_pos1);
            varyings.data_far_pos = vec4<f32>(ndc_to_data * ndc_pos2);

            return varyings;
        }
        """

    def fragment_shader(self):
        return """
        [[stage(fragment)]]
        fn fs_main(varyings: Varyings) -> FragmentOutput {

            // Get size of the volume
            let sizef = vec3<f32>(textureDimensions(t_img));

            // Determine the stepsize as a float in pixels.
            // This value should be between ~ 0.1 and 1. Smaller values yield better
            // results at the cost of performance. With larger values you may miss
            // small structures (and corners of larger structures) because the step
            // may skip over them.
            // We could make this a user-facing property. But for now we scale between
            // 0.1 and 0.8 based on the (sqrt of the) volume size.
            let relative_step_size = clamp(sqrt(max(sizef.x, max(sizef.y, sizef.z))) / 20.0, 0.1, 0.8);

            // Positions in data coordinates
            let back_pos = varyings.data_back_pos.xyz / varyings.data_back_pos.w;
            let far_pos = varyings.data_far_pos.xyz / varyings.data_far_pos.w;
            let near_pos = varyings.data_near_pos.xyz / varyings.data_near_pos.w;

            // Calculate unit vector pointing in the view direction through this fragment.
            let view_ray = normalize(far_pos - near_pos);

            // Calculate the (signed) distance, from back_pos to the first voxel
            // that must be sampled, expressed in data coords (voxels).
            var dist = dot(near_pos - back_pos, view_ray);
            dist = max(dist, min((-0.5 - back_pos.x) / view_ray.x, (sizef.x - 0.5 - back_pos.x) / view_ray.x));
            dist = max(dist, min((-0.5 - back_pos.y) / view_ray.y, (sizef.y - 0.5 - back_pos.y) / view_ray.y));
            dist = max(dist, min((-0.5 - back_pos.z) / view_ray.z, (sizef.z - 0.5 - back_pos.z) / view_ray.z));

            // Now we have the starting position. This is typically on a front face,
            // but it can also be incide the volume (on the near plane).
            let front_pos = back_pos + view_ray * dist;

            // Decide how many steps to take. If we'd not cul the front faces,
            // that would still happen here because nsteps would be negative.
            let nsteps = i32(-dist / relative_step_size + 0.5);
            if( nsteps < 1 ) { discard; }

            // Get starting positon and step vector in texture coordinates.
            let start_coord = (front_pos + vec3<f32>(0.5, 0.5, 0.5)) / sizef;
            let step_coord = ((back_pos - front_pos) / sizef) / f32(nsteps);

            // Render
            let render_out = render_func(sizef, nsteps, start_coord, step_coord);

            // Get world and ndc pos from the calculatex texture coordinate
            let data_pos = render_out.coord * sizef - vec3<f32>(0.5, 0.5, 0.5);
            let world_pos = u_wobject.world_transform * vec4<f32>(data_pos, 1.0);
            let ndc_pos = u_stdinfo.projection_transform * u_stdinfo.cam_transform * world_pos;

            // Maybe we did the work for nothing
            apply_clipping_planes(world_pos.xyz);

            // Get fragment output. Note that the depth arg only affects the
            // blending - setting the depth attribute acually sets the fragment depth.
            let depth = ndc_pos.z / ndc_pos.w;
            var out = get_fragment_output(depth, render_out.color);
            out.depth = depth;

            $$ if write_pick
            // The wobject-id must be 20 bits. In total it must not exceed 64 bits.
            out.pick = (
                pick_pack(u32(u_wobject.id), 20) +
                pick_pack(u32(render_out.coord.x * 16384.0), 14) +
                pick_pack(u32(render_out.coord.y * 16384.0), 14) +
                pick_pack(u32(render_out.coord.z * 16384.0), 14)
            );
            $$ endif
            return out;
        }
        """

    def render_function(self):
        # Triage over different render modes. Only one mode so far :)
        f = getattr(self, "render_mode_" + self.kwargs["mode"].lower(), "mip")

        preamble = """
        struct RenderOutput {
            color: vec4<f32>;
            coord: vec3<f32>;
        };
        """

        return preamble + f()

    def render_mode_mip(self):

        # Ideas for improvement:
        # * We could textureLoad() the 27 voxels surrounding the initial location
        #   and sample from that in the refinement step. Less texture loads and we
        #   could do linear interpolation also for formats like i16.
        # * Create helper textures at a lower resolution (e.g. min, max) so we can
        #   skip along the ray much faster. By taking smaller steps where needed,
        #   it will be both faster and more accurate.

        return """
        fn render_func(sizef: vec3<f32>, nsteps: i32, start_coord: vec3<f32>, step_coord: vec3<f32>) -> RenderOutput {

            let nstepsf = f32(nsteps);

            // Primary loop. The purpose is to find the approximate location where
            // the maximum is.
            var the_ref = -999999.0;
            var the_coord = start_coord;
            var the_value : vec4<f32>;
            for (var iter=0.0; iter<nstepsf; iter=iter+1.0) {
                let coord = start_coord + iter * step_coord;
                let value = sample_vol(coord, sizef);
                let ref = value.r;
                if (ref > the_ref) {
                    the_ref = ref;
                    the_coord = coord;
                    the_value = value;
                }
            }

            // Secondary loop to close in on a more accurate position using
            // a divide-by-two approach.
            var substep_coord = step_coord;
            for (var iter2=0; iter2<4; iter2=iter2+1) {
                substep_coord = substep_coord * 0.5;
                let coord1 = the_coord - substep_coord;
                let coord2 = the_coord + substep_coord;
                let value1 = sample_vol(coord1, sizef);
                let value2 = sample_vol(coord2, sizef);
                let ref1 = value1.r;
                let ref2 = value2.r;
                if (ref1 >= the_ref) {  // deliberate larger-equal
                    the_ref = ref1;
                    the_coord = coord1;
                    the_value = value1;
                } elseif (ref2 > the_ref) {
                    the_ref = ref2;
                    the_coord = coord2;
                    the_value = value2;
                }
            }

            // Colormapping
            let color = sampled_value_to_color(the_value);
            let albeido = color.rgb;
            let color = vec4<f32>(albeido, color.a * u_material.opacity);

            // Produce result
            var out: RenderOutput;
            out.color = color;
            out.coord = the_coord;
            return out;
        }
        """
