import wgpu  # only for flags/enums

from . import register_wgpu_render_function, WorldObjectShader, Binding, RenderMask
from ...objects import Points
from ...materials import PointsMaterial, GaussianPointsMaterial


@register_wgpu_render_function(Points, PointsMaterial)
class PointsShader(WorldObjectShader):
    # Notes:
    # In WGPU, the pointsize attribute can no longer be larger than 1 because
    # of restriction in some hardware/backend API's. So we use our storage-buffer
    # approach (similar for what we use for lines) to sort of fake a geometry shader.
    # An alternative is to use instancing. Could be worth testing both approaches
    # for performance ...

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
        elif color_mode == "vertex_map":
            self["color_mode"] = "vertex_map"
            self["color_buffer_channels"] = 0
            if material.map is None:
                raise ValueError(f"Cannot apply colormap is no material.map is set.")
        else:
            raise RuntimeError(f"Unknown color_mode: '{color_mode}'")

    def get_bindings(self, wobject, shared):
        geometry = wobject.geometry
        material = wobject.material

        rbuffer = "buffer/read_only_storage"
        bindings = [
            Binding("u_stdinfo", "buffer/uniform", shared.uniform_buffer),
            Binding("u_wobject", "buffer/uniform", wobject.uniform_buffer),
            Binding("u_material", "buffer/uniform", material.uniform_buffer),
            Binding("s_positions", rbuffer, geometry.positions, "VERTEX"),
        ]

        self["per_vertex_sizes"] = False
        if material.vertex_sizes:
            self["per_vertex_sizes"] = True
            bindings.append(Binding("s_sizes", rbuffer, geometry.sizes, "VERTEX"))

        # Per-vertex color, colormap, or a uniform color?
        if self["color_mode"] == "vertex":
            bindings.append(Binding("s_colors", rbuffer, geometry.colors, "VERTEX"))
        elif self["color_mode"] == "vertex_map":
            bindings.extend(
                self.define_texcoords_and_colormap(
                    material.map, geometry.texcoords, material.map_interpolation
                )
            )

        self["shape"] = "circle"
        if isinstance(material, GaussianPointsMaterial):
            self["shape"] = "gaussian"

        bindings = {i: b for i, b in enumerate(bindings)}
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
        material = wobject.material

        offset, size = wobject.geometry.positions.draw_range
        offset, size = offset * 6, size * 6

        render_mask = wobject.render_mask
        if not render_mask:
            if material.is_transparent:
                render_mask = RenderMask.transparent
            elif self["color_mode"] == "uniform":
                if material.color_is_transparent:
                    render_mask = RenderMask.transparent
                else:
                    render_mask = RenderMask.all
            else:
                render_mask = RenderMask.all

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

            let index = i32(in.vertex_index);
            let i0 = index / 6;
            let sub_index = index % 6;

            let raw_pos = load_s_positions(i0);
            let world_pos = u_wobject.world_transform * vec4<f32>(raw_pos, 1.0);
            let ndc_pos = u_stdinfo.projection_transform * u_stdinfo.cam_transform * world_pos;

            var deltas = array<vec2<f32>, 6>(
                vec2<f32>(-1.0, -1.0),
                vec2<f32>(-1.0,  1.0),
                vec2<f32>( 1.0, -1.0),
                vec2<f32>(-1.0,  1.0),
                vec2<f32>( 1.0, -1.0),
                vec2<f32>( 1.0,  1.0),
            );

            // Need size here in vertex shader too
            $$ if per_vertex_sizes
                let size = load_s_sizes(i0);
            $$ else
                let size = u_material.size;
            $$ endif

            let aa_margin = 1.0;
            let delta_logical = deltas[sub_index] * (size + aa_margin);
            let delta_ndc = delta_logical * (1.0 / u_stdinfo.logical_size);

            var varyings: Varyings;
            varyings.position = vec4<f32>(ndc_pos.xy + delta_ndc * ndc_pos.w, ndc_pos.zw);
            varyings.world_pos = vec3<f32>(world_pos.xyz / world_pos.w);
            varyings.pointcoord = vec2<f32>(delta_logical);
            varyings.size = f32(size);

            // Picking
            varyings.pick_idx = u32(i0);

            // per-vertex or per-face coloring
            $$ if color_mode == 'face' or color_mode == 'vertex'
                let color_index = i0;
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
            let tex_coord_index = i0;

            // Set texture coords
            $$ if colormap_dim == '1d'
            varyings.texcoord = f32(load_s_texcoords(tex_coord_index));
            $$ elif colormap_dim == '2d'
            varyings.texcoord = vec2<f32>(load_s_texcoords(tex_coord_index));
            $$ elif colormap_dim == '3d'
            varyings.texcoord = vec3<f32>(load_s_texcoords(tex_coord_index));
            $$ endif

            return varyings;
        }
        """

    def code_fragment(self):
        # Also see See https://github.com/vispy/vispy/blob/master/vispy/visuals/markers.py
        return """

        @fragment
        fn fs_main(varyings: Varyings) -> FragmentOutput {
            var final_color : vec4<f32>;

            let d = length(varyings.pointcoord);
            let aa_width = 1.0;

            $$ if per_vertex_sizes
                let size = varyings.size;
            $$ else
                let size = u_material.size;
            $$ endif

            $$ if color_mode == 'vertex'
                let color = varyings.color;
            $$ elif color_mode == 'map'
                let color = sample_colormap(varyings.texcoord);
            $$ else
                let color = u_material.color;
            $$ endif

            $$ if shape == 'circle'
                if (d <= size - 0.5 * aa_width) {
                    final_color = color;
                } else if (d <= size + 0.5 * aa_width) {
                    let alpha1 = 0.5 + (size - d) / aa_width;
                    let alpha2 = pow(alpha1, 2.0);  // this works better
                    final_color = vec4<f32>(color.rgb, color.a * alpha2);
                } else {
                    discard;
                }
            $$ elif shape == "gaussian"
                if (d <= size) {
                    let sigma = size / 3.0;
                    let t = d / sigma;
                    let a = exp(-0.5 * t * t);
                    final_color = vec4<f32>(color.rgb, color.a * a);
                } else {
                    discard;
                }
            $$ else
                invalid_point_type;
            $$ endif

            let physical_color = srgb2physical(final_color.rgb);
            let opacity = final_color.a * u_material.opacity;
            let out_color = vec4<f32>(physical_color, opacity);

            // Wrap up
            apply_clipping_planes(varyings.world_pos);
            var out = get_fragment_output(varyings.position.z, out_color);

            $$ if write_pick
            // The wobject-id must be 20 bits. In total it must not exceed 64 bits.
            out.pick = (
                pick_pack(u32(u_wobject.id), 20) +
                pick_pack(varyings.pick_idx, 26) +
                pick_pack(u32(varyings.pointcoord.x + 256.0), 9) +
                pick_pack(u32(varyings.pointcoord.y + 256.0), 9)
            );
            $$ endif

            return out;
        }
        """
