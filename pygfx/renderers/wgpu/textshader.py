import wgpu  # only for flags/enums

from . import register_wgpu_render_function, WorldObjectShader, Binding, RenderMask
from ...objects import Text
from ...materials import TextMaterial
from ...utils.text import glyph_atlas
from ...utils.text._shaper import REF_GLYPH_SIZE

# from ...resources import Texture, TextureView

GLYPH_SIZE = glyph_atlas.glyph_size


@register_wgpu_render_function(Text, TextMaterial)
class TextShader(WorldObjectShader):

    type = "render"

    def get_bindings(self, wobject, shared):

        geometry = wobject.geometry
        material = wobject.material

        self["screen_space"] = material.screen_space
        self["aa"] = material.aa

        sbuffer = "buffer/read_only_storage"
        bindings = [
            Binding("u_stdinfo", "buffer/uniform", shared.uniform_buffer),
            Binding("u_wobject", "buffer/uniform", wobject.uniform_buffer),
            Binding("u_material", "buffer/uniform", material.uniform_buffer),
            Binding("s_indices", sbuffer, geometry.indices, "VERTEX"),
            Binding("s_positions", sbuffer, geometry.positions, "VERTEX"),
            Binding("s_sizes", sbuffer, geometry.sizes, "VERTEX"),
        ]

        rects = shared.glyph_atlas_rects_buffer
        view = shared.glyph_atlas_texture_view
        bindings.append(Binding("s_rects", sbuffer, rects, "VERTEX"))
        bindings.append(Binding("s_atlas", "sampler/filtering", view, "FRAGMENT"))
        bindings.append(Binding("t_atlas", "texture/auto", view, "FRAGMENT"))

        # Let the shader generate code for our bindings
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
        n = wobject.geometry.positions.nitems * 6
        render_mask = wobject.render_mask
        if not render_mask:
            if material.is_transparent:
                render_mask = RenderMask.transparent
            elif material.color_is_transparent:
                render_mask = RenderMask.transparent
            else:
                render_mask = RenderMask.all
        return {
            "indices": (n, 1),
            "render_mask": render_mask,
        }

    def get_code(self):
        sizes = f"""
        let GLYPH_SIZE: i32 = {GLYPH_SIZE};
        let REF_GLYPH_SIZE: i32 = {REF_GLYPH_SIZE};
        """

        return (
            self.code_definitions()
            + sizes
            + self.code_common()
            + self.code_vertex()
            + self.code_fragment()
        )

    def code_vertex(self):

        return """

        struct VertexInput {
            @builtin(vertex_index) vertex_index : u32,
        };


        @stage(vertex)
        fn vs_main(in: VertexInput) -> Varyings {

            let screen_factor = u_stdinfo.logical_size.xy / 2.0;

            let raw_index = i32(in.vertex_index);
            let index = raw_index / 6;
            let sub_index = raw_index % 6;

            // Load glyph info
            let glyph_index = i32(load_s_indices(index));
            let font_size = load_s_sizes(index);
            let glyph_pos = load_s_positions(index);

            // Load meta-data of the glyph in the atlas
            let bitmap_rect = load_s_rects(glyph_index);

            // Prep correction vectors
            // The first puts the rectangle to put it on the baseline/origin.
            // The second puts it at the end of the atlas-glyph rectangle.
            let pos_offset1 = vec2<f32>(bitmap_rect.xy) / f32(REF_GLYPH_SIZE);
            let pos_offset2 = vec2<f32>(bitmap_rect.zw) / f32(REF_GLYPH_SIZE);

            var corners = array<vec2<f32>, 6>(
                vec2<f32>(0.0, 0.0),
                vec2<f32>(0.0, 1.0),
                vec2<f32>(1.0, 0.0),
                vec2<f32>(0.0, 1.0),
                vec2<f32>(1.0, 0.0),
                vec2<f32>(1.0, 1.0),
            );
            let corner = corners[sub_index];

            let pos_corner_factor = corner * vec2<f32>(1.0, -1.0);
            let vertex_pos = glyph_pos + (pos_offset1 + pos_offset2 * pos_corner_factor) * font_size;
            let glyph_texcoord = corner * vec2<f32>(bitmap_rect.zw) / f32(GLYPH_SIZE);

            $$ if screen_space

                // We take the object's pos (model pos is origin), move to NDC, and apply the
                // glyph-positioning in logical screen coords.

                let raw_pos = vec3<f32>(0.0, 0.0, 0.0);
                let world_pos = u_wobject.world_transform * vec4<f32>(raw_pos, 1.0);
                let ndc_pos = u_stdinfo.projection_transform * u_stdinfo.cam_transform * world_pos;
                let delta_ndc = vertex_pos / screen_factor;

                // Pixel scale is easy

                let atlas_pixel_scale = font_size / f32(GLYPH_SIZE);

            $$ else

                // We take the glyph positions as model pos, move to world and then NDC.

                let raw_pos = vec4<f32>(vertex_pos, 0.0, 1.0);
                let world_pos = u_wobject.world_transform * raw_pos;
                let ndc_pos = u_stdinfo.projection_transform * u_stdinfo.cam_transform * world_pos;
                let delta_ndc = vec2<f32>(0.0, 0.0);

                // For the pixel scale, we first project a second point that is
                // offset diagonally from the current vertex. The two points are
                // mapped to screen coords and the smallest distance is used for scale.

                let one_atlas_pixel = vec2<f32>(font_size / f32(GLYPH_SIZE));

                let raw_pos2 = vec4<f32>(vertex_pos + one_atlas_pixel, 0.0, 1.0);
                let world_pos2 = u_wobject.world_transform * raw_pos2;
                let ndc_pos2 = u_stdinfo.projection_transform * u_stdinfo.cam_transform * world_pos2;

                let screen_pos1 = (ndc_pos.xy / ndc_pos.w) * screen_factor;
                let screen_pos2 = (ndc_pos2.xy / ndc_pos2.w) * screen_factor;
                let screen_diff = abs(screen_pos1 - screen_pos2);
                let atlas_pixel_scale = min(screen_diff.x, screen_diff.y);

            $$ endif

            var varyings: Varyings;
            varyings.position = vec4<f32>(ndc_pos.xy + delta_ndc * ndc_pos.w, ndc_pos.zw);
            varyings.world_pos = vec3<f32>(world_pos.xyz / world_pos.w);
            varyings.font_size = f32(font_size);
            varyings.atlas_pixel_scale = f32(atlas_pixel_scale);
            varyings.glyph_texcoord = vec2<f32>(glyph_texcoord);
            varyings.glyph_index = i32(glyph_index);

            // Picking
            varyings.pick_idx = u32(index);

            return varyings;
        }
        """

    def code_fragment(self):
        return """

        fn _sdf_smoothstep( low : f32, high : f32, x : f32 ) -> f32 {
            let t = clamp( ( x - low ) / ( high - low ), 0.0, 1.0 );
            return t * t * ( 3.0 - 2.0 * t );
        }

        @stage(fragment)
        fn fs_main(varyings: Varyings) -> FragmentOutput {

            // The glyph is represented in the texture as a square region
            // of fixed size, though only a subrectangle is actually used:
            // the coverage.
            //
            // o-------   →  glyph_texcoord.x in 0..cover_x (max 1.0)
            // |       |
            // |       |  ↓  glyph_texcoord.y in 0..cover_y (max 1.0)
            // |       |
            //  -------

            // Calculate texture position (in the atlas)
            let atlas_size = textureDimensions(t_atlas);
            let glyph_index = varyings.glyph_index;
            let ncols = atlas_size.x / GLYPH_SIZE;
            let col_row = vec2<i32>(glyph_index % ncols, glyph_index / ncols);
            let texcoord = (vec2<f32>(col_row) + varyings.glyph_texcoord) * f32(GLYPH_SIZE) / vec2<f32>(atlas_size);

            // Sample distance. A value of 0.5 represents the edge of the glyph,
            // with positive values representing the inside.
            let atlas_value = textureSample(t_atlas, s_atlas, texcoord).r;

            // Convert to a more useful measure, where the border is at 0.0,
            // the inside is negative, and values represent distances in atlas-pixels.
            let distance = (0.5 - atlas_value) * 128.0;

            // Determine cutoff, we can tweak the glyph thickness here.
            // Default thickness is 1. With log2(1) == 1, default cutoff is 0.
            let cut_off = - varyings.font_size * 0.1 * log2(clamp(0.1, 10.0, u_material.thickness));

            $$ if aa
                // We use smoothstep to include alpha blending.
                // The smoothness is calculated from the scale of one atlas-pixel in screen space.
                // High smoothness values also result in lower alpha to prevent artifacts under high angles.
                let max_softness = f32(GLYPH_SIZE);
                let softness = clamp(0.0, max_softness, 5.0 / varyings.atlas_pixel_scale);
                let softener = 1.0 - max(softness / max_softness - 0.5, 0.0);
                let alpha = softener * _sdf_smoothstep(cut_off - softness, cut_off + softness, -distance);
            $$ else
                // Do a hard transition
                let alpha = select(0.0, 1.0, distance < cut_off);
            $$ endif

            // Outline
            //let outline_thickness = 4.0;
            //let outline_softness = 2.0;
            //let outline_color = vec4<f32>(1.0, 0.0, 0.0, 1.0);
            //let outline = _sdf_smoothstep(outline_thickness - outline_softness, outline_thickness + outline_softness, -distance);

            // Early exit
            if (alpha <= 0.0) { discard; }

            // Compose the final color
            let color_srgb = u_material.color;
            let color = srgb2physical(color_srgb.rgb);
            let opacity = color_srgb.a * u_material.opacity * alpha;
            var color_out = vec4<f32>(color, opacity);

            // Debug
            //color_out = vec4<f32>(atlas_value, 1.0, 0.0, 1.0);

            // Wrap up
            apply_clipping_planes(varyings.world_pos);
            var out = get_fragment_output(varyings.position.z, color_out);

            $$ if write_pick
            // The wobject-id must be 20 bits. In total it must not exceed 64 bits.
            out.pick = (
                pick_pack(u32(u_wobject.id), 20) +
                pick_pack(varyings.pick_idx, 26) +
                pick_pack(u32(varyings.glyph_texcoord.x + 256.0), 9) +
                pick_pack(u32(varyings.glyph_texcoord.y + 256.0), 9)
            );
            $$ endif

            return out;
        }
        """
