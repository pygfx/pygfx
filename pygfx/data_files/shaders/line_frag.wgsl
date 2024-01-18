@fragment
        fn fs_main(varyings: Varyings, @builtin(front_facing) is_front: bool) -> FragmentOutput {

            // Discard fragments outside of the radius. This is what makes round
            // joins and caps. If we ever want bevel or miter joins, we should
            // change the vertex positions a bit, and drop these lines below.

            // Butt cap
            //if (varyings.vec_from_line_p.x > 0.0) {
            //     discard;
            //}
            let l2p = u_stdinfo.physical_size.x / u_stdinfo.logical_size.x;

            let half_thickness_p = 0.5 * varyings.thickness_p;

            // Discard invalid faces.
            // These are faces for which *all* 3 verts are set to zero.
            if (varyings.valid == 0.0) {
                discard;
            }

            // Determine whether we are at a join (i.e. an unbroken corner).
            // These are faces for which *any* vert is nonzero.
            let is_join = varyings.is_join != 0.0;

            //let join_coord = (varyings.join_coord.x / varyings.join_coord.y);
            let join_coord = varyings.join_coord.x;
        
            // Get the line coord in physical pixels. We need a different varying
            // depending on whether this is a join. The vector's direction is in "screen coords",
            // we can only really use it's length.
            let line_coord_p = select(varyings.vec_from_line_p, varyings.vec_from_node_p, is_join);

            let side = varyings.vec_from_node_p.y;

            //let free_zone = abs(line_coord_p.x) > 0.5 * varyings.segment_inset;//is_join && (varyings.is_join * side) < 0.0;
            let free_zone =is_join &&  abs(join_coord) > 0.5;

            $$ if dashing

                let cum_dist = select(varyings.cum_dist, varyings.cum_dist_join / varyings.cum_dist_divisor, is_join);
                //let cum_dist =  varyings.cum_dist;

                // A builtin offset to position the dash nicer.
                let local_dash_offset = 0.0;

                // Calculate dash_progress, a number 0..1, indicating the fase of the dash.
                let dash_size = u_material.dash_size;
                let dash_ratio = u_material.dash_ratio;
                let dash_progress = ((cum_dist + local_dash_offset) % dash_size) / dash_size;

                // Get distance to dash-stroke. We make the stroke the center
                // of the dash period, which makes the math easier.
                //
                //        ratio e.g. 0.6
                //       /       \
                //  ----|---------|----
                //  0       0.5       1    dash_progress
                // 0.2  0  -0.3   0  0.2   dist_to_stroke
                //
                var dist_to_stroke = abs(dash_progress - 0.5) - 0.5 * dash_ratio;
                dist_to_stroke = max(0.0, dist_to_stroke);

                // Convert to (physical) pixel units
                let dist_to_stroke_p = dash_size * dist_to_stroke * l2p;

                // The vector to the stoke (at the line-center)
                let vec_to_stroke_p = vec2<f32>(dist_to_stroke_p, varyings.vec_from_line_p.y);

                // Butt caps
                if (dist_to_stroke > 0.0) {
                    //discard;
                }

                // Round caps
                if (length(vec_to_stroke_p) > half_thickness_p) {
                    discard;
                }


            $$ endif

            let dist_to_node_p = length(line_coord_p);
            if (dist_to_node_p >  half_thickness_p && !free_zone) {
                discard;
            }

            // Prep
            var alpha: f32 = 1.0;

            // Anti-aliasing. Note that because of the discarding above, we cannot use MSAA.
            // By default, the renderer uses SSAA (super-sampling), but if we apply AA for the edges
            // here this will help the end result. Because this produces semitransparent fragments,
            // it relies on a good blend method, and the object gets drawn twice.
            $$ if false
                let aa_width = 1.0;
                alpha = (half_thickness_p - abs(dist_to_node_p)) / aa_width;
                alpha = clamp(alpha, 0.0, 1.0);
            $$ endif

            $$ if color_mode == 'vertex' or color_mode == 'face'
                let color = varyings.color;
            $$ elif color_mode == 'vertex_map' or color_mode == 'face_map'
                let color = sample_colormap(varyings.texcoord);
            $$ else
                let color = u_material.color;
            $$ endif

            var physical_color = srgb2physical(color.rgb);
            $$ if false 
                // DEBUG
                //physical_color = vec3<f32>(abs(1.0/vec_to_stroke_p.y), 0.0, 0.0);
                // physical_color = vec3<f32>(abs(0.01 * vec_to_stroke_p.x), 0.0, 0.0);
                physical_color = vec3<f32>(0.0,f32(abs(join_coord)), 1.0);
            $$ endif
            let opacity = min(1.0, color.a) * alpha * u_material.opacity;

            //let opacity_multiplier = select(-1.0, 1.0, !is_front);
            let out_color = vec4<f32>(physical_color, opacity);

            // Wrap up
            apply_clipping_planes(varyings.world_pos);
            var out = get_fragment_output(varyings.position.z, out_color);

            // Set picking info.
            $$ if write_pick
            // The wobject-id must be 20 bits. In total it must not exceed 64 bits.
            // The pick_idx is int-truncated, so going from a to b, it still has the value of a
            // even right up to b. The pick_zigzag alternates between 0 (even indices) and 1 (odd indices).
            // Here we decode that. The result is that we can support vertex indices of ~32 bits if we want.
            let is_even = varyings.pick_idx % 2u == 0u;
            var coord = select(varyings.pick_zigzag, 1.0 - varyings.pick_zigzag, is_even);
            coord = select(coord, coord - 1.0, coord > 0.5);
            let idx = varyings.pick_idx + select(0u, 1u, coord < 0.0);
            out.pick = (
                pick_pack(u32(u_wobject.id), 20) +
                pick_pack(u32(idx), 26) +
                pick_pack(u32(coord * 100000.0 + 100000.0), 18)
            );
            $$ endif

            // The outer edges with lower alpha for aa are pushed a bit back to avoid artifacts.
            // This is only necessary for blend method "ordered1"
            //out.depth = varyings.position.z + 0.0001 * (0.8 - min(0.8, alpha));

            return out;
        }