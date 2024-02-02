$$ if dashing
    const dash_count = {{dash_count}};
$$ endif

@fragment
fn fs_main(varyings: Varyings, @builtin(front_facing) is_front: bool) -> FragmentOutput {
            
            let l2p = u_stdinfo.physical_size.x / u_stdinfo.logical_size.x;

            // Get the half-thickness in physical coordinates. This is the reference thickness.
            // If aa is used, the line is actually a bit thicker, leaving space to do aa.
            // TODO: might as well pass the screen thicknes as a varying :)
            let half_thickness_p = 0.5 * varyings.thickness_p; 
            let thickness_s = half_thickness_p * 2.0 / l2p;

            // Discard invalid faces. These are faces for which *all* 3 verts are set to zero. (trick 5b)
            if (varyings.valid_if_nonzero == 0.0) {
                discard;
            }

            // Determine whether we are at a join (i.e. an unbroken corner).
            // These are faces for which *any* vert is nonzero. (trick 5a)
            let is_join = varyings.join_coord != 0.0;

            // Obtain the join coordinates. It comes in two flavours, linear and fan-shaped,
            // which each serve a different purpose. These represent trick 3 and 4, respectively.
            //
            // join_coord_lin      join_coord_fan
            //
            // | | | | |-          | | | / / ╱
            // | | | |- -          | | | / ╱ ⟋
            // | | |- - -          | | | ╱ ⟋ ⟋
            //      - - -                - - -
            //      - - -                - - -
            //
            let join_coord_lin = varyings.join_coord;
            let join_coord_fan = join_coord_lin / varyings.is_outer_corner;
        
            // Get the line coord in physical pixels.
            // For joins, the outer vertices are inset, and we need to take that into account,
            // so that the origin is at the node (i.e. the pivot point).
            var segment_coord_p = varyings.segment_coord_p;
            if (is_join) {
                let dist_from_segment = abs(join_coord_lin);
                let a = segment_coord_p.x / dist_from_segment;
                segment_coord_p = vec2<f32>(max(0.0, dist_from_segment - 0.5) * a, segment_coord_p.y);
            }

            // Calculate the distance to the stroke's edge. Negative means inside, positive means outside. Just like SDF.
            var dist_to_stroke_p = length(segment_coord_p) - half_thickness_p;

            $$ if dashing
                
                // Calculate the cumulative distance along the line. We need a continuous value to parametrize
                // the dash (and its cap). Going around the corner, it will compress on the inside, and expand
                // on the outer side, deforming dashes as they move around the corner, appropriately.
                // We also need a linear value to map to physical screen coords (by taking its derivative).
                var cumdist_continuous : f32;
                var cumdist_linear: f32;
                if (is_join) {
                    // First calculate the cumdist at the edge where segment and join meet. 
                    // Note that cumdist_vertex == cumdist_node at the outer-corner-vertex.
                    let cumdist_segment = varyings.cumdist_node - (varyings.cumdist_node - varyings.cumdist_vertex) / (1.0 - abs(join_coord_lin));
                    // Calculate the continous cumdist, by interpolating using join_coord_fan
                    cumdist_continuous = mix(cumdist_segment, varyings.cumdist_node, abs(join_coord_fan));
                    // Almost the same mix, but using join_coord_lin, and a factor two, because the vertex in the outer corner
                    // is actually further than the node (with a factor 2), from the pov of the segments.
                    cumdist_linear = mix(cumdist_segment, varyings.cumdist_node, (2.0 * abs(join_coord_lin)));
                } else {
                    // In a segment everything is straight.
                    cumdist_continuous = varyings.cumdist_vertex;
                    cumdist_linear = cumdist_continuous;
                }

                // Define dash pattern, scale with (local) thickness
                var stroke_sizes = array<f32,dash_count>{{dash_pattern[::2]}};
                var gap_sizes = array<f32,dash_count>{{dash_pattern[1::2]}};
                for (var i=0; i<dash_count; i+=1) {
                    stroke_sizes[i] = stroke_sizes[i] * thickness_s;
                    gap_sizes[i] = gap_sizes[i] * thickness_s;
                }

                // Calculate the total dash size, and the size of the last gap. The dash_count is a const
                var dash_size = 0.0;
                var last_gap = 0.0;
                for (var i=0; i<dash_count; i+=1) {
                    dash_size += stroke_sizes[i];
                    last_gap = gap_sizes[i];
                    dash_size += last_gap;
                }

                // Calculate dash_progress, a number 0..dash_size, indicating the fase of the dash.
                // Except that we shift it, so that half of the final gap gets in front (as a negative number).
                let dash_progress = (cumdist_continuous + 0.5 * last_gap) % dash_size - 0.5 * last_gap;
                
                // Its looks a bit like this. Now we select the nearest stroke, and calculate the
                // distance to the beginning and end of that stroke. 
                //
                //  -0.5*last_gap      0                                              dash_size-0.5*last_gap     dash_size
                //     |               |                                                          |               |
                //     |---------------|XXXXXXXXXXXXXXX|-------|-------|XXXXXXXXXX|---------------|...............|
                //     |               |               |       |
                //  gap_begin    stroke_begin    stroke_end    gap_end (i.e. begin of next stroke)
                //
                var dist_to_begin = 0.0;
                var dist_to_end = 0.0;
                var gap_begin = -0.5 * last_gap;
                var stroke_begin = 0.0;
                for (var i=0; i<dash_count; i+=1) {
                    let half_gap_size = 0.5 * gap_sizes[i];
                    let stroke_end = stroke_begin + stroke_sizes[i];
                    let gap_end = stroke_end + half_gap_size;
                    if (dash_progress >= gap_begin && dash_progress <= gap_end) {
                        dist_to_begin = stroke_begin - dash_progress;
                        dist_to_end = dash_progress - stroke_end;
                        break;
                    }
                    // Next
                    gap_begin = gap_end;
                    stroke_begin = gap_end + half_gap_size;
                }

                // The distance to the dash's stoke is now easy to calculate.
                // Note that it's also possible to calculate dist_to_dash without dist_to_begin
                // and dist_to_end, but we need these for the trick in broken joins below.
                let dist_to_dash = max(0.0, max(dist_to_begin, dist_to_end));

                // Get cumdist scale factor
                let dpd_cumdist = length(vec2<f32>(dpdxFine(cumdist_linear), dpdyFine(cumdist_linear)));
                let dashdist_to_physical = 1.0 / dpd_cumdist;

                // Convert to (physical) pixel units
                let dist_to_begin_p = dist_to_begin * dashdist_to_physical;
                let dist_to_end_p = dist_to_end * dashdist_to_physical;
                let dist_to_dash_p = dist_to_dash * dashdist_to_physical;

                // At broken joins there is overlapping cumdist in both caps. The code below
                // avoids (not 100% prevents) the begin or end of a cap to be drawn twice.
                // The logic is basically: if we are in the cap (of a broken join), and if the
                // current dash would not be drawn in the segment attached to this cap, we
                // don't draw it here either.
                let is_broken_join = !is_join && segment_coord_p.x != 0.0;
                if (is_broken_join){
                    let dist_at_segment_p = select(dist_to_end_p, dist_to_begin_p, segment_coord_p.x > 0.0) + abs(segment_coord_p.x);
                    if (dist_at_segment_p > half_thickness_p) {
                       discard;
                    } 
                }

                // The vector to the stoke (at the line-center)
                var yy = length(segment_coord_p.y);
                if (abs(join_coord_lin) > 0.5) {
                    yy = length(segment_coord_p);  // smoother dash-turns
                }
                let vec_to_dash_p = vec2<f32>(dist_to_dash_p, yy);
                
                // Apply cap
                //let dist_to_stroke_dash_p = vec_to_dash_p.x;  // Butt caps
                let dist_to_stroke_dash_p = length(vec_to_dash_p) - half_thickness_p; // Round caps
               
                // Update dist_to_stroke_p with dash info
                dist_to_stroke_p = max(dist_to_stroke_p, dist_to_stroke_dash_p);

                // end dashing
            $$ endif

            // Anti-aliasing.
            // By default, the renderer uses SSAA (super-sampling), but if we apply AA for the edges
            // here this will help the end result. Because this produces semitransparent fragments,
            // it relies on a good blend method, and the object gets drawn twice.
            var alpha: f32 = 1.0;
            $$ if aa
                alpha = clamp(0.5 - dist_to_stroke_p, 0.0, 1.0);
                alpha = sqrt(alpha);  // this prevents aa lines from looking thinner
                if (alpha <= 0.0) { discard; }
            $$ else
                if (dist_to_stroke_p > 0.0) { discard; }
            $$ endif

            // Determine srgb color
            $$ if color_mode == 'vertex' or color_mode == 'face'
                let color = varyings.color;
            $$ elif color_mode == 'vertex_map' or color_mode == 'face_map'
                let color = sample_colormap(varyings.texcoord);
            $$ else
                let color = u_material.color;
            $$ endif

            // DEBUG
            $$ if false
                physical_color = vec3<f32>(abs(vec_to_dash_p.y / 20.0), 0.0, 0.0);
            $$ endif

            // Determine final rgba value            
            var physical_color = srgb2physical(color.rgb);
            let opacity = min(1.0, color.a) * alpha * u_material.opacity;
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