 fn get_vertex_result(
            index:i32, screen_factor:vec2<f32>, half_thickness:f32, l2p:f32
        ) -> VertexFuncOutput {
            //
            // This vertex shader uses VertexId and storage buffers instead of
            // vertex buffers. It creates 6 vertices for each point on the line.
            // The extra vertices are used to cover more fragments at
            // the joins and caps. In the fragment shader we discard fragments
            // that are "out of range" for the current join/cap shape, using
            // parameters passed as varyings.
            //
            // Definitions:
            //
            // - node: the positions that define the line. In other contexts these
            //   may be called vertices or points.
            // - vertex: the "virtual vertices" generated in the vertex shader,
            //   in order to create a thick line with nice joins and caps.
            // - segment: the straight piece of the line between two consecutive
            //   nodes. A quadrilateral (two faces) but not necesarily rectangular.
            // - join: the piece of the line to connect two segments. There are
            //   a few different shapes that can be applied.
            // - broken join: joins with too sharp corners are rendered as two
            //   separate segments with caps.
            // - cap: the beginning/end of the line and dashes. It typically extends
            //   a bit beyond the node (or dash end). There are multiple cap shapes.
            // - stroke: when dashing is enabled, the stoke represents the "on" piece.
            //   This is the visible piece to which caps are added. Can go over a
            //   join, i.e. is not always straight.
            //
            // Basic algorithm and definitions:
            //
            // - We read the positions of three nodes, the current, previous, and next.
            // - These are converted to logical pixel screen space.
            // - We define six normal vectors which represent the (virtual) vertices.
            //   The first two close the previous segment, the last two start the next
            //   segment, the two in the middle help define the join.
            // - These calculations are done for each vertex (yeah, bit of a waste),
            //   we select just one as output.
            //
            //            /  o     node 3
            //           /  /  /
            //          6  /  /
            //   - - - 2  /  /     segment-vertices 1, 2, 5, 6
            //   o-------o  /      the vertices 3 and 4 are in between to help the join
            //   - - - - 1 5
            //                node 2
            //  node 1
            //
            //
            // Possible improvements:
            //
            // - we can prepare the nodes' screen coordinates in a compute shader.

            // Indexing
            let i = index / 6;
            let sub_index = index % 6;
            let vertex_num = sub_index + 1;
            let fi = (index + 2) / 6;

            // Sample the current node and it's two neighbours, and convert to NDC
            // Note that if we sample out of bounds, this affects the shader in mysterious ways (21-12-2021).
            let node1n = get_point_ndc(max(0, i - 1));
            let node2n = get_point_ndc(i);
            let node3n = get_point_ndc(min(u_renderer.last_i, i + 1));

            // Convert to logical screen coordinates, because that's where the lines work
            let node1s = (node1n.xy / node1n.w + 1.0) * screen_factor;
            let node2s = (node2n.xy / node2n.w + 1.0) * screen_factor;
            let node3s = (node3n.xy / node3n.w + 1.0) * screen_factor;

            // Get vectors representing the two incident line segments
            var nodevec1: vec2<f32> = node2s.xy - node1s.xy;
            var nodevec2: vec2<f32> = node3s.xy - node2s.xy;

            var angle1 = atan2(nodevec1.y, nodevec1.x);
            var angle2 = atan2(nodevec2.y, nodevec2.x);

            // Declare (relative) vectors representing the 6 vertices.
            // These are relative to node2, and expressed as a coordinate that must be
            // scaled with half_line_width to get into screen space.
            var vert1: vec2<f32>;
            var vert2: vec2<f32>;
            var vert3: vec2<f32>;
            var vert4: vec2<f32>;
            var vert5: vec2<f32>;
            var vert6: vec2<f32>;

            // Declare matching line cords (x along line, y perpendicular to it)
            // The coords 1 and 5 have a positive y coord, the coords 2 and 6 a negative.
            var coord1: vec2<f32>;
            var coord2: vec2<f32>;
            var coord3: vec2<f32>;
            var coord4: vec2<f32>;
            var coord5: vec2<f32>;
            var coord6: vec2<f32>;

            // Valued for the valid varying. A triangle is dropped if all it's valid are one's.
            var valid_array = array<f32,6>(1.0, 1.0, 1.0, 1.0, 1.0, 1.0);

            var zero_cumdist_join = false;
            var vertex_is_inner_corner = false;

            // Whether the current vertex represents the join. Only nonzero for
            // sub_index 2 or 3, the signs is -1 and +1, respectively, signaling the side.
            // In the fragmnent shader this is used to determine whether the
            // vec_from_line or vec_from_node is used as the coord to sample the shape.
            var is_join = 0.0;  // todo: rename to vertex_is_outer_corner?

            // Whether this node represents a join, and thus not a cap or broken join (which has two caps).
            var node_is_join = false;

            // The offset of this vertex for the cumulative distance for dashing.
            // This value is expressed as a fraction of the segment length.
            // Negative means it relates to the segment before, positive means it
            // relates to the next segment.
            var dist_offset = 0.0;
            var dist_offset_multiplier = 1.0;

            if ( i == 0 || is_nan_or_zero(node1n.w) ) {
                // This is the first point on the line: create a cap.
                nodevec1 = nodevec2;
                angle1 = angle2;

                coord1 = vec2<f32>(-1.0, 1.0);
                coord2 = coord1;
                coord3 = coord2;
                coord4 = vec2<f32>(-1.0, -1.0);
                coord5 = vec2<f32>(0.0, 1.0);
                coord6 = vec2<f32>(0.0, -1.0);

                $$ if dashing
                    if (vertex_num <= 4) {
                        dist_offset = half_thickness / length(nodevec2);
                        dist_offset_multiplier = -1.0;
                    }
                $$ endif

            } else if ( i == u_renderer.last_i || is_nan_or_zero(node3n.w) )  {
                // This is the last point on the line: create a cap.
                nodevec2 = nodevec1;
                angle2 = angle1;

                coord1 = vec2<f32>(0.0, 1.0);
                coord2 = vec2<f32>(0.0, -1.0);
                coord3 = vec2<f32>(1.0, 1.0);
                coord4 = vec2<f32>(1.0, -1.0);
                coord5 = coord4;
                coord6 = coord4;

                $$ if dashing
                    if (vertex_num >= 3) {
                        dist_offset = -half_thickness / length(nodevec1);
                        dist_offset_multiplier = -1.0;
                    }
                $$ endif

            } else {
                // Create a join

                // TODO: if the line is solid and not dashed, it may be more performant to just draw the separate line segments (broken joins allways)

                // Determine the angle of the corner. If this angle is smaller than zero,
                // the inside of the join is at vert2/vert6, otherwise it is at vert1/vert5.
                let angle = atan2( nodevec1.x * nodevec2.y - nodevec1.y * nodevec2.x,
                                   nodevec1.x * nodevec2.x + nodevec1.y * nodevec2.y );

                // Which way does the join bent?
                let inner_corner_is_at_135 = angle >= 0.0;

                // The direction in which to place the vert3 and vert4.
                // TODO: maybe refactor this to not have to calculate vert1 and vert5
                vert1 = normalize(vec2<f32>(-nodevec1.y, nodevec1.x));
                vert5 = normalize(vec2<f32>(-nodevec2.y, nodevec2.x));
                let join_vec = normalize(vert1 + vert5);

                // Now calculate how far along this vector we can go without
                // introducing overlapping faces, which would result in glitchy artifacts.
                let nodevec1_norm = normalize(nodevec1);
                let nodevec2_norm = normalize(nodevec2);
                let join_vec_on_nodevec1 = dot(join_vec, nodevec1_norm) * nodevec1_norm;
                let join_vec_on_nodevec2 = dot(join_vec, nodevec2_norm) * nodevec2_norm;
                var max_vec_mag = {{ "1.5" if dashing else "100.0" }};  // 1.5 corresponds to about 90 degrees
                max_vec_mag = min(max_vec_mag, 0.49 * length(nodevec1) / length(join_vec_on_nodevec1) / half_thickness);
                max_vec_mag = min(max_vec_mag, 0.49 * length(nodevec2) / length(join_vec_on_nodevec2) / half_thickness);

                // Now use the angle to determine the join_vec magnitude required to draw this join.
                // For the inner corner this represent the intersection of the line edges,
                // i.e. the point where we should move the two other vertices-at-the-inner-corner to.
                // For the outer corner this represents the miter,
                // i.e. the extra space we need to draw the join shape.
                // Note that when the angle is ~pi, the magnitude is near infinity.
                let vec_mag = 1.0 / cos(0.5 * angle);

                // Clamp the magnitude with the limit we calculated above.
                let vec_mag_clamped = clamp(vec_mag, 1.0, max_vec_mag);

                // If the magnitude got clamped, we cannot draw the join as a contiguous line.
                var join_is_contiguous = vec_mag_clamped == vec_mag;

                if (join_is_contiguous) {
                    // Round or miter, shallow (enough) corner

                    node_is_join = true;

                    // Get the avg angle
                    let avg_vec = normalize(nodevec1) + normalize(nodevec2);
                    angle1 = atan2(avg_vec.y, avg_vec.x);
                    angle2 = angle1;

                    let half_angle = angle / 2.0;
                    coord1 = rotate_vec2(vec2<f32>(0.0, 1.0), -half_angle);
                    coord2 = -coord1;
                    coord5 = vec2<f32>(-coord1.x, coord1.y);  //rotate_vec2(vec2<f32>(0.0, 1.0), half_angle);
                    coord6 = -coord5;

                    coord3 = vec2<f32>(0.0, 1.0) * vec_mag_clamped;
                    coord4 = -coord3;

                    let dist_offset_inner_corner = distance(coord1, coord3);
                    let dist_offset_divisor = select(-length(nodevec1), length(nodevec2), vertex_num >= 4) / half_thickness;

                    // Put the 3 vertices in the inner corner at the same (center) position.
                    // Adjust the corner_coords in the same way, or they would not be correct.

                    // TODO: move this bit to the root and end of the function?
                    if (inner_corner_is_at_135) {

                        if (vertex_num == 1 || vertex_num == 3 || vertex_num == 5) {
                                vertex_is_inner_corner = true;
                        }

                        $$ if dashing

                            // This gives some cumdist-space in the corner, and works fine
                            // up to 90 degree corners
                            if (vertex_num == 1 || vertex_num == 3 || vertex_num == 5) {
                                zero_cumdist_join = true;
                                dist_offset = 1.0 * dist_offset_inner_corner / dist_offset_divisor;
                            }
                            if (vertex_num == 2 || vertex_num == 6) {
                                dist_offset = 1.0 * dist_offset_inner_corner / dist_offset_divisor;
                            }

                        $$ endif

                        let d1 = coord3 - coord1;
                        let d2 = coord3 - coord5;
                        coord1 = coord1 + d1;
                        coord5 = coord5 + d2;
                        coord2 = coord2 + d1;
                        coord6 = coord6 + d2;
        
                        is_join = -f32(vertex_num == 4);

                    } else {
                        
                        if (vertex_num == 2 || vertex_num == 4 || vertex_num == 6 ) {
                            vertex_is_inner_corner = true;
                        }

                        $$ if dashing
                            if (vertex_num == 2 || vertex_num == 4 || vertex_num == 6 ) {
                                zero_cumdist_join = true;
                                dist_offset = 1.0 * dist_offset_inner_corner / dist_offset_divisor;
                            }
                            if (vertex_num == 1 || vertex_num == 5) {
                               dist_offset = 1.0 * dist_offset_inner_corner / dist_offset_divisor;
                            }
                        $$ endif


                        let d1 = coord4 - coord2;
                        let d2 = coord4 - coord6;
                        coord2 = coord2 + d1;
                        coord6 = coord6 + d2;
                        coord1 = coord1 + d1;
                        coord5 = coord5 + d2;

                        is_join = f32(vertex_num == 3);
                    }

                } else {
                    // Broken join: render as separate segments with caps.

                    let miter_length = 4.0;

                    coord1 = vec2<f32>(          0.0, 1.0);
                    coord2 = vec2<f32>(          0.0, -1.0);
                    coord3 = vec2<f32>( miter_length, 0.0);
                    coord4 = vec2<f32>(-miter_length, 0.0);
                    coord5 = vec2<f32>(          0.0, 1.0);
                    coord6 = vec2<f32>(          0.0, -1.0);

                    valid_array[1] = 0.0;
                    valid_array[2] = 0.0;
                    valid_array[3] = 0.0;
                    valid_array[4] = 0.0;

                    $$ if dashing
                        if (vertex_num == 3) {
                            dist_offset = - miter_length * half_thickness / length(nodevec1);
                            dist_offset_multiplier = -1.0;
                        } else if (vertex_num == 4) {
                            dist_offset = miter_length * half_thickness / length(nodevec2);
                            dist_offset_multiplier = -1.0;
                        }
                    $$ endif
                }
            }

            vert1 = rotate_vec2(coord1, angle1);
            vert2 = rotate_vec2(coord2, angle1);
            vert3 = rotate_vec2(coord3, angle1);
            vert4 = rotate_vec2(coord4, angle2);
            vert5 = rotate_vec2(coord5, angle2);
            vert6 = rotate_vec2(coord6, angle2);

            // Select the current vector.
            var vert_array = array<vec2<f32>,6>(vert1, vert2, vert3, vert4, vert5, vert6);
            let the_vert_s = vert_array[sub_index] * half_thickness;
            let the_pos_s = node2s + the_vert_s;
            let the_pos_n = vec4<f32>((the_pos_s / screen_factor - 1.0) * node2n.w, node2n.zw);
            var coord_array = array<vec2<f32>,6>(coord1, coord2, coord3, coord4, coord5, coord6);

            let the_coord = coord_array[sub_index];

            // Calculate side
            let side = f32(vertex_num % 2) * 2.0 - 1.0;  // positive for coord1

            let segment_coord = select(the_coord, vec2<f32>(0.0, side), node_is_join);

            //let vec_from_node_p = (the_coord + side * vec2<f32>(0.0, -1.0)) * half_thickness * l2p;
            let vec_from_node_p = the_coord * half_thickness * l2p;

            // The join coord interpolates (in a join) from -1 to 0 and then from 0 to 1, in the
            // direction of the respective segments.
            // We also allow it to interpolate skewed, for the cumdist ... XXXX
            // To realize this, we use a "barycentric coords trick"
            // using a vec2, where the second element is a divisor that we apply in the frag shader.
            // This looks a bit like the w element for perspective division.
            var join_coord_x = select(-1.0, 1.0, vertex_num >= 4);
            join_coord_x = select(join_coord_x, 0.0, is_join != 0.0);
            let join_coord_multiplier = f32(!vertex_is_inner_corner);
            let join_coord = vec3<f32>(join_coord_x, join_coord_x * join_coord_multiplier, join_coord_multiplier);

            var out : VertexFuncOutput;
            out.i = i;
            out.fi = fi;
            out.pos = the_pos_n;
            out.thickness_p = half_thickness * 2.0 * l2p;
            out.vec_from_line_p = segment_coord * half_thickness * l2p;
            out.vec_from_node_p = vec_from_node_p;
            out.is_join = is_join;
            out.valid = valid_array[sub_index];
            out.side = the_coord.y; // todo: remove varying?
            out.dist_offset = dist_offset;
            out.dist_offset_multiplier = dist_offset_multiplier;
            out.zero_cumdist_join = zero_cumdist_join;
            out.join_coord = join_coord;

            return out;
        }