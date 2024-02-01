 fn get_vertex_result(
            index:i32, screen_factor:vec2<f32>, thickness:f32, l2p:f32
        ) -> VertexFuncOutput {
            //
            // This vertex shader uses VertexId and storage buffers instead of vertex buffers.
            // It creates 6 vertices for each point on the line, and a triangle-strip topology.
            // That gives 6 faces, of which 4 are used: 2 for the rectangular segment to the
            // previous join, and two for the join or cap(s). In each configuration 2 faces
            // are dropped.
            //
            // The resulting shapes are made up of triangles. In the fragment shader we discard
            // fragments depending on join and cap shapes, and we use aa for crisp edges.
            //
            // Definitions:
            //
            // - node: the positions that define the line. In other contexts these
            //   may be called vertices or points.
            // - vertex: the "virtual vertices" generated in the vertex shader,
            //   in order to create a thick line with nice joins and caps.
            // - segment: the rectangular piece of the line between two nodes.
            // - join: the piece of the line to connect two segments.
            // - broken join: joins with too sharp corners are rendered as two
            //   separate segments with caps.
            // - cap: the beginning/end of the line and dashes. It typically extends
            //   a bit beyond the node (or dash end). There are multiple cap shapes.
            // - stroke: when dashing is enabled, the stoke represents the "on" piece.
            //   This is the visible piece to which caps are added. Can go over a
            //   join, i.e. is not always straight. The gap is the off-piece.
            //
            // Basic algorithm and definitions:
            //
            // - We read the positions of three nodes, the previous, current, and next.
            // - These are converted to logical pixel screen space.
            // - We define six coordinate vectors which represent the (virtual) vertices.
            //   The first two end the previous segment, the last two start the next
            //   segment, the two in the middle help define the join/caps.
            // - To obtain the positions, the above coordinates are rotated, added to the
            //   node positions, and then converted to ndc.
            // - To some degree these calculations are done for all 6 vertices, and the
            //   one corresponding to the current vertex_index is selected.
            //
            //            /  o     node 3
            //           /  /  /
            //        5 /  /  /
            //   - - - 1  /  /     segment-vertices 1, 2, 5, 6
            //   o-------o  6      the vertices 3 and 4 are in the outer corner.
            //   - - - 2 - 34
            //                node 2
            //  node 1
            //

            // Indexing
            let node_index = index / 6;
            let vertex_index = index % 6;
            let vertex_num = vertex_index + 1;
            let face_index = (index + 2) / 6;

            // Sample the current node and it's two neighbours, and convert to NDC
            // Note that if we sample out of bounds, this affects the shader in mysterious ways (21-12-2021).
            let node1n = get_point_ndc(max(0, node_index - 1));
            let node2n = get_point_ndc(node_index);
            let node3n = get_point_ndc(min(u_renderer.last_i, node_index + 1));

            // Convert to logical screen coordinates, because that's where the lines work
            let node1s = (node1n.xy / node1n.w + 1.0) * screen_factor;
            let node2s = (node2n.xy / node2n.w + 1.0) * screen_factor;
            let node3s = (node3n.xy / node3n.w + 1.0) * screen_factor;

            // Get vectors representing the two incident line segments
            var nodevec1: vec2<f32> = node2s.xy - node1s.xy;
            var nodevec2: vec2<f32> = node3s.xy - node2s.xy;

            // Calculate the angle between them. We use this at the end to rotate the coord.
            var angle1 = atan2(nodevec1.y, nodevec1.x);
            var angle2 = atan2(nodevec2.y, nodevec2.x);

            // The thickness of the line in terms of geometry is a wee bit thicker.
            // Just enough so that fragments that are partially on the line, are also included
            // in the fragment shader. That way we can do aa without making the lines thinner.
            // All logic in this function works with the ticker line width. But we pass the real line width as a varying.
            let extra_thick = 0.5 / l2p;  // on each side.
            let half_thickness = 0.5 * thickness + extra_thick * {{ '1.0' if aa else '0.0' }};

            // Declare vertex cords (x along segment, y perpendicular to it).
            // The coords 1 and 5 have a positive y coord, the coords 2 and 6 negative.
            // These values are relative to the line width.
            var coord1: vec2<f32>;
            var coord2: vec2<f32>;
            var coord3: vec2<f32>;
            var coord4: vec2<f32>;
            var coord5: vec2<f32>;
            var coord6: vec2<f32>;

            // The vertex inset, in coord-coords. Is set for joins to keep the segments rectangular.
            // The value will depend on the angle between the segments, and the line thickness.
            var vertex_inset = 0.0;

            // Array for the valid_if_nonzero varying. A triangle is dropped if (and only if) all verts have their value set to zero. (Trick 5)
            var valid_array = array<f32,6>(1.0, 1.0, 1.0, 1.0, 1.0, 1.0);

            // Whether this node represents a join, and thus not a cap or broken join (which has two caps).
            // Used internally in this shader (not a varying).
            var node_is_join = false;

            // The join_coord. If this is a join, the value is 1.0 and -1.0 for vertex_num 3 and 4, respectively.
            // In the fragment shader we also identify join faces with it (trick 5).
            var join_coord = 0.0;

            // In joins, this is 1.0 for the vertices in the outer corner.
            var is_outer_corner = 0.0;

            $$ if dashing
                // The offset of this vertex for the cumulative distance for dashing.
                // This value is expressed as a fraction of the segment length.
                // Negative means it relates to the segment before, positive means it
                // relates to the next segment. The multiplier (sign) is used so we can extrapolate in the caps.
                var cumdist_offset = 0.0;
                var cumdist_multiplier = 1.0;
            $$ endif

            if ( node_index == 0 || is_nan_or_zero(node1n.w) ) {
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
                        cumdist_offset = half_thickness / length(nodevec2);
                        cumdist_multiplier = -1.0;
                    }
                $$ endif

            } else if ( node_index == u_renderer.last_i || is_nan_or_zero(node3n.w) )  {
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
                        cumdist_offset = -half_thickness / length(nodevec1);
                        cumdist_multiplier = -1.0;
                    }
                $$ endif

            } else {
                // Create a join

                // Determine the angle of the corner. If this angle is smaller than zero,
                // the inside of the join is at vert2/vert6, otherwise it is at vert1/vert5.
                let angle = atan2( nodevec1.x * nodevec2.y - nodevec1.y * nodevec2.x,
                                   nodevec1.x * nodevec2.x + nodevec1.y * nodevec2.y );

                // Which way does the join bent?
                let inner_corner_is_at_15 = angle >= 0.0;

                // The direction in which to place the vert3 and vert4.
                let vert1 = normalize(vec2<f32>(-nodevec1.y, nodevec1.x));
                let vert5 = normalize(vec2<f32>(-nodevec2.y, nodevec2.x));
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
                // For the inner corner this represents the intersection of the line edges,
                // i.e. the point where we should move both vertices at the inner corner to.
                // For the outer corner this represents the miter, i.e. the extra space we need to draw the join shape.
                // Note that when the angle is ~pi, the magnitude is near infinity.
                let vec_mag = 1.0 / cos(0.5 * angle);

                // Clamp the magnitude with the limit we calculated above.
                let vec_mag_clamped = clamp(vec_mag, 1.0, max_vec_mag);

                // If the magnitude got clamped, we cannot draw the join as a contiguous line.
                var join_is_contiguous = vec_mag_clamped == vec_mag;

                if (!join_is_contiguous) {
                    // Create a broken join: render as separate segments with caps.

                    let miter_length = 4.0;

                    coord1 = vec2<f32>(          0.0,  1.0);
                    coord2 = vec2<f32>(          0.0, -1.0);
                    coord3 = vec2<f32>( miter_length,  0.0);
                    coord4 = vec2<f32>(-miter_length,  0.0);
                    coord5 = vec2<f32>(          0.0,  1.0);
                    coord6 = vec2<f32>(          0.0, -1.0);

                    // Drop two triangles in between
                    valid_array[1] = 0.0;
                    valid_array[2] = 0.0;
                    valid_array[3] = 0.0;
                    valid_array[4] = 0.0;

                    $$ if dashing
                        if (vertex_num == 3) {
                            cumdist_offset = - miter_length * half_thickness / length(nodevec1);
                            cumdist_multiplier = -1.0;
                        } else if (vertex_num == 4) {
                            cumdist_offset = miter_length * half_thickness / length(nodevec2);
                            cumdist_multiplier = -1.0;
                        }
                    $$ endif
                    
                } else {
                    // Create a proper join

                    node_is_join = true;
                    join_coord = f32(vertex_num == 3) - f32(vertex_num == 4);

                    // The gap between the segment's end (at the node) and the intersection.
                    vertex_inset = tan(abs(0.5 * angle)) * 1.0;

                    // Vertex 3 and 4 are both in the ourer corner.
                    let sign34 = select(1.0, -1.0, inner_corner_is_at_15);
                    
                    // Express coords in segment coordinates. 
                    // Note that coord3 and coord4 are different, but the respective vertex positions will be the same (except for float inaccuraries).
                    coord1 = vec2<f32>(0.0, 1.0);
                    coord2 = vec2<f32>(0.0, -1.0);
                    coord3 = vec2<f32>( 2.0 * vertex_inset, sign34);
                    coord4 = vec2<f32>(-2.0 * vertex_inset, sign34);
                    coord5 = vec2<f32>(0.0, 1.0);
                    coord6 = vec2<f32>(0.0, -1.0);
                    
                    // Get wheter this is an outer corner
                    let vertex_num_is_even = (vertex_num % 2) == 0;
                    if (inner_corner_is_at_15) { 
                        is_outer_corner = f32(vertex_num_is_even || vertex_num == 3);
                    } else {
                        is_outer_corner = f32((!vertex_num_is_even) || vertex_num == 4);
                    }

                    $$ if dashing
                        if (!(vertex_num == 3 || vertex_num == 4)) {
                            cumdist_offset = vertex_inset * half_thickness / select(-length(nodevec1), length(nodevec2), vertex_num >= 4);
                        }
                    $$ endif
                }
            }

            // Select the current coord
            var coord_array = array<vec2<f32>,6>(coord1, coord2, coord3, coord4, coord5, coord6);
            let the_coord = coord_array[vertex_index];

            // Calculate the relative vertex, in screen coords, from the coord.
            // If the vertex_num is 4, the resulting vertex should be the same, but it might not be
            // due to floating point errors. So we use the coord3-path in that case.
            let override_use_coord3 = node_is_join && vertex_num == 4;
            let use_456 = vertex_num >= 4 && !override_use_coord3;
            let vertex_offset = vec2<f32>(select(-vertex_inset, vertex_inset, use_456), 0.0);
            let ref_coord = select(the_coord, coord3, override_use_coord3);
            let ref_angle = select(angle1, angle2, use_456);
            let relative_vert_s = rotate_vec2(ref_coord + vertex_offset, ref_angle) * half_thickness;

            // Calculate vertex position.
            let the_pos_s = node2s + relative_vert_s;
            let the_pos_n = vec4<f32>((the_pos_s / screen_factor - 1.0) * node2n.w, node2n.zw);

            // Build output
            var out : VertexFuncOutput;
            out.node_index = node_index;
            out.face_index = face_index;
            out.pos = the_pos_n;
            // Varyings
            out.thickness_p = thickness * l2p;  // the real thickness
            out.segment_coord_p = the_coord * half_thickness * l2p;  // uses a slightly wider thickness
            out.join_coord = join_coord;
            out.is_outer_corner = is_outer_corner;
            out.valid_if_nonzero = valid_array[vertex_index];
            
            $$ if dashing
                // Calculate the cumdist for the node and vertex edge
                let cumdist_node = f32(load_s_cumdist(node_index));
                var cumdist_vertex = cumdist_node;  // Important default, see frag-shader.
                if (cumdist_offset < 0.0) {
                    let cumdist_before = f32(load_s_cumdist(node_index - 1));
                    cumdist_vertex = cumdist_node + cumdist_multiplier * cumdist_offset * (cumdist_node - cumdist_before);
                } else if (cumdist_offset > 0.0) {
                    let cumdist_after = f32(load_s_cumdist(node_index + 1));
                    cumdist_vertex = cumdist_node + cumdist_multiplier * cumdist_offset * (cumdist_after - cumdist_node);
                }
                // Set two varyings, so that we can correctly interpolate the cumdist in the joins
                out.cumdist_node = cumdist_node;
                out.cumdist_vertex = cumdist_vertex;
            $$ endif

            return out;
        }