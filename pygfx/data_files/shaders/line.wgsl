// Line shader
// 
// The vertex shader uses VertexId and storage buffers instead of vertex buffers.
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


struct VertexInput {
    @builtin(vertex_index) index : u32,
};


fn is_nan_or_zero(v:f32) -> bool {
    // Naga has removed isNan checks, because backends may be using fast-math,
    // in which case nan is assumed not to happen, and isNan would always be false.
    // If we assume that some nan mechanics still work, we can still detect it.
    // This won't work however: `return v != v`, because the compiler will
    // optimize it out. The same holds for similar constructs.
    // Maybe the same happens if we turn `<`  into `<=`.
    // So we and up with an equation that detects either NaN or zero,
    // which is fine if we use it on a .w attribute.
    return !(v < 0.0) && !(v > 0.0);
}


fn rotate_vec2(v:vec2<f32>, angle:f32) -> vec2<f32> {
    return vec2<f32>(cos(angle) * v.x - sin(angle) * v.y, sin(angle) * v.x + cos(angle) * v.y);
}


// -------------------- vertex shader --------------------


@vertex
fn vs_main(in: VertexInput) -> Varyings {

    let screen_factor:vec2<f32> = u_stdinfo.logical_size.xy / 2.0;
    let l2p:f32 = u_stdinfo.physical_size.x / u_stdinfo.logical_size.x;

    // Indexing
    let index = i32(in.index);
    let node_index = index / 6;
    let vertex_index = index % 6;
    let vertex_num = vertex_index + 1;
    var face_index = node_index;  // corrected below, depending on configuration

    // Sample the current node and it's two neighbours. Model coords.
    // Note that if we sample out of bounds, this affects the shader in mysterious ways (21-12-2021).
    let node1m = load_s_positions(max(0, node_index - 1));
    let node2m = load_s_positions(node_index);
    let node3m = load_s_positions(min(u_renderer.last_i, node_index + 1));
    // Convert to world
    let node1w = u_wobject.world_transform * vec4<f32>(node1m.xyz, 1.0);
    let node2w = u_wobject.world_transform * vec4<f32>(node2m.xyz, 1.0);
    let node3w = u_wobject.world_transform * vec4<f32>(node3m.xyz, 1.0);
    // Convert to camera view
    let node1c = u_stdinfo.cam_transform * node1w;
    let node2c = u_stdinfo.cam_transform * node2w;
    let node3c = u_stdinfo.cam_transform * node3w;
    // convert to NDC
    let node1n = u_stdinfo.projection_transform * node1c;
    let node2n = u_stdinfo.projection_transform * node2c;
    let node3n = u_stdinfo.projection_transform * node3c;
    // Convert to logical screen coordinates, because that's where the lines work
    let node1s = (node1n.xy / node1n.w + 1.0) * screen_factor;
    let node2s = (node2n.xy / node2n.w + 1.0) * screen_factor;
    let node3s = (node3n.xy / node3n.w + 1.0) * screen_factor;

    // Get vectors representing the two incident line segments
    var nodevec1: vec2<f32> = node2s.xy - node1s.xy;  // from node 1 (to node 2)
    var nodevec3: vec2<f32> = node3s.xy - node2s.xy;  // to node 3 (from node 2)

    // Calculate the angle between them. We use this at the end to rotate the coord.
    var angle1 = atan2(nodevec1.y, nodevec1.x);
    var angle3 = atan2(nodevec3.y, nodevec3.x);

    // The thickness of the line in terms of geometry is a wee bit thicker.
    // Just enough so that fragments that are partially on the line, are also included
    // in the fragment shader. That way we can do aa without making the lines thinner.
    // All logic in this function works with the ticker line width. But we pass the real line width as a varying.
    $$ if thickness_space == "screen"
        let thickness_ratio = 1.0;
    $$ else
        // The thickness is expressed in world space. So we first check where a point, moved 1 logic pixel away
        // from the node, ends up in world space. We actually do that for both x and y, in case there's anisotropy.
        let node2s_shiftedx = node2s + vec2<f32>(1.0, 0.0);
        let node2s_shiftedy = node2s + vec2<f32>(0.0, 1.0);
        let node2n_shiftedx = vec4<f32>((node2s_shiftedx / screen_factor - 1.0) * node2n.w, node2n.z, node2n.w);
        let node2n_shiftedy = vec4<f32>((node2s_shiftedy / screen_factor - 1.0) * node2n.w, node2n.z, node2n.w);
        let node2w_shiftedx = u_stdinfo.cam_transform_inv * u_stdinfo.projection_transform_inv * node2n_shiftedx;
        let node2w_shiftedy = u_stdinfo.cam_transform_inv * u_stdinfo.projection_transform_inv * node2n_shiftedy;
        $$ if thickness_space == "model"
            // Transform back to model space
            let node2m_shiftedx = u_wobject.world_transform_inv * node2w_shiftedx;
            let node2m_shiftedy = u_wobject.world_transform_inv * node2w_shiftedy;
            // Distance in model space
            let thickness_ratio = 0.5 * (distance(node2m.xyz, node2m_shiftedx.xyz) + distance(node2m.xyz, node2m_shiftedy.xyz));
        $$ else
            // Distance in world space
            let thickness_ratio = 0.5 * (distance(node2w.xyz, node2w_shiftedx.xyz) + distance(node2w.xyz, node2w_shiftedy.xyz));
        $$ endif 
    $$ endif
    let thickness:f32 = u_material.thickness / thickness_ratio;  // Logical pixels
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

    // A value to offset certain varyings towards the neighbouring nodes. The first value
    // represents moving towards node1, the second value represents moving towards node2.
    // Only one should be nonzero. A negative value results in extrapolation (in the other direction).
    // Used to set correct values for e.g. depth, cumdist, per-vertex colors.
    var offset_ratio_multiplier = vec2<f32>(0.0, 0.0);

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

    // Determine whether to draw a cap. Either on the left, the right, or both! In the latter case
    // we draw a double-cap, which results in a circle for round caps.
    // A cap is needed when:
    // - This is the first / last point on the line.
    // - The neighbouring node is nan.
    // - The neighbouring node is equal: length(nodevec) < eps (because round-off errors)
    // - If the line segment's direction has a significant component in the camera view direction,
    //   i.e. a depth component, then a cap is created if there is sufficient overlap with the neighbouring cap.
    //   This prevents that the extrapolation of the segment's cap takes up a large portion of the screen.

    // Is this a line that "goes deep"?
    let nodevec1_c = vec3<f32>(node2c.xyz - node1c.xyz);
    let nodevec3_c = vec3<f32>(node3c.xyz - node2c.xyz);
    let nodevec1_has_significant_depth_component = abs(nodevec1_c.z) > 1.0 * length(nodevec1_c.xy);
    let nodevec3_has_significant_depth_component = abs(nodevec3_c.z) > 1.0 * length(nodevec3_c.xy);
    // Determine capp-ness
    let minor_dist_threshold = 0.0001;
    let major_dist_threshold = 0.5 * max(1.0, half_thickness);
    var left_is_cap = is_nan_or_zero(node1n.w) || length(nodevec1) < select(minor_dist_threshold, major_dist_threshold, nodevec1_has_significant_depth_component);
    var right_is_cap = is_nan_or_zero(node3n.w) || length(nodevec3) < select(minor_dist_threshold, major_dist_threshold, nodevec3_has_significant_depth_component);

    $$ if line_type in ['segment', 'arrow']
        // Implementing segments is pretty easy
        left_is_cap = left_is_cap || node_index % 2 == 0;
        right_is_cap = right_is_cap || node_index % 2 == 1;
    $$ endif

    // The big triage ...

    if (left_is_cap && right_is_cap) {
        // Create two caps
        nodevec1 = vec2<f32>(0.0, 0.0);
        nodevec3 = nodevec1;
        angle1 = 0.0;
        angle3 = 0.0;

        coord1 = vec2<f32>(-1.0, 1.0);
        coord2 = coord1;
        coord3 = vec2<f32>(-1.0, -1.0);
        coord4 = vec2<f32>(1.0, 1.0);
        coord5 = vec2<f32>(1.0, -1.0);
        coord6 = coord5;

    } else if (left_is_cap) {
        /// Create a cap using vertex 4, 5, 6
        nodevec1 = nodevec3;
        angle1 = angle3;

        coord1 = vec2<f32>(-1.0, 1.0);
        coord2 = coord1;
        coord3 = coord2;
        coord4 = vec2<f32>(-1.0, -1.0);
        coord5 = vec2<f32>(0.0, 1.0);
        coord6 = vec2<f32>(0.0, -1.0);

        if (vertex_num <= 4) { 
            offset_ratio_multiplier = vec2<f32>(0.0, - 1.0);
        }

    } else if (right_is_cap)  {
        // Create a cap using vertex 4, 5, 6
        nodevec3 = nodevec1;
        angle3 = angle1;

        coord1 = vec2<f32>(0.0, 1.0);
        coord2 = vec2<f32>(0.0, -1.0);
        coord3 = vec2<f32>(1.0, 1.0);
        coord4 = vec2<f32>(1.0, -1.0);
        coord5 = coord4;
        coord6 = coord4;

        if (vertex_num >= 3) {
            offset_ratio_multiplier = vec2<f32>(- 1.0, 0.0);
        }
        face_index = face_index - 1;  // belongs to previous face

    } else {
        face_index = face_index - i32(vertex_num <= 3);

        $$ if line_type == 'quickline'

        // Joins in quick lines are always broken
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
        // Handle offset
        if (vertex_num == 3) {
            offset_ratio_multiplier = vec2<f32>(-miter_length, 0.0);
        } else if (vertex_num == 4) {
            offset_ratio_multiplier = vec2<f32>(0.0, -miter_length); 
        }

        $$ elif line_type == 'line'

        // Create a join

        // Determine the angle of the corner. If this angle is smaller than zero,
        // the inside of the join is at vert2/vert6, otherwise it is at vert1/vert5.
        let angle = atan2( nodevec1.x * nodevec3.y - nodevec1.y * nodevec3.x,
                            nodevec1.x * nodevec3.x + nodevec1.y * nodevec3.y );

        // Which way does the join bent?
        let inner_corner_is_at_15 = angle >= 0.0;

        // The direction in which to place the vert3 and vert4.
        let vert1 = normalize(vec2<f32>(-nodevec1.y, nodevec1.x));
        let vert5 = normalize(vec2<f32>(-nodevec3.y, nodevec3.x));
        let join_vec = normalize(vert1 + vert5);

        // Now calculate how far along this vector we can go without
        // introducing overlapping faces, which would result in glitchy artifacts.
        let nodevec1_norm = normalize(nodevec1);
        let nodevec3_norm = normalize(nodevec3);
        let join_vec_on_nodevec1 = dot(join_vec, nodevec1_norm) * nodevec1_norm;
        let join_vec_on_nodevec3 = dot(join_vec, nodevec3_norm) * nodevec3_norm;
        var max_vec_mag = {{ "1.5" if dashing else "100.0" }};  // 1.5 corresponds to about 90 degrees
        max_vec_mag = min(max_vec_mag, 0.49 * length(nodevec1) / length(join_vec_on_nodevec1) / half_thickness);
        max_vec_mag = min(max_vec_mag, 0.49 * length(nodevec3) / length(join_vec_on_nodevec3) / half_thickness);

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

            if (vertex_num == 3) {
                offset_ratio_multiplier = vec2<f32>(-miter_length, 0.0);
            } else if (vertex_num == 4) {
                offset_ratio_multiplier = vec2<f32>(0.0, -miter_length); 
            }
            
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
            // These represent the segment coords. They are also used to calculate the vertex position, by rotating it and adding to node2.
            // However, the point of rotation will be shifted with the vertex_inset (see use of vertex_inset further down).
            coord1 = vec2<f32>(0.0, 1.0);
            coord2 = vec2<f32>(0.0, -1.0);
            coord3 = vec2<f32>( 2.0 * vertex_inset, sign34);
            coord4 = vec2<f32>(-2.0 * vertex_inset, sign34);
            coord5 = vec2<f32>(0.0, 1.0);
            coord6 = vec2<f32>(0.0, -1.0);
            
            // For 
            if ( vertex_num <= 2) {
                offset_ratio_multiplier = vec2<f32>(vertex_inset, 0.0);
            } else if (vertex_num >= 5) {
                offset_ratio_multiplier = vec2<f32>(0.0, vertex_inset);
            }

            // Get wheter this is an outer corner
            let vertex_num_is_even = (vertex_num % 2) == 0;
            if (inner_corner_is_at_15) { 
                is_outer_corner = f32(vertex_num_is_even || vertex_num == 3);
            } else {
                is_outer_corner = f32((!vertex_num_is_even) || vertex_num == 4);
            }
        }
        $$ endif
    }

    // Zero the multiplier if the divisor is going to be zero
    offset_ratio_multiplier = offset_ratio_multiplier * vec2<f32>(f32(length(nodevec1) > 0.0), f32(length(nodevec3) > 0.0));
    
    // Prepare values for applying offset_ratio_multiplier
    var z = node2n.z;
    var w = node2n.w;
    $$ if dashing
        let cumdist_node = f32(load_s_cumdist(node_index));
        var cumdist_vertex = cumdist_node;  // Important default, see frag-shader.
    $$ endif
    $$ if color_mode == 'vertex'
        let color_node = load_s_colors(node_index);  // type depends on color depth
        var color_vert = color_node;
    $$ endif
    
    // Interpolate / extrapolate
    if (offset_ratio_multiplier.x != 0.0) {
        // Get ratio in screen space, and then correct for perspective.
        // I derived that step by calculating the new w from the ratio, and then substituting terms.
        var ratio = offset_ratio_multiplier.x * half_thickness / length(nodevec1);
        ratio = (1.0 - ratio) * ratio * node2n.w / node1n.w + ratio * ratio;
        // Interpolate on the left
        z = mix(z, node1n.z, ratio);
        w = mix(w, node1n.w, ratio);
        $$ if dashing
            let cumdist_before = f32(load_s_cumdist(node_index - 1));
            cumdist_vertex = mix(cumdist_node, cumdist_before, ratio);
        $$ endif
        if (node_is_join) {
            // Some values only interpolate for joins
            $$ if color_mode == 'vertex'
                let color_before = load_s_colors(node_index - 1);
                color_vert = mix(color_vert, color_before, ratio);
             $$ endif
        }
    } else if (offset_ratio_multiplier.y != 0.0) {
         // Get ratio in screen space, and then correct for perspective.
        var ratio = offset_ratio_multiplier.y * half_thickness / length(nodevec3);
        ratio =  (1.0 - ratio) * ratio * node2n.w / node3n.w + ratio * ratio;
        // Interpolate on the right
        z = mix(z, node3n.z, ratio);
        w = mix(w, node3n.w, ratio);
        $$ if dashing
            let cumdist_after = f32(load_s_cumdist(node_index + 1));
            cumdist_vertex = mix(cumdist_node, cumdist_after, ratio);
        $$ endif
        if (node_is_join) {
            // Some values only interpolate for joins
            $$ if color_mode == 'vertex'
                let color_after = load_s_colors(node_index + 1); 
                color_vert = mix(color_vert, color_after, ratio);
            $$ endif
        }
    }

    // Select the current coord
    var coord_array = array<vec2<f32>,6>(coord1, coord2, coord3, coord4, coord5, coord6);
    let the_coord = coord_array[vertex_index];

    // Calculate the relative vertex, in screen coords, from the coord.
    // If the vertex_num is 4, the resulting vertex should be the same as 3, but it might not be
    // due to floating point errors. So we use the coord3-path in that case.
    let override_use_coord3 = node_is_join && vertex_num == 4;
    let use_456 = vertex_num >= 4 && !override_use_coord3;
    let vertex_offset = vec2<f32>(select(-vertex_inset, vertex_inset, use_456), 0.0);
    let ref_coord = select(the_coord, coord3, override_use_coord3);
    let ref_angle = select(angle1, angle3, use_456);
    let relative_vert_s = rotate_vec2(ref_coord + vertex_offset, ref_angle) * half_thickness;

    // Calculate vertex position.
    // NOTE: the extrapolated positions (most notably the caps) are at the same depth as the node.
    // From the use-cases I have seen so far this is not a problem (might even be favorable?),
    // but something to keep in mind when someone encounters unexpected behavior related to caps and depth.
    let the_pos_s = node2s + relative_vert_s;
    let the_pos_n = vec4<f32>((the_pos_s / screen_factor - 1.0) * w, z, w);

    // Build varyings output
    
    var varyings: Varyings;
    // Position
    varyings.position = vec4<f32>(the_pos_n);
    varyings.world_pos = vec3<f32>(ndc_to_world_pos(the_pos_n));
    //  Thickness and segment coord. These are corrected for perspective, otherwise the dashes are malformed in 3D.
    varyings.w = f32(w);
    varyings.thickness_pw = f32(thickness * l2p * w);  // the real thickness, in physical coords
    varyings.segment_coord_pw = vec2<f32>(the_coord * half_thickness * l2p * w);  // uses a slightly wider thickness
    // Coords related to joins
    varyings.join_coord = f32(join_coord);
    varyings.is_outer_corner = f32(is_outer_corner);
    varyings.valid_if_nonzero = f32(valid_array[vertex_index]);
    
    $$ if debug
        // Include barycentric coords so we can draw the triangles that make up the line
        varyings.bary = vec3<f32>(f32(vertex_index % 3 == 0), f32(vertex_index % 3 == 1), f32(vertex_index % 3 == 2));
    $$ endif
    $$ if dashing
        // Set two varyings, so that we can correctly interpolate the cumdist in the joins.
        // If the thickness is in screen space, we need to correct for perspective division
        varyings.cumdist_node = f32(cumdist_node)  {{ '* w' if thickness_space == "screen" else '' }};
        varyings.cumdist_vertex = f32(cumdist_vertex)  {{ '* w' if thickness_space == "screen" else '' }};
    $$ endif
    $$ if line_type == 'arrow'
        // Include coord that goes from 0 to 1 over the segment, so we can shape the arrow
        varyings.line_type_segment_coord = f32(node_index % 2);
    $$ endif

    // Picking
    // Note: in theory, we can store ints up to 16_777_216 in f32,
    // but in practice, its about 4_000_000 for f32 varyings (in my tests).
    // We use a real u32 to not lose presision, see frag shader for details.
    varyings.pick_idx = u32(node_index);
    varyings.pick_zigzag = f32(select(0.0, 1.0, node_index % 2 == 0));

    // per-vertex or per-face coloring
    $$ if color_mode == 'face' or color_mode == 'vertex'
        $$ if color_mode == 'face'
            let color_node = load_s_colors(face_index);
            let color_vert = color_node;
        $$ else
            // The color_node and color_vert are defined (and interpolated) above.
        $$ endif
        $$ if color_buffer_channels == 1
            varyings.color_node = vec4<f32>(color_node, color_node, color_node, 1.0);
            varyings.color_vert = vec4<f32>(color_vert, color_vert, color_vert, 1.0);
        $$ elif color_buffer_channels == 2
            varyings.color_node = vec4<f32>(color_node.r, color_node.r, color_node.r, color_node.g);
            varyings.color_vert = vec4<f32>(color_vert.r, color_vert.r, color_vert.r, color_vert.g);
        $$ elif color_buffer_channels == 3
            varyings.color_node = vec4<f32>(color_node, 1.0);
            varyings.color_vert = vec4<f32>(color_vert, 1.0);
        $$ elif color_buffer_channels == 4
            varyings.color_node = vec4<f32>(color_node);
            varyings.color_vert = vec4<f32>(color_vert);
        $$ endif
    $$ endif

    // How to index into tex-coords
    $$ if color_mode == 'face_map'
    let tex_coord_index = face_index;
    $$ else
    let tex_coord_index = node_index;
    $$ endif

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


// --------------------  fragment shader --------------------


$$ if dashing
    const dash_count = {{dash_count}};
$$ endif


@fragment
fn fs_main(varyings: Varyings, @builtin(front_facing) is_front: bool) -> FragmentOutput {
            
    // Get the half-thickness in physical coordinates. This is the reference thickness.
    // If aa is used, the line is actually a bit thicker, leaving space to do aa.
    let half_thickness_p = 0.5 * varyings.thickness_pw / varyings.w; 

    // Discard invalid faces. These are faces for which *all* 3 verts are set to zero. (trick 5b)
    if (varyings.valid_if_nonzero == 0.0) {
        discard;
    }

    // Determine whether we are at a join (i.e. an unbroken corner).
    // These are faces for which *any* vert is nonzero. (trick 5a)
    $$ if line_type == 'quickline'
    let is_join = false;  // hard-coded to false. I'm assuming the Naga optimizer will eliminate dead code.
    $$ else
    let is_join = varyings.join_coord != 0.0;
    $$ endif

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
    var segment_coord_p = varyings.segment_coord_pw / varyings.w;
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
        $$ if thickness_space == "screen"
            cumdist_continuous = cumdist_continuous / varyings.w;
            cumdist_linear = cumdist_linear / varyings.w;
        $$ endif

        // Define dash pattern, scale with (uniform) thickness.
        // Note how the pattern is templated (triggering recompilation when it changes), wheras the thickness is a uniform.
        var stroke_sizes = array<f32,dash_count>{{dash_pattern[::2]}};
        var gap_sizes = array<f32,dash_count>{{dash_pattern[1::2]}};
        for (var i=0; i<dash_count; i+=1) {
            stroke_sizes[i] = stroke_sizes[i] * u_material.thickness;
            gap_sizes[i] = gap_sizes[i] * u_material.thickness;
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

    $$ if debug
        // In debug-mode, use barycentric coords to draw the edges of the faces.
        dist_to_stroke_p = -1.0;
        if (min(varyings.bary.x, min(varyings.bary.y, varyings.bary.z)) > 0.1) {
            dist_to_stroke_p = 1.0;
        }
    $$ endif

    $$ if line_type == 'arrow' 
        // Arrow shape
        let arrow_head_factor = 1.0 - varyings.line_type_segment_coord;
        let arrow_tail_factor = 1.0 - varyings.line_type_segment_coord * 3.0;
        dist_to_stroke_p = max(
                abs(segment_coord_p.y) - half_thickness_p * arrow_head_factor,
                half_thickness_p * arrow_tail_factor- abs(segment_coord_p.y)
        );
        // Ignore caps
        dist_to_stroke_p = select(dist_to_stroke_p, 9999999.0, segment_coord_p.x != 0.0);
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
    $$ if color_mode == 'vertex'
        var color = varyings.color_vert;
        if (is_join) {
            let color_segment = varyings.color_node - (varyings.color_node - varyings.color_vert) / (1.0 - abs(join_coord_lin));
            color = mix(color_segment, varyings.color_node, abs(join_coord_fan));
        }
    $$ elif color_mode == 'face'
        let color = varyings.color_vert;
    $$ elif color_mode == 'vertex_map' or color_mode == 'face_map'
        let color = sample_colormap(varyings.texcoord);
    $$ else
        let color = u_material.color;
    $$ endif
    var physical_color = srgb2physical(color.rgb);

    $$ if false
        // Alternative debug options during dev.
        physical_color = vec3<f32>(abs(cumdist_linear % 1.0) / 1.0, 0.0, 0.0);
    $$ endif

    // Determine final rgba value            
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
