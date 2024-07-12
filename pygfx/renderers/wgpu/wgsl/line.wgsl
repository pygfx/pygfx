// # Line shader
//
// ## References:
//
// * https://almarklein.org/line_rendering.html -> for a high level understaning of what this shader does.
// * https://almarklein.org/triangletricks.html -> for an explanation of some of the referred tricks.
// * https://jcgt.org/published/0002/02/08/paper.pdf -> Nicolas Rougier's paper that was a great inspiration.
//
// ## Summary
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
// ## Definitions
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
// ## Basic algorithm
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
//   o-------o  6      the vertices 3 and 4 are both in the outer corner.
//   - - - 2 - 34
//                node 2
//  node 1
//


{# Includes #}
{$ include 'pygfx.std.wgsl' $}
$$ if colormap_dim
    {$ include 'pygfx.colormap.wgsl' $}
$$ endif



// -------------------- functions --------------------


fn is_finite_vec(v:vec3<f32>) -> bool {
    return is_finite(v.x) && is_finite(v.y) && is_finite(v.z);
}

// Naga has removed isNan checks, because backends may be using fast-math, in
// which case nan is assumed not to happen, and isNan would always be false. If
// we assume that some nan mechanics still work, we can still detect it.
// See https://github.com/pygfx/wgpu-py/blob/main/tests/test_not_finite.py
// NOTE: Other option is loading as i32, checking bitmask, and then bitcasting to float.
//       -> This might be faster, but we need a benchmark to make sure.
fn is_nan(v:f32) -> bool {
    return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0;
}
fn is_inf(v:f32) -> bool {
    return v != 0.0 && v * 2.0 == v;
}
fn is_finite(v:f32) -> bool {
    return !is_nan(v) && !is_inf(v);
}

fn rotate_vec2(v:vec2<f32>, angle:f32) -> vec2<f32> {
    return vec2<f32>(cos(angle) * v.x - sin(angle) * v.y, sin(angle) * v.x + cos(angle) * v.y);
}


// -------------------- vertex shader --------------------


struct VertexInput {
    @builtin(vertex_index) index : u32,
};

@vertex
fn vs_main(in: VertexInput) -> Varyings {

    let screen_factor:vec2<f32> = u_stdinfo.logical_size.xy / 2.0;
    let l2p:f32 = u_stdinfo.physical_size.x / u_stdinfo.logical_size.x;

    // Indexing
    let index = i32(in.index);
    let node_index = index / 6;
    let vertex_index = index % 6;
    let vertex_num = vertex_index + 1;
    var face_index = node_index;  // corrected below if necessary, depending on configuration
    let node_index_is_even = node_index % 2 == 0;

    // Sample the current node and it's two neighbours. Model coords.
    // Note that if we sample out of bounds, this affects the shader in mysterious ways (21-12-2021).
    let pos_m_prev = load_s_positions(max(0, node_index - 1));
    let pos_m_node = load_s_positions(node_index);
    let pos_m_next = load_s_positions(min(u_renderer.last_i, node_index + 1));
    // Convert to world
    let pos_w_prev = u_wobject.world_transform * vec4<f32>(pos_m_prev.xyz, 1.0);
    let pos_w_node = u_wobject.world_transform * vec4<f32>(pos_m_node.xyz, 1.0);
    let pos_w_next = u_wobject.world_transform * vec4<f32>(pos_m_next.xyz, 1.0);
    // Convert to camera view
    let pos_c_prev = u_stdinfo.cam_transform * pos_w_prev;
    let pos_c_node = u_stdinfo.cam_transform * pos_w_node;
    let pos_c_next = u_stdinfo.cam_transform * pos_w_next;
    // convert to NDC
    let pos_n_prev = u_stdinfo.projection_transform * pos_c_prev;
    let pos_n_node = u_stdinfo.projection_transform * pos_c_node;
    let pos_n_next = u_stdinfo.projection_transform * pos_c_next;
    // Convert to logical screen coordinates, because that's where the lines work
    let pos_s_prev = (pos_n_prev.xy / pos_n_prev.w + 1.0) * screen_factor;
    let pos_s_node = (pos_n_node.xy / pos_n_node.w + 1.0) * screen_factor;
    let pos_s_next = (pos_n_next.xy / pos_n_next.w + 1.0) * screen_factor;

    // Get vectors representing the two incident line segments (screen coords)
    var vec_s_prev: vec2<f32> = pos_s_node.xy - pos_s_prev.xy;  // from node 1 (to node 2)
    var vec_s_next: vec2<f32> = pos_s_next.xy - pos_s_node.xy;  // to node 3 (from node 2)

    // Calculate the angle between them. We use this at the end to rotate the coord.
    var angle1 = atan2(vec_s_prev.y, vec_s_prev.x);
    var angle3 = atan2(vec_s_next.y, vec_s_next.x);

    // The thickness of the line in terms of geometry is a wee bit thicker.
    // Just enough so that fragments that are partially on the line, are also included
    // in the fragment shader. That way we can do aa without making the lines thinner.
    // All logic in this function works with the ticker line width. But we pass the real line width as a varying.
    $$ if thickness_space == 'screen'
        let thickness_ratio = 1.0;
    $$ else
        // The thickness is expressed in world space. So we first check where a point, moved shift_factor logic pixels away
        // from the node, ends up in world space. We actually do that for both x and y, in case there's anisotropy.
        // The shift_factor was added to alleviate issues with the point jitter when the user zooms in
        // See https://github.com/pygfx/pygfx/issues/698
        // and https://github.com/pygfx/pygfx/pull/706/files
        let shift_factor = 1000.0;
        let pos_s_node_shiftedx = pos_s_node + vec2<f32>(shift_factor, 0.0);
        let pos_s_node_shiftedy = pos_s_node + vec2<f32>(shift_factor, 1.0);
        let pos_n_node_shiftedx = vec4<f32>((pos_s_node_shiftedx / screen_factor - 1.0) * pos_n_node.w, pos_n_node.z, pos_n_node.w);
        let pos_n_node_shiftedy = vec4<f32>((pos_s_node_shiftedy / screen_factor - 1.0) * pos_n_node.w, pos_n_node.z, pos_n_node.w);
        let pos_w_node_shiftedx = u_stdinfo.cam_transform_inv * u_stdinfo.projection_transform_inv * pos_n_node_shiftedx;
        let pos_w_node_shiftedy = u_stdinfo.cam_transform_inv * u_stdinfo.projection_transform_inv * pos_n_node_shiftedy;
        $$ if thickness_space == 'model'
            // Transform back to model space
            let pos_m_node_shiftedx = u_wobject.world_transform_inv * pos_w_node_shiftedx;
            let pos_m_node_shiftedy = u_wobject.world_transform_inv * pos_w_node_shiftedy;
            // Distance in model space
            let thickness_ratio = (1.0 / shift_factor) * 0.5 * (distance(pos_m_node.xyz, pos_m_node_shiftedx.xyz) + distance(pos_m_node.xyz, pos_m_node_shiftedy.xyz));
        $$ else
            // Distance in world space
            let thickness_ratio = (1.0 / shift_factor) * 0.5 * (distance(pos_w_node.xyz, pos_w_node_shiftedx.xyz) + distance(pos_w_node.xyz, pos_w_node_shiftedy.xyz));
        $$ endif
    $$ endif
    let min_size_for_pixel = 1.415 / l2p;  // For minimum pixel coverage. Use sqrt(2) to take diagonals into account.
    $$ if aa
    let thickness:f32 = u_material.thickness / thickness_ratio;  // Logical pixels
    let half_thickness = 0.5 * max(min_size_for_pixel, thickness + 1.0 / l2p);  // add 0.5 physical pixel on each side.
    $$ else
    let thickness:f32 = max(min_size_for_pixel, u_material.thickness / thickness_ratio);  // non-aa lines get no thinner than 1 px
    let half_thickness = 0.5 * thickness;
    $$ endif

    // Declare vertex cords (x along segment, y perpendicular to it).
    // The coords 1 and 5 have a positive y coord, the coords 2 and 6 negative.
    // These values are relative to the line width.
    var coord1: vec2<f32>;
    var coord2: vec2<f32>;
    var coord3: vec2<f32>;
    var coord4: vec2<f32>;
    var coord5: vec2<f32>;
    var coord6: vec2<f32>;

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

    // The vertex inset, in coord-coords. Is set for joins to keep the segments rectangular.
    // The value will depend on the angle between the segments, and the line thickness.
    var vertex_inset = 0.0;

    // A value to offset certain varyings towards the neighbouring nodes. The value indicated how
    // much a value should be moved towards the "other" value. In units of thickness.
    var offset_ratio = 0.0;

    // Init other "other" values
    var pos_n_other = pos_n_node;
    var pos_s_other = pos_s_node;

    $$ if dashing
        $$ if line_type == 'line'
            let cumdist_node = f32(load_s_cumdist(node_index));
            var cumdist_other = 0.0;
        $$ else
            let cumdist_before = 0.0;
            $$ if thickness_space == 'screen'
            let cumdist_after = length(pos_s_node.xy - select(pos_s_prev.xy, pos_s_next.xy, node_index_is_even));
            $$ elif thickness_space == 'world'
            let cumdist_after = length(pos_w_node.xyz - select(pos_w_prev.xyz, pos_w_next.xyz, node_index_is_even));
            $$ elif thickness_space == 'model'
            let cumdist_after = length(pos_m_node.xyz - select(pos_m_prev.xyz, pos_m_next.xyz, node_index_is_even));
            $$ endif
            let cumdist_node = select(cumdist_before, cumdist_after, !node_index_is_even);
            let cumdist_other = select(cumdist_before, cumdist_after, node_index_is_even);
        $$ endif
    $$ endif

    $$ if color_mode == 'vertex'
        let color_node = load_s_colors(node_index);
        var color_other = color_node;
    $$ elif color_mode == 'vertex_map'
        let texcoord_node = load_s_texcoords(node_index);
        var texcoord_other = texcoord_node;
    $$ endif

    // Determine whether to draw a cap. Either on the left, the right, or both! In the latter case
    // we draw a double-cap, which results in a circle for round caps.
    // A cap is needed when:
    // - This is the first / last point on the line.
    // - The neighbouring node is nan.
    // - The neighbouring node is equal.
    // - If the line segment's direction has a significant component in the camera view direction,
    //   i.e. a depth component, then a cap is created if there is sufficient overlap with the neighbouring cap.
    //
    // Note that in densely sampled lines, rendering caps for a series of nodes
    // makes transparent lines appear opaque. This can happen when the line has
    // sharp corners, even if the corner (on screen) is due to 3D camera
    // orientation. So let's be pretty conservative. However, caps can still be
    // introduced by broken joins, so we cannot completely prevent the
    // aforementioned effect by turning things off/down here.

    // Is this a line that "goes deep"?
    let vec_s_prev_c = vec3<f32>(pos_c_node.xyz - pos_c_prev.xyz);
    let vec_s_next_c = vec3<f32>(pos_c_next.xyz - pos_c_node.xyz);
    let vec_s_prev_has_significant_depth_component = abs(vec_s_prev_c.z) > 10.0 * length(vec_s_prev_c.xy);
    let vec_s_next_has_significant_depth_component = abs(vec_s_next_c.z) > 10.0 * length(vec_s_next_c.xy);
    // Determine capp-ness
    let minor_dist_threshold = 0.0;
    let major_dist_threshold = 0.125 * max(1.0, half_thickness);
    var left_is_cap = !is_finite_vec(pos_m_prev) || length(vec_s_prev) <= select(minor_dist_threshold, major_dist_threshold, vec_s_prev_has_significant_depth_component);
    var right_is_cap = !is_finite_vec(pos_m_next) || length(vec_s_next) <= select(minor_dist_threshold, major_dist_threshold, vec_s_next_has_significant_depth_component);

    $$ if line_type in ['segment', 'arrow']
    left_is_cap = left_is_cap || node_index_is_even;
    right_is_cap = right_is_cap || !node_index_is_even;
    $$ endif

    // The big triage ...

    if (left_is_cap && right_is_cap) {
        // Create two caps
        vec_s_prev = vec2<f32>(0.0, 0.0);
        vec_s_next = vec_s_prev;
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
        vec_s_prev = vec_s_next;
        angle1 = angle3;

        coord1 = vec2<f32>(-1.0, 1.0);
        coord2 = coord1;
        coord3 = coord2;
        coord4 = vec2<f32>(-1.0, -1.0);
        coord5 = vec2<f32>(0.0, 1.0);
        coord6 = vec2<f32>(0.0, -1.0);

        if (vertex_num <= 4) {
            offset_ratio = -1.0;
        }

        pos_s_other = pos_s_next;
        pos_n_other = pos_n_next;
        $$ if dashing and line_type == 'line'
        cumdist_other = f32(load_s_cumdist(node_index + 1));
        $$ endif

    } else if (right_is_cap)  {
        // Create a cap using vertex 4, 5, 6
        vec_s_next = vec_s_prev;
        angle3 = angle1;

        coord1 = vec2<f32>(0.0, 1.0);
        coord2 = vec2<f32>(0.0, -1.0);
        coord3 = vec2<f32>(1.0, 1.0);
        coord4 = vec2<f32>(1.0, -1.0);
        coord5 = coord4;
        coord6 = coord4;

        if (vertex_num >= 3) {
            offset_ratio = -1.0;
        }
        face_index = face_index - 1;  // belongs to previous face

        pos_s_other = pos_s_prev;
        pos_n_other = pos_n_prev;
        $$ if dashing and line_type == 'line'
        cumdist_other = f32(load_s_cumdist(node_index - 1));
        $$ endif

    } else {
        face_index = face_index - i32(vertex_num <= 3);

        pos_s_other = select(pos_s_prev, pos_s_next, vertex_num >= 4);
        pos_n_other = select(pos_n_prev, pos_n_next, vertex_num >= 4);
        $$ if dashing and line_type == 'line'
        cumdist_other = load_s_cumdist(node_index + select(-1, 1, vertex_num >= 4));
        $$ endif
        $$ if color_mode == 'vertex'
        color_other = load_s_colors(node_index + select(-1, 1, vertex_num >= 4));
        $$ elif color_mode == 'vertex_map'
        texcoord_other = load_s_texcoords(node_index + select(-1, 1, vertex_num >= 4));
        $$ endif

        $$ if line_type == 'quickline'

        // Probably faster than lines with joins, but have not tested yet.

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
        if (vertex_num == 3 || vertex_num == 4) {
            offset_ratio = -miter_length;
        }

        $$ elif line_type == 'line'

        // Create a join

        // Determine the angle of the corner. If this angle is smaller than zero,
        // the inside of the join is at vert2/vert6, otherwise it is at vert1/vert5.
        let angle = atan2( vec_s_prev.x * vec_s_next.y - vec_s_prev.y * vec_s_next.x,
                            vec_s_prev.x * vec_s_next.x + vec_s_prev.y * vec_s_next.y );

        // Which way does the join bent?
        let inner_corner_is_at_15 = angle >= 0.0;

        // The direction in which to place the vert3 and vert4.
        let vert1 = normalize(vec2<f32>(-vec_s_prev.y, vec_s_prev.x));
        let vert5 = normalize(vec2<f32>(-vec_s_next.y, vec_s_next.x));
        let join_vec = normalize(vert1 + vert5);

        // Now calculate how far along this vector we can go without
        // introducing overlapping faces, which would result in glitchy artifacts.
        let vec_s_prev_norm = normalize(vec_s_prev);
        let vec_s_next_norm = normalize(vec_s_next);
        let join_vec_on_vec_s_prev = dot(join_vec, vec_s_prev_norm) * vec_s_prev_norm;
        let join_vec_on_vec_s_next = dot(join_vec, vec_s_next_norm) * vec_s_next_norm;
        var max_vec_mag = {{ '1.5' if dashing else '100.0' }};  // 1.5 corresponds to about 90 degrees
        max_vec_mag = min(max_vec_mag, 0.49 * length(vec_s_prev) / length(join_vec_on_vec_s_prev) / half_thickness);
        max_vec_mag = min(max_vec_mag, 0.49 * length(vec_s_next) / length(join_vec_on_vec_s_next) / half_thickness);

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

            if (vertex_num == 3 || vertex_num == 4) {
                offset_ratio = -miter_length;
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

            if ( vertex_num <= 2 || vertex_num >= 5) {
                offset_ratio = vertex_inset;
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

    // Calculate interpolation ratio.
    // Get ratio in screen space, and then correct for perspective.
    // I derived this step by calculating the new w from the ratio, and then substituting terms.
    let ratio_divisor = length(pos_s_node - pos_s_other);
    var ratio_interp = offset_ratio * half_thickness / ratio_divisor;
    ratio_interp = select(ratio_interp, 0.0, ratio_divisor==0.0);  // prevent inf
    ratio_interp = (1.0 - ratio_interp) * ratio_interp * pos_n_node.w / pos_n_other.w + ratio_interp * ratio_interp;

    // Interpolate / extrapolate
    let z = mix(pos_n_node.z, pos_n_other.z, ratio_interp);
    let w = mix(pos_n_node.w, pos_n_other.w, ratio_interp);
    $$ if dashing
        let cumdist_vertex = mix(cumdist_node, cumdist_other, ratio_interp);
    $$ endif
    $$ if color_mode == 'vertex'
        let color_vert = mix(color_node, color_other, ratio_interp);
    $$ elif color_mode == 'vertex_map'
        let texcoord_vert = mix(texcoord_node, texcoord_other, ratio_interp);;
    $$ endif

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

    // Calculate vertex position in NDC.The z and w are inter/extra-polated.
    let the_pos_s = pos_s_node + relative_vert_s;
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
        varyings.cumdist_node = f32(cumdist_node)  {{ '* w' if thickness_space == 'screen' else '' }};
        varyings.cumdist_vertex = f32(cumdist_vertex)  {{ '* w' if thickness_space == 'screen' else '' }};
        varyings.cumdist_per_pixel = f32( abs(cumdist_node - cumdist_other) / length(pos_s_node - pos_s_other) / l2p )  {{ '' if thickness_space == 'screen' else '* (pos_n_node.w / pos_n_other.w) / w' }};
    $$ endif

    // Picking
    // Note: in theory, we can store ints up to 16_777_216 in f32,
    // but in practice, its about 4_000_000 for f32 varyings (in my tests).
    // We use a real u32 to not lose presision, see frag shader for details.
    varyings.pick_idx = u32(node_index);
    varyings.pick_zigzag = f32(node_index_is_even);

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

    // per-vertex or per-face texcoords
    $$ if color_mode == 'face_map' or color_mode == 'vertex_map'
        $$ if color_mode == 'face_map'
            let texcoord_node = load_s_texcoords(face_index);
            let texcoord_vert = texcoord_node;
        $$ else
            // The texcoord_node and texcoord_vert are defined (and interpolated) above.
        $$ endif
        $$ if colormap_dim == '1d'
            varyings.texcoord_node = f32(texcoord_node);
            varyings.texcoord_vert = f32(texcoord_vert);
        $$ elif colormap_dim == '2d'
            varyings.texcoord_node = vec2<f32>(texcoord_node);
            varyings.texcoord_vert = vec2<f32>(texcoord_vert);
        $$ elif colormap_dim == '3d'
            varyings.texcoord_node = vec3<f32>(texcoord_node);
            varyings.texcoord_vert = vec3<f32>(texcoord_vert);
        $$ endif
    $$ endif

    return varyings;
}


// --------------------  fragment shader --------------------


$$ if dashing
// Constant to help compiler create fixed-size arrays and loops.
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
        var cumdist_continuous : f32;
        var cumdist_per_pixel : f32;
        if (is_join) {
            // First calculate the cumdist at the edge where segment and join meet.
            // Note that cumdist_vertex == cumdist_node at the outer-corner-vertex.
            let cumdist_segment = varyings.cumdist_node - (varyings.cumdist_node - varyings.cumdist_vertex) / (1.0 - abs(join_coord_lin));
            // Calculate the continous cumdist, by interpolating using join_coord_fan
            cumdist_continuous = mix(cumdist_segment, varyings.cumdist_node, abs(join_coord_fan));
        } else {
            // In a segment everything is straight.
            cumdist_continuous = varyings.cumdist_vertex;
        }
        $$ if thickness_space == 'screen'
            cumdist_continuous = cumdist_continuous / varyings.w;
            cumdist_per_pixel = varyings.cumdist_per_pixel;
        $$ else
            cumdist_per_pixel = varyings.cumdist_per_pixel * varyings.w;
        $$ endif

        // Define dash pattern, scale with (uniform) thickness.
        // Note how the pattern is templated (triggering recompilation when it changes), wheras the thickness is a uniform.
        var stroke_sizes = array<f32,dash_count>{{dash_pattern[::2]}};
        var gap_sizes = array<f32,dash_count>{{dash_pattern[1::2]}};
        for (var i=0; i<dash_count; i+=1) {
            stroke_sizes[i] = stroke_sizes[i];
            gap_sizes[i] = gap_sizes[i];
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
        let cumdist_corrected = cumdist_continuous / u_material.thickness + u_material.dash_offset % dash_size;
        let dash_progress = (cumdist_corrected + 0.5 * last_gap) % dash_size - 0.5 * last_gap;

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

        // Convert to (physical) pixel units
        let dashdist_to_physical = u_material.thickness / cumdist_per_pixel;
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
        // let dist_to_stroke_dash_p = vec_to_dash_p.x;  // Butt caps
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
        // Arrow shape. Use pick_zigzag, because it has exactly what we need.
        let arrow_head_factor = varyings.pick_zigzag;
        let arrow_tail_factor = varyings.pick_zigzag * 3.0 - 2.0;
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
        if (half_thickness_p > 0.5) {
            alpha = clamp(0.5 - dist_to_stroke_p, 0.0, 1.0);
        } else {
            // Thin lines stay one physical pixel wide, but scale alpha as they get thinner
            let alpha_base = (1.0 - length(segment_coord_p));
            let thickness_scale = max(0.01, half_thickness_p * 2.0);
            alpha = alpha_base * thickness_scale;
            $$ if dashing
                alpha = alpha * clamp(0.5 - dist_to_stroke_dash_p, 0.0, 1.0);
            $$ endif
        }
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
    $$ elif color_mode == 'vertex_map'
        var texcoord = varyings.texcoord_vert;
        if (is_join) {
            let texcoord_segment = varyings.texcoord_node - (varyings.texcoord_node - varyings.texcoord_vert) / (1.0 - abs(join_coord_lin));
            texcoord = mix(texcoord_segment, varyings.texcoord_node, abs(join_coord_fan));
        }
        let color = sample_colormap(texcoord);
    $$ elif color_mode == 'face_map'
        let color = sample_colormap(varyings.texcoord_vert);
    $$ else
        let color = u_material.color;
    $$ endif
    var physical_color = srgb2physical(color.rgb);

    $$ if false
        // Alternative debug options during dev.
        physical_color = vec3<f32>(abs(dist_to_dash_p) / 20.0, 0.0, 0.0);
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
