

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

fn project_point_to_edge_ndc(point1: vec4f, point2: vec4f, can_move_backwards: bool) -> vec4f {
    // The line is defined by points p1 and p2.
    // We shift p1 over the line with p1' = p1 - factor * v
    // We find the factor such that p1' will be at the edge of the screen.
    // It selects the furthest of the two horizontal edges, and same for the vertical edges.
    // Then it selects either horizontal or vertical edge, depending on whether the line is more horizontal or vertical.

    // Move to 2D
    let p1: vec2f = point1.xy / point1.w;
    let p2: vec2f = point2.xy / point2.w;
    var v: vec2f = p2 - p1;
    // Early exit
    if (length(v) < 1e-9) { return point1; }
    // Get factors to shift to edge
    let factor1 = (p1 + 1.0) / v;  // Solve: p1 - factor * v = -1
    let factor2 = (p1 - 1.0) / v;  // Solve: p1 - factor * v = +1
    // Select the factor
    var factor = 0.0;
    if (abs(v.x) > abs(v.y)) {
        factor = max(factor1.x, factor2.x);
    } else {
        factor = max(factor1.y, factor2.y);
    }
    // Constrain moving backwards
    if (!can_move_backwards) {
        factor = max(factor, 0.0);
    }
    // Return as vec4
    return point1 - factor * (point2 - point1);
}


// -------------------- vertex shader --------------------


struct VertexInput {
    @builtin(vertex_index) index : u32,
    $$ if instanced
    @builtin(instance_index) instance_index : u32,
    $$ endif
};

$$ if instanced
struct InstanceInfo {
    transform: mat4x4<f32>,
    id: u32,
};
@group(1) @binding(0)
var<storage,read> s_instance_infos: array<InstanceInfo>;
$$ endif

@vertex
fn vs_main(in: VertexInput) -> Varyings {

    let screen_factor:vec2<f32> = u_stdinfo.logical_size.xy / 2.0;
    let l2p:f32 = u_stdinfo.physical_size.x / u_stdinfo.logical_size.x;

    // Get world transform
    $$ if instanced
        // NOTE: Only orthogonal instance transforms are supported
        let instance_info = s_instance_infos[in.instance_index];
        let world_transform = u_wobject.world_transform * instance_info.transform;
        let world_transform_inv = transpose(instance_info.transform) * u_wobject.world_transform_inv;
    $$ else
        let world_transform = u_wobject.world_transform;
        let world_transform_inv = u_wobject.world_transform_inv;
    $$ endif

    // Indexing
    let index = i32(in.index);
    var node_index1 = index / 6;
    let vertex_index = index % 6;
    let vertex_num = vertex_index + 1;
    var face_index = node_index1;  // corrected below if necessary, depending on configuration
    let node_index_is_even = node_index1 % 2 == 0;

    var node_index2 = min(u_renderer.last_i, node_index1 + 1);
    var node_index3 = min(u_renderer.last_i, node_index1 + 2);

    $$ if loop
    var is_first_node_in_loop = false;
    var is_connecting_node_in_loop = false;

    let loop_state: u32 = load_s_loop(node_index2);
    if (loop_state > 0x0fffffffu) {
        let loop_node_kind = loop_state >> 28;
        let loop_node_count = i32(loop_state & 0x0fffffff);
        if (loop_node_kind == 1u) { // first node
            is_first_node_in_loop = true;
            node_index1 = node_index2 + (loop_node_count - 1);
        } else if (loop_node_kind == 2u) { // last node
            node_index3 = node_index2 - (loop_node_count - 1);
        } else { // if (loop_node_kind == 3u) { // connecting node
            node_index2 = node_index2 - loop_node_count;
            node_index3 = node_index2 + 1;
            is_connecting_node_in_loop = true;
        }
    } else {
        node_index2 = min(u_renderer.last_i, node_index2);
    }
    $$ endif

    // Sample the current node and it's two neighbours. Model coords.
    // Note that if we sample out of bounds, this affects the shader in mysterious ways (21-12-2021).
    var pos_m_1 = load_s_positions(node_index1);
    var pos_m_2 = load_s_positions(node_index2);
    var pos_m_3 = load_s_positions(node_index3);
    // Convert to world
    var pos_w_1 = world_transform * vec4<f32>(pos_m_1.xyz, 1.0);
    var pos_w_2 = world_transform * vec4<f32>(pos_m_2.xyz, 1.0);
    var pos_w_3 = world_transform * vec4<f32>(pos_m_3.xyz, 1.0);
    // Convert to camera view
    var pos_c_1 = u_stdinfo.cam_transform * pos_w_1;
    var pos_c_2 = u_stdinfo.cam_transform * pos_w_2;
    var pos_c_3 = u_stdinfo.cam_transform * pos_w_3;
    // convert to NDC
    var pos_n_1 = u_stdinfo.projection_transform * pos_c_1;
    var pos_n_2 = u_stdinfo.projection_transform * pos_c_2;
    var pos_n_3 = u_stdinfo.projection_transform * pos_c_3;
    // Convert to logical screen coordinates, because that's where the lines work
    var pos_s_1 = (pos_n_1.xy / pos_n_1.w + 1.0) * screen_factor;
    var pos_s_2 = (pos_n_2.xy / pos_n_2.w + 1.0) * screen_factor;
    var pos_s_3 = (pos_n_3.xy / pos_n_3.w + 1.0) * screen_factor;

    $$ if line_type == 'infsegment'
    let can_move_backwards = {{ 'true' if (start_is_infinite and end_is_infinite) else 'false' }};
    let prev_node_ori = pos_n_2;
    let pos_n_next_ori = pos_n_3;
    let pos_n_prev_ori = pos_n_1;
    $$ if start_is_infinite
    if (node_index_is_even) {
        pos_n_2 = project_point_to_edge_ndc(prev_node_ori, pos_n_next_ori, can_move_backwards);
        pos_s_2 = (pos_n_2.xy / pos_n_2.w + 1.0) * screen_factor;
        pos_c_2 = u_stdinfo.projection_transform_inv * pos_n_2;
        pos_w_2 = u_stdinfo.cam_transform_inv * pos_c_2;
        pos_m_2 = (world_transform_inv * pos_w_2).xyz;
    } else {
        pos_n_1 = project_point_to_edge_ndc(pos_n_prev_ori, prev_node_ori, can_move_backwards);
        pos_s_1 = (pos_n_1.xy / pos_n_1.w + 1.0) * screen_factor;
        pos_c_1 = u_stdinfo.projection_transform_inv * pos_n_1;
        pos_w_1 = u_stdinfo.cam_transform_inv * pos_c_1;
        pos_m_1 = (world_transform_inv * pos_w_1).xyz;
    }
    $$ endif
    $$ if end_is_infinite
    if (node_index_is_even) {
        pos_n_3 = project_point_to_edge_ndc(pos_n_next_ori, prev_node_ori, can_move_backwards);
        pos_s_3 = (pos_n_3.xy / pos_n_3.w + 1.0) * screen_factor;
        pos_c_3 = u_stdinfo.projection_transform_inv * pos_n_3;
        pos_w_3 = u_stdinfo.cam_transform_inv * pos_c_3;
        pos_m_3 = (world_transform_inv * pos_w_3).xyz;
    } else {
        pos_n_2 = project_point_to_edge_ndc(prev_node_ori, pos_n_prev_ori, can_move_backwards);
        pos_s_2 = (pos_n_2.xy / pos_n_2.w + 1.0) * screen_factor;
        pos_c_2 = u_stdinfo.projection_transform_inv * pos_n_2;
        pos_w_2 = u_stdinfo.cam_transform_inv * pos_c_2;
        pos_m_2 = (world_transform_inv * pos_w_2).xyz;
    }
    $$ endif
    $$ endif

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
        let pos_s_node_shiftedx = pos_s_2 + vec2<f32>(shift_factor, 0.0);
        let pos_s_node_shiftedy = pos_s_2 + vec2<f32>(shift_factor, 1.0);
        let pos_n_node_shiftedx = vec4<f32>((pos_s_node_shiftedx / screen_factor - 1.0) * pos_n_2.w, pos_n_2.z, pos_n_2.w);
        let pos_n_node_shiftedy = vec4<f32>((pos_s_node_shiftedy / screen_factor - 1.0) * pos_n_2.w, pos_n_2.z, pos_n_2.w);
        let pos_w_node_shiftedx = u_stdinfo.cam_transform_inv * u_stdinfo.projection_transform_inv * pos_n_node_shiftedx;
        let pos_w_node_shiftedy = u_stdinfo.cam_transform_inv * u_stdinfo.projection_transform_inv * pos_n_node_shiftedy;
        $$ if thickness_space == 'model'
            // Transform back to model space
            let pos_m_node_shiftedx = world_transform_inv * pos_w_node_shiftedx;
            let pos_m_node_shiftedy = world_transform_inv * pos_w_node_shiftedy;
            // Distance in model space
            let thickness_ratio = (1.0 / shift_factor) * 0.5 * (distance(pos_m_2.xyz, pos_m_node_shiftedx.xyz) + distance(pos_m_2.xyz, pos_m_node_shiftedy.xyz));
        $$ else
            // Distance in world space
            let thickness_ratio = (1.0 / shift_factor) * 0.5 * (distance(pos_w_2.xyz, pos_w_node_shiftedx.xyz) + distance(pos_w_2.xyz, pos_w_node_shiftedy.xyz));
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

    // Get vectors representing the two incident line segments (screen coords)
    var vec_s_prev: vec2<f32> = pos_s_2.xy - pos_s_1.xy;  // from node 1 (to node 2)
    var vec_s_next: vec2<f32> = pos_s_3.xy - pos_s_2.xy;  // to node 3 (from node 2)

    // Calculate interpolation ratio.
    // Get ratio in screen space, and then correct for perspective.
    // I derived this step by calculating the new w from the ratio, and then substituting terms.
    let ratio_divisor = length(pos_n_1 - pos_n_2);
    let offset_ratio = -1.0;
    var ratio_interp = offset_ratio * half_thickness / ratio_divisor;
    ratio_interp = select(ratio_interp, 0.0, ratio_divisor==0.0);  // prevent inf
    ratio_interp = (1.0 - ratio_interp) * ratio_interp * pos_n_1.w / pos_n_2.w + ratio_interp * ratio_interp;


    let z = mix(pos_n_1.z, pos_n_2.z, ratio_interp);
    let w = mix(pos_n_1.w, pos_n_2.w, ratio_interp);

    let hlw_indir_line = normalize(vec_s_prev) * half_thickness;
    let hlw_ortog_line = vec2f(- hlw_indir_line.y, hlw_indir_line.x);


    let quad1 = pos_s_1.xy - hlw_indir_line  - hlw_ortog_line;
    let quad2 = pos_s_1.xy - hlw_indir_line  + hlw_ortog_line;
    let quad3 = pos_s_2.xy + hlw_indir_line  - hlw_ortog_line;
    let quad4 = pos_s_2.xy + hlw_indir_line  + hlw_ortog_line;

    var the_coord: vec2f;

    if (vertex_index < 3) {
        if vertex_index == 0 { the_coord = quad2; }
        if vertex_index == 1 { the_coord = quad3; }
        if vertex_index == 2 { the_coord = quad1; }
    } else {
        if vertex_index == 3 { the_coord = quad2; }
        if vertex_index == 4 { the_coord = quad4; }
        if vertex_index == 5 { the_coord = quad3; }
    }

    // Calculate vertex position in NDC.The z and w are inter/extra-polated.
    let the_pos_s = the_coord;
    var the_pos_n = vec4<f32>((the_pos_s / screen_factor - 1.0) * w, z, w);



    // Build varyings output
    var varyings: Varyings;

    // Position
    varyings.position = vec4<f32>(the_pos_n);
    varyings.world_pos = vec3<f32>(ndc_to_world_pos(the_pos_n));
    varyings.sdf_pos = vec2<f32>(the_pos_s) * l2p;  /// simply physical screen pos
    varyings.segment_cur = vec4<f32>(pos_s_1, pos_s_2) * l2p;
    varyings.segment_next = vec4<f32>(pos_s_2, pos_s_3) * l2p;

    //  Thickness and segment coord. These are corrected for perspective, otherwise the dashes are malformed in 3D.
    varyings.w = f32(w);
    varyings.thickness_pw = f32(thickness * l2p * w);  // the real thickness, in physical coords
    //varyings.segment_coord_pw = vec2<f32>(the_coord * half_thickness * l2p * w);  // uses a slightly wider thickness
    // Coords related to joins
    //varyings.join_coord = f32(join_coord);
    //varyings.is_outer_corner = f32(is_outer_corner);
    //varyings.valid_if_nonzero = f32(valid_array[vertex_index]);

    $$ if debug
        // Include barycentric coords so we can draw the triangles that make up the line
        varyings.bary = vec3<f32>(f32(vertex_index % 3 == 0), f32(vertex_index % 3 == 1), f32(vertex_index % 3 == 2));
    $$ endif
    $$ if dashing
        notyetsupported
        // Set two varyings, so that we can correctly interpolate the cumdist in the joins.
        // If the thickness is in screen space, we need to correct for perspective division
        varyings.cumdist_node = f32(cumdist_node)  {{ '* w' if thickness_space == 'screen' else '' }};
        varyings.cumdist_vertex = f32(cumdist_vertex)  {{ '* w' if thickness_space == 'screen' else '' }};
        varyings.cumdist_per_pixel = f32( abs(cumdist_node - cumdist_other) / length(pos_s_2 - pos_s_other) / l2p )  {{ '' if thickness_space == 'screen' else '* (pos_n_2.w / pos_n_other.w) / w' }};
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


fn sdSegment(p: vec2f, a: vec2f, b: vec2f ) -> f32
{
    let pa = p - a;
    let ba = b - a;
    let h: f32 = clamp( dot(pa,ba)/dot(ba,ba), 0.0, 1.0 );
    return length( pa - ba * h );
}

// --------------------  fragment shader --------------------


$$ if dashing
// Constant to help compiler create fixed-size arrays and loops.
const dash_count = {{dash_count}};
$$ endif


@fragment
fn fs_main(varyings: Varyings, @builtin(front_facing) is_front: bool) -> FragmentOutput {

    // clipping planes
    {$ include 'pygfx.clipping_planes.wgsl' $}

    // Get the half-thickness in physical coordinates. This is the reference thickness.
    // If aa is used, the line is actually a bit thicker, leaving space to do aa.
    let half_thickness_p = 0.5 * varyings.thickness_pw / varyings.w;


    let dist1 = sdSegment(varyings.sdf_pos, varyings.segment_cur.xy, varyings.segment_cur.zw);

    // TODO: above dist is in logical
    if (dist1 > half_thickness_p) {
        discard;
    }

    let dist2 = sdSegment(varyings.sdf_pos, varyings.segment_next.xy, varyings.segment_next.zw);

    if (dist2 < half_thickness_p) {
        discard;
    }


    let color = u_material.color;

    var physical_color = srgb2physical(color.rgb);

    let alpha = 1.0;
    // Determine final rgba value
    let opacity = min(1.0, color.a) * alpha * u_material.opacity;
    let out_color = vec4<f32>(physical_color, opacity);

    var out: FragmentOutput;
    out.color = out_color;

    return out;
}
