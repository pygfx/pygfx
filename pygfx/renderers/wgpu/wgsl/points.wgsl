// # Points shader
//
// ## References:
//
// * https://github.com/vispy/vispy/blob/master/vispy/visuals/markers.py
//
// ## Summary
//
// The vertex shader uses VertexId and storage buffers instead of vertex
// buffers. It creates 6 vertices for each point, using triangle_list topology.
// That gives 2 faces which form a quad.
//


// -------------------- functions --------------------


// See line.wgsl for details
fn is_finite_vec(v:vec3<f32>) -> bool {
    return is_finite(v.x) && is_finite(v.y) && is_finite(v.z);
}
fn is_nan(v:f32) -> bool {
    return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0;
}
fn is_inf(v:f32) -> bool {
    return v != 0.0 && v * 2.0 == v;
}
fn is_finite(v:f32) -> bool {
    return !is_nan(v) && !is_inf(v);
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

    // Sample the current node/point.
    let pos_m = load_s_positions(node_index);
    // Convert to world
    let pos_w = u_wobject.world_transform * vec4<f32>(pos_m.xyz, 1.0);
    // Convert to camera view
    let pos_c = u_stdinfo.cam_transform * pos_w;
    // convert to NDC
    let pos_n = u_stdinfo.projection_transform * pos_c;
    // Convert to logical screen coordinates
    let pos_s = (pos_n.xy / pos_n.w + 1.0) * screen_factor;

    // Get reference size
    $$ if size_mode == 'vertex'
        let size_ref = load_s_sizes(node_index);
    $$ else
        let size_ref = u_material.size;
    $$ endif

    // The size of the point in terms of geometry is a wee bit larger. Just
    // enough so that fragments that are partially on the (visible) point, are
    // also included in the fragment shader. That way we can do aa without
    // making the points smaller. All logic in this function works with the
    // larger size. But we pass the real size as a varying.
    $$ if size_space == 'screen'
        let size_ratio = 1.0;
    $$ else
        // The size is expressed in world space. So we first check where a point, moved shift_factor logical pixels away
        // from the node, ends up in world space. We actually do that for both x and y, in case there's anisotropy.
        let shift_factor = 1000.0;
        let pos_s_shiftedx = pos_s + vec2<f32>(shift_factor, 0.0);
        let pos_s_shiftedy = pos_s + vec2<f32>(0.0, shift_factor);
        let pos_n_shiftedx = vec4<f32>((pos_s_shiftedx / screen_factor - 1.0) * pos_n.w, pos_n.z, pos_n.w);
        let pos_n_shiftedy = vec4<f32>((pos_s_shiftedy / screen_factor - 1.0) * pos_n.w, pos_n.z, pos_n.w);
        let pos_w_shiftedx = u_stdinfo.cam_transform_inv * u_stdinfo.projection_transform_inv * pos_n_shiftedx;
        let pos_w_shiftedy = u_stdinfo.cam_transform_inv * u_stdinfo.projection_transform_inv * pos_n_shiftedy;
        $$ if size_space == 'model'
            // Transform back to model space
            let pos_m_shiftedx = u_wobject.world_transform_inv * pos_w_shiftedx;
            let pos_m_shiftedy = u_wobject.world_transform_inv * pos_w_shiftedy;
            // Distance in model space
            let size_ratio = (1.0 / shift_factor) * 0.5 * (distance(pos_m.xyz, pos_m_shiftedx.xyz) + distance(pos_m.xyz, pos_m_shiftedy.xyz));
        $$ else
            // Distance in world space
            let size_ratio = (1.0 / shift_factor) * 0.5 * (distance(pos_w.xyz, pos_w_shiftedx.xyz) + distance(pos_w.xyz, pos_w_shiftedy.xyz));
        $$ endif
    $$ endif
    let min_size_for_pixel = 1.415 / l2p;  // For minimum pixel coverage. Use sqrt(2) to take diagonals into account.
    $$ if aa
    let size:f32 = size_ref / size_ratio;  // Logical pixels
    let half_size = 0.5 * max(min_size_for_pixel, size + 1.0 / l2p);  // add 0.5 physical pixel on each side.
    $$ else
    let size:f32 = max(min_size_for_pixel, size_ref / size_ratio);  // non-aa don't get smaller.
    let half_size = 0.5 * size;
    $$ endif

    // Relative coords to create the (frontfacing) quad
    var deltas = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 1.0, -1.0),
        vec2<f32>(-1.0,  1.0),
        vec2<f32>(-1.0,  1.0),
        vec2<f32>( 1.0, -1.0),
        vec2<f32>( 1.0,  1.0),
    );
    var the_delta_s = deltas[vertex_index] * half_size;

    // Make a degenerate quad for non-finite positions
    if (!is_finite_vec(pos_m)) {
        the_delta_s = vec2<f32>(0.0, 0.0);
    }

    // Calculate the current virtual vertex position
    let the_pos_s = pos_s + the_delta_s;
    let the_pos_n = vec4<f32>((the_pos_s / screen_factor - 1.0) * pos_n.w, pos_n.z, pos_n.w);

    // Build varyings output
    var varyings: Varyings;

    // Position
    varyings.position = vec4<f32>(the_pos_n);
    varyings.world_pos = vec3<f32>(ndc_to_world_pos(the_pos_n));

    // Coordinates
    varyings.pointcoord_p = vec2<f32>(the_delta_s * l2p);
    varyings.size_p = f32(size * l2p);

    // Picking
    varyings.pick_idx = u32(node_index);

    // per-vertex or per-face coloring
    $$ if color_mode == 'vertex'
        let color_index = node_index;
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
    let tex_coord_index = node_index;

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


// -------------------- fragment shader --------------------


@fragment
fn fs_main(varyings: Varyings) -> FragmentOutput {

    let l2p:f32 = u_stdinfo.physical_size.x / u_stdinfo.logical_size.x;

    let half_size_p: f32 = 0.5 * varyings.size_p;
    let pointcoord_p: vec2<f32> = varyings.pointcoord_p;
    let pointcoord = pointcoord_p / l2p;

    // Get SDF
    let dist_to_center_p = length(pointcoord_p);
    let dist_to_edge_p = dist_to_center_p - half_size_p;

    // Determine alpha
    var alpha: f32 = 1.0;
    $$ if is_sprite
        // sprites have their alpha defined by the map and opacity only
    $$ elif shape == 'gaussian'
        let d = length(pointcoord_p);
        let sigma_p = half_size_p / 3.0;
        let t = d / sigma_p;
        alpha = exp(-0.5 * t * t);
        if (dist_to_edge_p > 0.0) { discard; }
    $$ elif aa
        if (half_size_p > 0.5) {
            alpha = clamp(0.5 - dist_to_edge_p, 0.0, 1.0);
        } else {
            // Tiny points, factor based on dist_to_center_p, scaled by the size (with a max)
            alpha = (1.0 - dist_to_center_p) * max(0.01, half_size_p * 2.0);
        }
        alpha = sqrt(alpha);  // this prevents aa lines from looking thinner
        if (alpha <= 0.0) { discard; }
    $$ else
        if (dist_to_edge_p > 0.0) { discard; }
    $$ endif

    // Determine color
    $$ if color_mode == 'vertex'
        let color = varyings.color;
    $$ elif color_mode == 'map' or color_mode == 'vertex_map'
        let color = sample_colormap(varyings.texcoord);
    $$ else
        let color = u_material.color;
    $$ endif
    var physical_color = srgb2physical(color.rgb);
    var user_alpha = color.a;

    // Multiply with sprite color?
    $$ if is_sprite == 2
        let sprite_coord = (pointcoord_p + half_size_p) / (2.0 * half_size_p);
        if (min(sprite_coord.x, sprite_coord.y) < 0.0) { discard; }
        if (max(sprite_coord.x, sprite_coord.y) > 1.0) { discard; }
        let sprite_value = textureSample(t_sprite, s_sprite, sprite_coord);
        $$ if sprite_nchannels == 1
            physical_color = physical_color * sprite_value.r;
        $$ elif sprite_nchannels == 2
            physical_color = physical_color * sprite_value.r;
            user_alpha = user_alpha * sprite_value.g;
        $$ elif sprite_nchannels == 3
            physical_color = physical_color * sprite_value.rgb;
        $$ else
            physical_color = physical_color * sprite_value.rgb;
            user_alpha = user_alpha * sprite_value.a;
        $$ endif
    $$ endif

    // Determine final rgba value
    let opacity = min(1.0, user_alpha) * alpha * u_material.opacity;
    let out_color = vec4<f32>(physical_color, opacity);

    // Wrap up
    apply_clipping_planes(varyings.world_pos);
    var out = get_fragment_output(varyings.position.z, out_color);

    $$ if write_pick
    // The wobject-id must be 20 bits. In total it must not exceed 64 bits.
    out.pick = (
        pick_pack(u32(u_wobject.id), 20) +
        pick_pack(varyings.pick_idx, 26) +
        pick_pack(u32(pointcoord.x + 256.0), 9) +
        pick_pack(u32(pointcoord.y + 256.0), 9)
    );
    $$ endif

    return out;
}
