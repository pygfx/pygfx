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
    $$ if size_mode == 'vertex'
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


@fragment
fn fs_main(varyings: Varyings) -> FragmentOutput {
    var final_color : vec4<f32>;

    let d = length(varyings.pointcoord);
    let aa_width = 1.0;

    let size = varyings.size;

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
