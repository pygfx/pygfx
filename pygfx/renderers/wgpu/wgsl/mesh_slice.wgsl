// Main shader for mesh slice

{# Includes #}
{$ include 'pygfx.std.wgsl' $}
$$ if colormap_dim
    {$ include 'pygfx.colormap.wgsl' $}
$$ endif


struct VertexInput {
    @builtin(vertex_index) vertex_index : u32,
};


@vertex
fn vs_main(in: VertexInput) -> Varyings {
    // This vertex shader uses VertexId and storage buffers instead of
    // vertex buffers. It creates 6 vertices for each face in the mesh,
    // drawn with triangle-list. For the faces that cross the plane, we
    // draw a (thick) line segment with round caps (we need 6 verts for that).
    // Other faces become degenerate triangles.
    let screen_factor = u_stdinfo.logical_size.xy / 2.0;
    let l2p = u_stdinfo.physical_size.x / u_stdinfo.logical_size.x;
    let thickness = u_material.thickness;  // in logical pixels

    // Get the face index, and sample the vertex indices
    let index = i32(in.vertex_index);
    let segment_index = index % 6;
    let face_index = (index - segment_index) / 6;
    let vii = vec3<i32>(load_s_indices(face_index));

    // Vertex positions of this face, in local object coordinates
    let pos1a = load_s_positions(vii[0]);
    let pos2a = load_s_positions(vii[1]);
    let pos3a = load_s_positions(vii[2]);
    let pos1b = u_wobject.world_transform * vec4<f32>(pos1a, 1.0);
    let pos2b = u_wobject.world_transform * vec4<f32>(pos2a, 1.0);
    let pos3b = u_wobject.world_transform * vec4<f32>(pos3a, 1.0);
    let pos1 = pos1b.xyz / pos1b.w;
    let pos2 = pos2b.xyz / pos2b.w;
    let pos3 = pos3b.xyz / pos3b.w;
    // Get the plane definition
    let plane = u_material.plane.xyzw;  // ax + by + cz + d == 0
    let n = plane.xyz;  // not necessarily a unit vector
    // Intersect the plane with pos 1 and 2
    var p: vec3<f32>;
    p = pos1.xyz;
    let denom1 = dot(n,  pos2.xyz - pos1.xyz);
    let t1 = -(plane.x * p.x + plane.y * p.y + plane.z * p.z + plane.w) / denom1;
    // Intersect the plane with pos 2 and 3
    p = pos2.xyz;
    let denom2 = dot(n, pos3.xyz - pos2.xyz);
    let t2 = -(plane.x * p.x + plane.y * p.y + plane.z * p.z + plane.w) / denom2;
    // Intersect the plane with pos 3 and 1
    p = pos3.xyz;
    let denom3 = dot(n, pos1.xyz - pos3.xyz);
    let t3 = -(plane.x * p.x + plane.y * p.y + plane.z * p.z + plane.w) / denom3;
    // Selectors (the denom check seems not needed, but it feels safer to do, in case e.g. the precision changes)
    let b1 = select(0, 4, (t1 >= 0.0) && (t1 <= 1.0) && (denom1 != 0.0));
    let b2 = select(0, 2, (t2 >= 0.0) && (t2 <= 1.0) && (denom2 != 0.0));
    let b3 = select(0, 1, (t3 >= 0.0) && (t3 <= 1.0) && (denom3 != 0.0));
    var pos_index: i32;
    pos_index = b1 + b2 + b3;

    // The big triage
    var the_pos: vec4<f32>;
    var the_coord: vec2<f32>;
    var segment_length: f32;
    var pick_idx = u32(0u);
    var pick_coords = vec3<f32>(0.0);
    if (pos_index < 3 || pos_index == 4) {
        // Just return the same vertex, resulting in degenerate triangles
        the_pos = u_stdinfo.projection_transform * u_stdinfo.cam_transform * vec4<f32>(pos1, 1.0);
        the_coord = vec2<f32>(0.0, 0.0);
        segment_length = 0.0;
    } else {
        if (pos_index == 7) {
            // For each edge we check whether it looks like it is selected. The value
            // of the other t does not matter because it's always on the edge too.
            // I would expect that comparing with 0.0 and 1.0 would work, but
            // apparently the t-values can also be 0.5.
            if (t1 < 0.5 && t2 >= 0.5) { pos_index = 6; }
            else if (t2 < 0.5 && t3 >= 0.5) { pos_index = 3; }
            else if (t3 < 0.5 && t1 >= 0.5) { pos_index = 5; }
        }
        // Get the positions where the frame intersects the plane
        let pos00: vec3<f32> = pos1;
        let pos12: vec3<f32> = mix(pos1, pos2, vec3<f32>(t1, t1, t1));
        let pos23: vec3<f32> = mix(pos2, pos3, vec3<f32>(t2, t2, t2));
        let pos31: vec3<f32> = mix(pos3, pos1, vec3<f32>(t3, t3, t3));
        // b1+b2+b3     000    001    010    011    100    101    110    111
        var positions_a = array<vec3<f32>, 8>(pos00, pos00, pos00, pos23, pos00, pos12, pos12, pos00);
        var positions_b = array<vec3<f32>, 8>(pos00, pos00, pos00, pos31, pos00, pos31, pos23, pos00);
        // Select the two positions that define the line segment
        let pos_a = positions_a[pos_index];
        let pos_b = positions_b[pos_index];
        // Same for face weights
        let fw00 = vec3<f32>(0.5, 0.5, 0.5);
        let fw12 = mix(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(t1, t1, t1));
        let fw23 = mix(vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(t2, t2, t2));
        let fw31 = mix(vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(t3, t3, t3));
        var fws_a = array<vec3<f32>, 8>(fw00, fw00, fw00, fw23, fw00, fw12, fw12, fw00);
        var fws_b = array<vec3<f32>, 8>(fw00, fw00, fw00, fw31, fw00, fw31, fw23, fw00);
        let fw_a = fws_a[pos_index];
        let fw_b = fws_b[pos_index];
        // Go from local coordinates to NDC
        var npos_a: vec4<f32> = u_stdinfo.projection_transform * u_stdinfo.cam_transform * vec4<f32>(pos_a, 1.0);
        var npos_b: vec4<f32> = u_stdinfo.projection_transform * u_stdinfo.cam_transform * vec4<f32>(pos_b, 1.0);
        // Don't forget to "normalize"!
        npos_a = npos_a / npos_a.w;
        npos_b = npos_b / npos_b.w;
        // And to logical pixel coordinates (don't worry about offset)
        let ppos_a = npos_a.xy * screen_factor;
        let ppos_b = npos_b.xy * screen_factor;
        // Get the segment vector, its length, and how much it scales because of thickness
        let v0 = ppos_b - ppos_a;
        segment_length = length(v0);  // in logical pixels;
        let segment_factor = (segment_length + thickness) / segment_length;
        // Get the (orthogonal) unit vectors that span the segment
        let v1 = normalize(v0);
        let v2 = vec2<f32>(v1.y, -v1.x);
        // Get the vector, in local logical pixels for the segment's square
        let pvec_local = 0.5 * vec2<f32>(segment_length + thickness, thickness);
        // Select one of the four corners of the segment rectangle
        var vecs = array<vec2<f32>, 6>(
            vec2<f32>(-1.0, -1.0),
            vec2<f32>( 1.0,  1.0),
            vec2<f32>(-1.0,  1.0),
            vec2<f32>( 1.0,  1.0),
            vec2<f32>(-1.0, -1.0),
            vec2<f32>( 1.0, -1.0),
        );
        let the_vec = vecs[segment_index];
        // Construct the position, also make sure zw scales correctly
        let pvec = the_vec.x * pvec_local.x * v1 + the_vec.y * pvec_local.y * v2;
        let z_range = (npos_b.z - npos_a.z) * segment_factor;
        let the_pos_p = 0.5 * (ppos_a + ppos_b) + pvec;
        let the_pos_z = 0.5 * (npos_a.z + npos_b.z) + the_vec.x * z_range * 0.5;
        let depth_offset = -0.0001;  // to put the mesh slice atop a mesh
        the_pos = vec4<f32>(the_pos_p / screen_factor, the_pos_z + depth_offset, 1.0);
        // Define the local coordinate in physical pixels
        the_coord = the_vec * pvec_local;
        // Picking info
        pick_idx = u32(face_index);
        let mixval = the_vec.x * 0.5 + 0.5;
        pick_coords = vec3<f32>(mix(fw_a, fw_b, vec3<f32>(mixval, mixval, mixval)));
    }

    // Shader output
    var varyings: Varyings;
    varyings.position = vec4<f32>(the_pos);
    varyings.world_pos = vec3<f32>(ndc_to_world_pos(the_pos));
    varyings.dist2center = vec2<f32>(the_coord * l2p);
    varyings.segment_length = f32(segment_length * l2p);
    varyings.segment_width = f32(thickness * l2p);
    varyings.pick_idx = u32(pick_idx);
    varyings.pick_coords = vec3<f32>(pick_coords);

    // per-vertex or per-face coloring
    $$ if color_mode == 'face' or color_mode == 'vertex'
        $$ if color_mode == 'face'
            let cvalue = load_s_colors(face_index);
        $$ else
            let cvalue = (  load_s_colors(vii[0]) * pick_coords[0] +
                            load_s_colors(vii[1]) * pick_coords[1] +
                            load_s_colors(vii[2]) * pick_coords[2] );
        $$ endif
        $$ if color_buffer_channels == 1
            varyings.color = vec4<f32>(cvalue, cvalue, cvalue, 1.0);
        $$ elif color_buffer_channels == 2
            varyings.color = vec4<f32>(cvalue.r, cvalue.r, cvalue.r, cvalue.g);
        $$ elif color_buffer_channels == 3
            varyings.color = vec4<f32>(cvalue, 1.0);
        $$ elif color_buffer_channels == 4
            varyings.color = vec4<f32>(cvalue);
        $$ endif
    $$ endif

    // Set texture coords
    $$ if color_mode == 'face_map' or color_mode == 'vertex_map'
        $$ if color_mode == 'face_map'
            let cvalue = load_s_texcoords(face_index);
        $$ else
            let cvalue = (  load_s_texcoords(vii[0]) * pick_coords[0] +
                            load_s_texcoords(vii[1]) * pick_coords[1] +
                            load_s_texcoords(vii[2]) * pick_coords[2] );
        $$ endif
        $$ if colormap_dim == '1d'
            varyings.texcoord = f32(cvalue);
        $$ elif colormap_dim == '2d'
            varyings.texcoord = vec2<f32>(cvalue);
        $$ elif colormap_dim == '3d'
            varyings.texcoord = vec3<f32>(cvalue);
        $$ endif
    $$ endif

    return varyings;
}



@fragment
fn fs_main(varyings: Varyings) -> FragmentOutput {
    // Discart fragments that are too far from the centerline. This makes round caps.
    // Note that we operate in physical pixels here.
    let distx = max(0.0, abs(varyings.dist2center.x) - 0.5 * varyings.segment_length);
    let dist = length(vec2<f32>(distx, varyings.dist2center.y));
    if (dist > varyings.segment_width * 0.5) {
        discard;
    }

    $$ if color_mode == 'vertex' or color_mode == 'face'
        let color_value = varyings.color;
        let albeido = color_value.rgb;
    $$ elif color_mode == 'vertex_map' or color_mode == 'face_map'
        let color_value = sample_colormap(varyings.texcoord);
        let albeido = color_value.rgb;  // no more colormap
    $$ else
        let color_value = u_material.color;
        let albeido = color_value.rgb;
    $$ endif

    // No aa. This is something we need to decide on. See line renderer.
    // Making this < 1 would affect the suggested_render_mask.
    let alpha = 1.0;
    // Set color
    let physical_color = srgb2physical(albeido);
    let opacity = min(1.0, color_value.a) * alpha;
    let out_color = vec4<f32>(physical_color, opacity);

    // Wrap up
    apply_clipping_planes(varyings.world_pos);
    var out = get_fragment_output(varyings.position.z, out_color);
    $$ if write_pick
    // The wobject-id must be 20 bits. In total it must not exceed 64 bits.
    out.pick = (
        pick_pack(u32(u_wobject.id), 20) +
        pick_pack(varyings.pick_idx, 26) +
        pick_pack(u32(varyings.pick_coords.x * 63.0), 6) +
        pick_pack(u32(varyings.pick_coords.y * 63.0), 6) +
        pick_pack(u32(varyings.pick_coords.z * 63.0), 6)
    );
    $$ endif

    // We curve the line away to the background, so that line pieces overlap each-other
    // in a better way, especially avoiding the joins from overlapping the next line piece.
    out.depth = varyings.position.z + 0.0001 * dist;
    return out;
}