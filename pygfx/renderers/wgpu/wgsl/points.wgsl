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

{# Includes #}
{$ include 'pygfx.std.wgsl' $}
$$ if colormap_dim
    {$ include 'pygfx.colormap.wgsl' $}
$$ endif

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
    $$ if color_mode == 'debug' or draw_line_on_edge
        let edge_width = u_material.edge_width / size_ratio;  // expressed in logical screen pixels
    $$ else
        let edge_width = 0.0;
    $$ endif
    $$ if aa
        let size:f32 = size_ref / size_ratio;  // Logical pixels
        $$ if edge_mode == 'outer'
        let half_size = edge_width + 0.5 * max(min_size_for_pixel, size + 1.0 / l2p);  // add 0.5 physical pixel on each side.
        $$ elif edge_mode == 'inner'
        let half_size = 0.5 * max(min_size_for_pixel, size + 1.0 / l2p);  // add 0.5 physical pixel on each side.
        $$ else
        // elif edge_mode == 'centered'
        let half_size = 0.5 * edge_width + 0.5 * max(min_size_for_pixel, size + 1.0 / l2p);  // add 0.5 physical pixel on each side.
        $$ endif
    $$ else
        let size:f32 = max(min_size_for_pixel, size_ref / size_ratio);  // non-aa don't get smaller.
        $$ if edge_mode == 'outer'
        let half_size = edge_width + 0.5 * size;
        $$ elif edge_mode == 'inner'
        let half_size = 0.5 * size;
        $$ else
        // elif edge_mode == 'centered'
        let half_size = 0.5 * edge_width + 0.5 * size;
        $$ endif
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
    $$ if draw_line_on_edge
        varyings.edge_width_p = f32(edge_width * l2p);
    $$ endif

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

    let dist_to_face_center_p = length(pointcoord_p);
    let dist_to_face_edge_p = get_signed_distance_to_shape_edge(pointcoord_p, varyings.size_p);

    // Determine face_alpha based on shape and aa
    var face_alpha: f32 = 1.0;
    $$ if is_sprite
        // sprites have their alpha defined by the map and opacity only
    $$ elif color_mode == 'debug'
        face_alpha = 1.0;
    $$ elif shape == 'gaussian'
        let d = length(pointcoord_p);
        let sigma_p = half_size_p / 3.0;
        let t = d / sigma_p;
        face_alpha = exp(-0.5 * t * t);
        if (dist_to_face_edge_p > 0.0) { face_alpha = 0.0; }
    $$ elif aa
        if (half_size_p > 0.5) {
            face_alpha = clamp(0.5 - dist_to_face_edge_p, 0.0, 1.0);
        } else {
            // Tiny points, factor based on dist_to_face_center_p, scaled by the size (with a max)
            face_alpha = max(0.0, 1.0 - dist_to_face_center_p) * max(0.01, half_size_p * 2.0);
        }
        face_alpha = sqrt(face_alpha);  // Visual trick to help aa markers look equal size as non-aa.
    $$ else
        face_alpha = f32(dist_to_face_edge_p <= 0.0);  // boolean alpha
    $$ endif

    // Sample face color
    $$ if color_mode == 'vertex'
        let sampled_face_color = varyings.color;
    $$ elif color_mode == 'map' or color_mode == 'vertex_map'
        let sampled_face_color = sample_colormap(varyings.texcoord);
    $$ elif color_mode == 'debug'
        let d = dist_to_face_edge_p / half_size_p * 2;
        var col : vec3<f32> = vec3<f32>(1.0) - sign(d)*vec3<f32>(0.1,0.4,0.7);
        col *= 1.0 - exp(-2.0 * abs(d));
        col *= 0.8 + 0.2 * cos(120.0 * d);
        col = mix(col, vec3<f32>(1.0), 1.0 - smoothstep(0.0, 0.02, abs(d)));
        let sampled_face_color = vec4<f32>(col, 1.);
    $$ else
        let sampled_face_color = u_material.color;
    $$ endif

    // Define face color+alpha.
    var face_color = vec4<f32>(sampled_face_color.rgb, clamp(sampled_face_color.a, 0.0, 1.0) * face_alpha);

    // For sprites, multiply face_color with sprite color
    $$ if is_sprite == 2
        let sprite_coord = (pointcoord_p + half_size_p) / (2.0 * half_size_p);
        if (min(sprite_coord.x, sprite_coord.y) < 0.0 || max(sprite_coord.x, sprite_coord.y) > 1.0) {
            face_color = vec4<f32>(face_color.rgb, 0.0);  // out of sprite range
        } else {
            let sprite_value = textureSample(t_sprite, s_sprite, sprite_coord);
            $$ if sprite_nchannels == 1
                face_color = vec4<f32>(face_color.rgb * sprite_value.r, face_color.a);
            $$ elif sprite_nchannels == 2
                face_color = vec4<f32>(face_color.rgb * sprite_value.r, face_color.a * sprite_value.g);
            $$ elif sprite_nchannels == 3
                face_color = vec4<f32>(face_color.rgb * sprite_value.rgb, face_color.a);
            $$ else
                face_color = vec4<f32>(face_color.rgb * sprite_value.rgb, face_color.a * sprite_value.a);
            $$ endif
        }
    $$ endif

    // Init the final color
    var the_color = face_color;

    // If we have an edge, determine edge_color, and mix it with the face_color
    $$ if draw_line_on_edge and not color_mode == 'debug'
        // In MPL the edge is centered on what would normally be the edge, i.e.
        // half the edge is over the face, half extends beyond it. Plotly does
        // the same. The face and edge are drawn as if they were separate
        // entities. I.e. a semi-transparent edge blends (its overlapping part)
        // with the feace.

        // Calculate "SDF"
        let half_edge_width_p: f32 = 0.5 * varyings.edge_width_p;
        $$ if edge_mode == 'outer'
        let dist_to_line_center_p: f32 = abs(dist_to_face_edge_p - half_edge_width_p);
        $$ elif edge_mode == 'inner'
        let dist_to_line_center_p: f32 = abs(dist_to_face_edge_p + half_edge_width_p);
        $$ else
        // elif edge_mode == 'centered'
        let dist_to_line_center_p: f32 = abs(dist_to_face_edge_p);
        $$ endif
        let dist_to_line_edge_p: f32 = dist_to_line_center_p - half_edge_width_p;
        // Calculate edge_alpha based on marker shape end edge thickness
        var edge_alpha = 0.0;
        $$ if aa
            if (half_edge_width_p > 0.5) {
                edge_alpha = clamp(0.5 - dist_to_line_edge_p, 0.0, 1.0);
            } else {
                // Thin line, factor based on dist_to_line_center_p, scaled by the size (with a max)
                edge_alpha = max(0.0, 1.0 - dist_to_line_center_p) * max(0.01, half_edge_width_p * 2.0);
            }
            edge_alpha = sqrt(edge_alpha);  // looks better
        $$ else
            edge_alpha = f32(dist_to_line_edge_p <= 0.0);  // boolean alpha
        $$ endif
        // Sample the edge color. Always a uniform, currently.
        let sampled_edge_color = u_material.edge_color;
        // Combine
        let edge_color = vec4<f32>(sampled_edge_color.rgb, sampled_edge_color.a * edge_alpha);
        // Mix edge over face!
        let face_factor = (1.0 - edge_color.a) * f32(face_color.a > 0.0);
        the_color = vec4<f32>(
            face_factor * face_color.rgb + (1.0 - face_factor) * edge_color.rgb,
            1.0 - (1.0 - edge_color.a) * (1.0 - face_color.a),
        );
    $$ endif

    // Determine final color and opacity
    if (the_color.a <= 0.0) { discard; }
    $$ if color_mode == 'debug'
        let out_color = vec4<f32>(srgb2physical(the_color.rgb), the_color.a);
    $$ else
        let out_color = vec4<f32>(srgb2physical(the_color.rgb), the_color.a * u_material.opacity);
    $$ endif

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


fn get_signed_distance_to_shape_edge(coord: vec2<f32>, size: f32) -> f32 {
    // Thank you Nicolas!
    //
    // The paper "Antialiased 2D Grid, Marker, and Arrow Shaders", by Nicolas
    // Rougier (https://jcgt.org/published/0003/04/01/) has been instrumental to
    // implement these marker shapes.
    //
    // Other potentially interesting shapes from the paper that we can
    // implement: chevron, arrow, tag. A star shape would be nice, but I don't
    // know how hard it is to do it with this technique. Other possible
    // variations: a thinner variant of plusses and crosses.

    $$ if shape == 'circle' or shape == 'gaussian'
        // A simple disk
        return length(coord) - size * 0.5;

    $$ elif shape == 'ring'
        // A ring is the difference of two discs
        let r1 = length(coord) - size / 2.0;
        let r2 = length(coord) - size / 4.0;
        return max(r1, -r2);

    $$ elif shape == 'square'
        // A square is the intersection of four half-planes, but we can use the symmetry of the object (abs) to shorten the code.
        // Chosing the full square means that it is/appears larger than the circle and the diamond.
        let square_sdf = max( abs(coord.x), abs(coord.y) );
        return square_sdf - size * 0.5;  // Square occupies full quad (consistent with MPL)
        // return square_sdf - size / (2.0*SQRT_2);  // Square fits inside circle (Rougier's implementation)

    $$ elif shape == 'diamond'
        // A diamond is the rotation of a square
        let x = 0.5 * SQRT_2 * (coord.x + coord.y);
        let y = 0.5 * SQRT_2 * (coord.x - coord.y);
        return max( abs(x), abs(y) ) - size / (2.0 * SQRT_2);

    $$ elif shape == 'plus'
        // A plus is the intersection of eight half-planes that can be reduced to four using symmetries.
        let x = coord.x;
        let y = coord.y;
        let r1 = max(abs(x - size/3.0), abs(x + size/3.0));
        let r2 = max(abs(y - size/3.0), abs(y + size/3.0));
        let r3 = max(abs(x), abs(y));
        return max(min(r1,r2),r3) - size/2.0;

    $$ elif shape == 'cross'
        // A cross is a rotated plus
        let x = 0.5 * SQRT_2 * (coord.x + coord.y);
        let y = 0.5 * SQRT_2 * (coord.x - coord.y);
        let r1 = max(abs(x - size/3.0), abs(x + size/3.0));
        let r2 = max(abs(y - size/3.0), abs(y + size/3.0));
        let r3 = max(abs(x), abs(y));
        return max(min(r1,r2),r3) - size/2.0;

    $$ elif shape == 'asterix'
        // An asterisk is the union of a cross and a plus.
        let x1 = coord.x;
        let y1 = coord.y;
        let x2 = 0.5 * SQRT_2 * (coord.x + coord.y);
        let y2 = 0.5 * SQRT_2 * (coord.x - coord.y);
        let r1 = max(abs(x2)- size/2.0, abs(y2)- size/10.0);
        let r2 = max(abs(y2)- size/2.0, abs(x2)- size/10.0);
        let r3 = max(abs(x1)- size/2.0, abs(y1)- size/10.0);
        let r4 = max(abs(y1)- size/2.0, abs(x1)- size/10.0);
        return min( min(r1,r2), min(r3,r4));

    $$ elif shape.startswith('triangle')
        // A triangle is the intersection of three half-planes
        // y-offset to center the shape by 0.25*size
        $$ if shape.endswith('down')
            let coord_triangle = vec2<f32>(coord.x, coord.y - 0.25*size);
        $$ elif shape.endswith('left')
            let coord_triangle = vec2<f32>(-coord.y, coord.x - 0.25*size);
        $$ elif shape.endswith('right')
            let coord_triangle = vec2<f32>(-coord.y, -coord.x - 0.25*size);
        $$ elif shape.endswith('up')
            let coord_triangle = vec2<f32>(coord.x, -coord.y - 0.25*size);
        $$ endif
        let x = 0.5 * SQRT_2 * (coord_triangle.x - coord_triangle.y);
        let y = 0.5 * SQRT_2 * (coord_triangle.x + coord_triangle.y);
        let r1 = max(abs(x), abs(y)) - size/(2*SQRT_2);
        let r2 = coord_triangle.y;
        return max(r1, r2);

    $$ elif shape == 'heart'
        // A heart is the union of a diamond and two discs.
        let x = 0.5 * SQRT_2 * (coord.x + coord.y);
        let y = 0.5 * SQRT_2 * (coord.x - coord.y);
        let r1 = max(abs(x),abs(y))-size/3.5;
        let r2 = length(coord - SQRT_2/2.0*vec2<f32>( 1.0,1.0)*size/3.5) - size/3.5;
        let r3 = length(coord - SQRT_2/2.0*vec2<f32>(-1.0,1.0)*size/3.5) - size/3.5;
        return min(min(r1,r2),r3);

    $$ elif shape == 'spade'
        // A spade is an inverted heart and a tail is made of two discs and two half-planes.
        // Reversed heart (diamond + 2 circles)
        let s = size * 0.85 / 3.5;
        let x = SQRT_2/2.0 * (coord.x - coord.y) + 0.4*s;
        let y = SQRT_2/2.0 * (coord.x + coord.y) - 0.4*s;
        let r1 = max(abs(x),abs(y)) - s;
        let r2 = length(coord + SQRT_2/2.0*vec2<f32>(-1.0,0.2)*s) - s;
        let r3 = length(coord + SQRT_2/2.0*vec2<f32>( 1.0,0.2)*s) - s;
        let r4 = min(min(r1,r2),r3);
        // Root (2 circles and 2 half-planes)
        let c1 = vec2<f32>(-0.65, 0.125);
        let c2 = vec2<f32>( 0.65, 0.125);
        let r5 = length(coord+c1*size) - size/1.6;
        let r6 = length(coord+c2*size) - size/1.6;
        let r7 = -coord.y - 0.5*size;
        let r8 = 0.1*size + coord.y;
        let r9 = max(-min(r5,r6), max(r7,r8));
        return min(r4,r9);

    $$ elif shape == 'club'
        // A club is a clover and a tail.
        // clover (3 discs)
        let t1 = -PI/2.0;
        let c1 = 0.225*vec2<f32>(cos(t1),-sin(t1));
        let t2 = t1+2*PI/3.0;
        let c2 = 0.225*vec2<f32>(cos(t2),-sin(t2));
        let t3 = t2+2*PI/3.0;
        let c3 = 0.225*vec2<f32>(cos(t3),-sin(t3));
        let r1 = length( coord - c1*size) - size/4.25;
        let r2 = length( coord - c2*size) - size/4.25;
        let r3 = length( coord - c3*size) - size/4.25;
        let r4 = min(min(r1,r2),r3);
        // Root (2 circles and 2 half-planes)
        let c4 = vec2<f32>( 0.65, 0.125);
        let c5 = vec2<f32>(-0.65, 0.125);
        let r5 = length(coord+c4*size) - size/1.6;
        let r6 = length(coord+c5*size) - size/1.6;
        let r7 = -coord.y - 0.5*size;
        let r8 = 0.2*size + coord.y;
        let r9 = max(-min(r5,r6), max(r7,r8));
        return min(r4,r9);

    $$ elif shape == 'pin'
        // Simplified formula for the usecase of a pin taken from
        // https://www.shadertoy.com/view/4lcBWn
        var p = - coord / size;
        p.x = abs(p.x);

        let ra = 0.33;
        let h = 2 * 0.33;
        let b = 0.5;

        let rin = 0.33 / 2;

        p.y = p.y + ra / 2;

        let c = vec2(sqrt(1.0-b*b), b);
        let k = dot(c, vec2(p.y, -p.x));

        // Below the pin all toegether
        if(k > c.x * h) {return size * length(p - vec2(0., h));}

        // the opening circle of the pin
        let q = - (length(p) - rin);
        let m = dot(c, p);
        let n = dot(p, p);

        // the top of the circle
        if(k < 0.0    ) {return size * max(q, sqrt(n)    - ra);}
        // Intesection of the triangle cone and the big circle
                         return size * max(q, m          - ra);
    $$ elif shape == 'custom'
        {{ custom_sdf }}

    $$ else
        unknown marker shape! // deliberate wgsl syntax error

    $$ endif
}
