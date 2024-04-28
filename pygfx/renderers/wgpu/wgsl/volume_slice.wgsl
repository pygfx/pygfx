// Volume slice rendering


{# Includes #}
{$ include 'pygfx.std.wgsl' $}
$$ if colormap_dim
    {$ include 'pygfx.colormap.wgsl' $}
$$ endif
{$ include 'pygfx.volume_common.wgsl' $}


struct VertexInput {
    @builtin(vertex_index) vertex_index : u32,
};


@vertex
fn vs_main(in: VertexInput) -> Varyings {

    // Our geometry is implicitly defined by the volume dimensions.
    var geo = get_vol_geometry();

    // This layout is like this:
    //
    //   Vertices       Planes (right, left, back, front, top, bottom)
    //                            0      1    2      3     4     5
    //
    //    5----0        0: 0231        +----+
    //   /|   /|        1: 7546       /|24 /|
    //  7----2 |        2: 5014      +----+ |0
    //  | 4--|-1        3: 2763     1| +--|-+
    //  |/   |/         4: 0572      |/35 |/
    //  6----3          5: 3641      +----+

    let plane = u_material.plane.xyzw;  // ax + by + cz + d
    let n = plane.xyz;

    // Define edges (using vertex indices), and their matching plane
    // indices (each edge touches two planes). Note that these need to
    // match the above figure, and that needs to match with the actual
    // BoxGeometry implementation!
    var edges = array<vec2<i32>,12>(
        vec2<i32>(0, 2), vec2<i32>(2, 3), vec2<i32>(3, 1), vec2<i32>(1, 0),
        vec2<i32>(4, 6), vec2<i32>(6, 7), vec2<i32>(7, 5), vec2<i32>(5, 4),
        vec2<i32>(5, 0), vec2<i32>(1, 4), vec2<i32>(2, 7), vec2<i32>(6, 3),
    );
    var ed2pl = array<vec2<i32>,12>(
        vec2<i32>(0, 4), vec2<i32>(0, 3), vec2<i32>(0, 5), vec2<i32>(0, 2),
        vec2<i32>(1, 5), vec2<i32>(1, 3), vec2<i32>(1, 4), vec2<i32>(1, 2),
        vec2<i32>(2, 4), vec2<i32>(2, 5), vec2<i32>(3, 4), vec2<i32>(3, 5),
    );

    // Init intersection info
    var intersect_flags = array<i32,12>(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    var intersect_positions = array<vec3<f32>,12>(
        vec3<f32>(0.0, 0.0, 0.0), vec3<f32>(0.0, 0.0, 0.0), vec3<f32>(0.0, 0.0, 0.0),
        vec3<f32>(0.0, 0.0, 0.0), vec3<f32>(0.0, 0.0, 0.0), vec3<f32>(0.0, 0.0, 0.0),
        vec3<f32>(0.0, 0.0, 0.0), vec3<f32>(0.0, 0.0, 0.0), vec3<f32>(0.0, 0.0, 0.0),
        vec3<f32>(0.0, 0.0, 0.0), vec3<f32>(0.0, 0.0, 0.0), vec3<f32>(0.0, 0.0, 0.0),
    );
    var intersect_texcoords = array<vec3<f32>,12>(
        vec3<f32>(0.0, 0.0, 0.0), vec3<f32>(0.0, 0.0, 0.0), vec3<f32>(0.0, 0.0, 0.0),
        vec3<f32>(0.0, 0.0, 0.0), vec3<f32>(0.0, 0.0, 0.0), vec3<f32>(0.0, 0.0, 0.0),
        vec3<f32>(0.0, 0.0, 0.0), vec3<f32>(0.0, 0.0, 0.0), vec3<f32>(0.0, 0.0, 0.0),
        vec3<f32>(0.0, 0.0, 0.0), vec3<f32>(0.0, 0.0, 0.0), vec3<f32>(0.0, 0.0, 0.0),
    );

    // Intersect the 12 edges
    for (var i:i32=0; i<12; i=i+1) {
        let edge = edges[i];
        let p1_raw = geo.positions[ edge[0] ];
        let p2_raw = geo.positions[ edge[1] ];
        let p1_p = u_wobject.world_transform * vec4<f32>(p1_raw, 1.0);
        let p2_p = u_wobject.world_transform * vec4<f32>(p2_raw, 1.0);
        let p1 = p1_p.xyz / p1_p.w;
        let p2 = p2_p.xyz / p2_p.w;
        let tc1 = geo.texcoords[ edge[0] ];
        let tc2 = geo.texcoords[ edge[1] ];
        let u = p2 - p1;
        let t = -(plane.x * p1.x + plane.y * p1.y + plane.z * p1.z + plane.w) / dot(n, u);
        let intersects:bool = t > 0.0 && t < 1.0;
        intersect_flags[i] = select(0, 1, intersects);
        intersect_positions[i] = mix(p1, p2, vec3<f32>(t, t, t));
        intersect_texcoords[i] = mix(tc1, tc2, vec3<f32>(t, t, t));
    }

    // Init six vertices
    var vertices = array<vec3<f32>,6>(
        vec3<f32>(0.0, 0.0, 0.0),
        vec3<f32>(0.0, 0.0, 0.0),
        vec3<f32>(0.0, 0.0, 0.0),
        vec3<f32>(0.0, 0.0, 0.0),
        vec3<f32>(0.0, 0.0, 0.0),
        vec3<f32>(0.0, 0.0, 0.0),
    );
    var texcoords = array<vec3<f32>,6>(
        vec3<f32>(0.0, 0.0, 0.0),
        vec3<f32>(0.0, 0.0, 0.0),
        vec3<f32>(0.0, 0.0, 0.0),
        vec3<f32>(0.0, 0.0, 0.0),
        vec3<f32>(0.0, 0.0, 0.0),
        vec3<f32>(0.0, 0.0, 0.0),
    );

    // Find first intersection point. This can be any valid intersection.
    // In ed2pl[i][0], the 0 could also be a one. It would mean that we'd
    // move around the box in the other direction.
    var plane_index: i32 = 0;
    var i:i32;
    for (i=0; i<12; i=i+1) {
        if (intersect_flags[i] == 1) {
            plane_index = ed2pl[i][0];
            vertices[0] = intersect_positions[i];
            texcoords[0] = intersect_texcoords[i];
            break;
        }
    }

    // From there take (at most) 5 steps
    let i_start: i32 = i;
    var i_last: i32 = i;
    var max_iter: i32 = 6;
    for (var iter:i32=1; iter<max_iter; iter=iter+1) {
        for (var i:i32=0; i<12; i=i+1) {
            if (i != i_last && intersect_flags[i] == 1) {
                if (ed2pl[i][0] == plane_index) {
                    vertices[iter] = intersect_positions[i];
                    texcoords[iter] = intersect_texcoords[i];
                    plane_index = ed2pl[i][1];
                    i_last = i;
                    break;
                } else if (ed2pl[i][1] == plane_index) {
                    vertices[iter] = intersect_positions[i];
                    texcoords[iter] = intersect_texcoords[i];
                    plane_index = ed2pl[i][0];
                    i_last = i;
                    break;
                }
            }
        }
        if (i_last == i_start) {
            max_iter = iter;
            break;
        }
    }

    // Make the rest degenerate triangles
    for (var i:i32=max_iter; i<6; i=i+1) {
        vertices[i] = vertices[0];
    }

    // Now select the current vertex. We mimic a triangle fan with a triangle list.
    // This works whether the number of vertices/intersections is 3, 4, 5, and 6.
    let index = i32(in.vertex_index);
    var indexmap = array<i32,12>(0, 1, 2, 0, 2, 3, 0, 3, 4, 0, 4, 5);
    let world_pos = vertices[ indexmap[index] ];
    let ndc_pos = u_stdinfo.projection_transform * u_stdinfo.cam_transform * vec4<f32>(world_pos, 1.0);

    var varyings : Varyings;
    varyings.position = vec4<f32>(ndc_pos);
    varyings.world_pos = vec3<f32>(world_pos);
    varyings.texcoord = vec3<f32>(texcoords[ indexmap[index] ]);
    return varyings;
}


@fragment
fn fs_main(varyings: Varyings) -> FragmentOutput {
    let sizef = vec3<f32>(textureDimensions(t_img));
    let value = sample_vol(varyings.texcoord.xyz, sizef);
    let color = sampled_value_to_color(value);

    // Move to physical colorspace (linear photon count) so we can do math
    $$ if colorspace == 'srgb'
        let physical_color = srgb2physical(color.rgb);
    $$ else
        let physical_color = color.rgb;
    $$ endif
    let opacity = color.a * u_material.opacity;
    let out_color = vec4<f32>(physical_color, opacity);

    // Wrap up
    apply_clipping_planes(varyings.world_pos);
    var out = get_fragment_output(varyings.position.z, out_color);

    $$ if write_pick
    // The wobject-id must be 20 bits. In total it must not exceed 64 bits.
    out.pick = (
        pick_pack(u32(u_wobject.id), 20) +
        pick_pack(u32(varyings.texcoord.x * 16383.0), 14) +
        pick_pack(u32(varyings.texcoord.y * 16383.0), 14) +
        pick_pack(u32(varyings.texcoord.z * 16383.0), 14)
    );
    $$ endif

    return out;
}
