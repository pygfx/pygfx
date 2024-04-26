// Main shader for mesh normal lines

{# Includes #}
{$ include 'pygfx.std.wgsl' $}


struct VertexInput {
    @builtin(vertex_index) vertex_index : u32,
};


@vertex
fn vs_main(in: VertexInput) -> Varyings {
    let index = i32(in.vertex_index);
    let r = index % 2;
    let i0 = index / 2;

    // Get regular position
    let raw_pos = load_s_positions(i0);
    var world_pos = u_wobject.world_transform * vec4<f32>(raw_pos, 1.0);

    // Get the normal, expressed in world coords. Use the normal-matrix
    // to take anisotropic scaling into account.
    let normal_matrix = transpose(u_wobject.world_transform_inv);
    let raw_normal = load_s_normals(i0);
    let world_normal = normalize((normal_matrix * vec4<f32>(raw_normal, 0.0)).xyz);

    // Calculate the two end-pieces of the line that we want to show.
    let pos1 = world_pos.xyz / world_pos.w;
    let pos2 = pos1 + world_normal * u_material.line_length;

    // Select either end of the line and make this the world pos
    let pos3 = pos1 * f32(r) + pos2 * (1.0 - f32(r));
    world_pos = vec4<f32>(pos3 * world_pos.w, world_pos.w,);

    // To NDC
    let ndc_pos = u_stdinfo.projection_transform * u_stdinfo.cam_transform * world_pos;

    var varyings: Varyings;
    varyings.world_pos = vec3<f32>(world_pos.xyz / world_pos.w);
    varyings.position = vec4<f32>(ndc_pos);

    // Stub varyings, because the mesh varyings are based on face index
    varyings.normal = vec3<f32>(world_normal);
    varyings.pick_id = u32(u_wobject.id);
    varyings.pick_idx = u32(0);
    varyings.pick_coords = vec3<f32>(0.0);

    return varyings;
}


@fragment
fn fs_main(varyings: Varyings, @builtin(front_facing) is_front: bool) -> FragmentOutput {

    let color_value = u_material.color;
    let albeido = color_value.rgb;

    // Move to physical colorspace (linear photon count) so we can do math
    $$ if colorspace == 'srgb'
        let physical_albeido = srgb2physical(albeido);
    $$ else
        let physical_albeido = albeido;
    $$ endif
    let opacity = color_value.a * u_material.opacity;

    var physical_color = physical_albeido;
    let out_color = vec4<f32>(physical_color, opacity);

    // Wrap up

    apply_clipping_planes(varyings.world_pos);
    var out = get_fragment_output(varyings.position.z, out_color);

    $$ if write_pick
    // The wobject-id must be 20 bits. In total it must not exceed 64 bits.
    out.pick = (
        pick_pack(varyings.pick_id, 20) +
        pick_pack(varyings.pick_idx, 26) +
        pick_pack(u32(varyings.pick_coords.x * 63.0), 6) +
        pick_pack(u32(varyings.pick_coords.y * 63.0), 6) +
        pick_pack(u32(varyings.pick_coords.z * 63.0), 6)
    );
    $$ endif

    return out;
}