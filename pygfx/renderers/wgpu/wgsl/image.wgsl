// Image rendering.

{# Includes #}
{$ include 'pygfx.std.wgsl' $}
$$ if colormap_dim
    {$ include 'pygfx.colormap.wgsl' $}
$$ endif
{$ include 'pygfx.image_common.wgsl' $}


struct VertexInput {
    @builtin(vertex_index) vertex_index : u32,
};


@vertex
fn vs_main(in: VertexInput) -> Varyings {

    var geo = get_im_geometry();

    // Select what face we're at
    let index = i32(in.vertex_index);
    let i0 = geo.indices[index];

    // Sample position, and convert to world pos, and then to ndc
    let data_pos = vec4<f32>(geo.positions[i0], 1.0);
    let world_pos = u_wobject.world_transform * data_pos;
    let ndc_pos = u_stdinfo.projection_transform * u_stdinfo.cam_transform * world_pos;

    var varyings: Varyings;
    varyings.position = vec4<f32>(ndc_pos);
    varyings.world_pos = vec3<f32>(world_pos.xyz);
    varyings.texcoord = vec2<f32>(geo.texcoords[i0]);
    return varyings;
}


@fragment
fn fs_main(varyings: Varyings) -> FragmentOutput {
    let sizef = vec2<f32>(textureDimensions(t_img));
    let value = sample_im(varyings.texcoord.xy, sizef);
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
        pick_pack(u32(varyings.texcoord.x * 4194303.0), 22) +
        pick_pack(u32(varyings.texcoord.y * 4194303.0), 22)
    );
    $$ endif

    return out;
}
