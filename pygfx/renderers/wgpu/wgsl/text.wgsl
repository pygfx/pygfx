// Text rendering

{# Includes #}
{$ include 'pygfx.std.wgsl' $}


const REF_GLYPH_SIZE: i32 = {{ REF_GLYPH_SIZE }};


struct VertexInput {
    @builtin(vertex_index) vertex_index : u32,
};

struct GlyphInfo {
    origin: vec2<i32>, // origin in the atlas
    size: vec2<i32>,  // size of the glyph (in the atlas)
    offset: vec2<f32>,  // positional offset of the glyph
};

@group(1) @binding(0)
var<storage,read> s_glyph_infos: array<GlyphInfo>;


@vertex
fn vs_main(in: VertexInput) -> Varyings {

    let screen_factor = u_stdinfo.logical_size.xy / 2.0;

    let raw_index = i32(in.vertex_index);
    let index = raw_index / 6;
    let sub_index = raw_index % 6;

    // Load glyph info
    let glyph_index_raw = u32(load_s_indices(index));
    let font_size = load_s_sizes(index);
    let glyph_pos = load_s_positions(index);

    // Extract actual glyph index and the encoded font props
    let glyph_index = i32(glyph_index_raw & 0x00FFFFFFu);
    let weight_0_15 = (glyph_index_raw & 0xF0000000u) >> 28u;  // highest 4 bits
    let is_slanted = bool(glyph_index_raw & 0x08000000u);
    //let reserved1 = bool(glyph_index_raw & 0x04000000u);
    //let reserved2 = bool(glyph_index_raw & 0x02000000u);
    //let reserved3 = bool(glyph_index_raw & 0x01000000u);

    // Load meta-data of the glyph in the atlas
    let glyph_info = s_glyph_infos[glyph_index];
    let bitmap_rect = vec4<i32>(glyph_info.origin, glyph_info.size );

    // Prep correction vectors
    // The first offsets the rectangle to put it on the baseline/origin.
    // The second puts it at the end of the atlas-glyph rectangle.
    let pos_offset1 = glyph_info.offset / f32(REF_GLYPH_SIZE);
    let pos_offset2 = vec2<f32>(bitmap_rect.zw) / f32(REF_GLYPH_SIZE);

    var corners = array<vec2<f32>, 6>(
        vec2<f32>(0.0, 0.0), vec2<f32>(0.0, 1.0), vec2<f32>(1.0, 0.0),
        vec2<f32>(0.0, 1.0), vec2<f32>(1.0, 0.0), vec2<f32>(1.0, 1.0),
    );
    let corner = corners[sub_index];

    // Apply slanting, a.k.a. automated obliques, a.k.a. fake italics
    let slant_factor = f32(is_slanted) * 0.23;  // empirically selected based on NotoSans-Italic
    let slant = vec2<f32>(0.5 - corner.y, 0.0) * slant_factor;

    let pos_corner_factor = corner * vec2<f32>(1.0, -1.0);
    let vertex_pos = glyph_pos + (pos_offset1 + pos_offset2 * pos_corner_factor + slant) * font_size;
    let texcoord_in_pixels = vec2<f32>(bitmap_rect.xy) + vec2<f32>(bitmap_rect.zw) * corner;

    $$ if screen_space

        // We take the object's pos (model pos is origin), move to NDC, and apply the
        // glyph-positioning in logical screen coords. The text position is affected
        // by the world_transform, but the local scale and rotation do not affect the position.
        // We apply these separately in screen space, so the user can scale and rotate the text that way.

        let raw_pos = vec3<f32>(0.0, 0.0, 0.0);
        let world_pos = u_wobject.world_transform * vec4<f32>(raw_pos, 1.0);
        let ndc_pos = u_stdinfo.projection_transform * u_stdinfo.cam_transform * world_pos;
        let vertex_pos_rotated_and_scaled = u_wobject.rot_scale_transform * vec4<f32>(vertex_pos, 0.0, 1.0);
        let delta_ndc = vertex_pos_rotated_and_scaled.xy / screen_factor;

        // Pixel scale is easy
        let atlas_pixel_scale = font_size / f32(REF_GLYPH_SIZE);

    $$ else

        // We take the glyph positions as model pos, move to world and then NDC.

        let raw_pos = vec4<f32>(vertex_pos, 0.0, 1.0);
        let world_pos = u_wobject.world_transform * raw_pos;
        let ndc_pos = u_stdinfo.projection_transform * u_stdinfo.cam_transform * world_pos;
        let delta_ndc = vec2<f32>(0.0, 0.0);

        // For the pixel scale, we first project points in x and y direction, and calculate
        // their distance in screen space. The smallest distance is used for scale.
        // In other words. we measure how out-of-plane the text is to determine the amount of aa.

        // Part of the out-of-plane skew might be due to intentional anisotropic scaling (stretched text).
        // Therefore we measure the intentional part here, so we can compensate below.
        // Stretched text still becomes somewhat jaggy or blurry, but not as much as it normally would.
        let sx = length(vec3<f32>(u_wobject.world_transform [0][0], u_wobject.world_transform [0][1], u_wobject.world_transform [0][2]));
        let sy = length(vec3<f32>(u_wobject.world_transform [1][0], u_wobject.world_transform [1][1], u_wobject.world_transform [1][2]));
        let aspect_scale = sqrt(sy / sx);
        let scale_correct = vec2<f32>(aspect_scale, 1.0 / aspect_scale);

        let full_matrix = u_stdinfo.projection_transform * u_stdinfo.cam_transform * u_wobject.world_transform;

        let atlas_pixel_dx = vec2<f32>(font_size / f32(REF_GLYPH_SIZE), 0.0) * scale_correct.x;
        let raw_pos_dx = vec4<f32>(vertex_pos + atlas_pixel_dx, 0.0, 1.0);
        let ndc_pos_dx = full_matrix * raw_pos_dx;

        let atlas_pixel_dy = vec2<f32>(0.0, font_size / f32(REF_GLYPH_SIZE)) * scale_correct.y;
        let raw_pos_dy = vec4<f32>(vertex_pos + atlas_pixel_dy, 0.0, 1.0);
        let ndc_pos_dy = full_matrix * raw_pos_dy;

        let screen_pos = (ndc_pos.xy / ndc_pos.w) * screen_factor;
        let screen_pos_dx = (ndc_pos_dx.xy / ndc_pos_dx.w) * screen_factor;
        let screen_pos_dy = (ndc_pos_dy.xy / ndc_pos_dy.w) * screen_factor;
        let atlas_pixel_scale = min(distance(screen_pos, screen_pos_dx), distance(screen_pos, screen_pos_dy));

    $$ endif

    var varyings: Varyings;
    varyings.position = vec4<f32>(ndc_pos.xy + delta_ndc * ndc_pos.w, ndc_pos.zw);
    varyings.world_pos = vec3<f32>(world_pos.xyz / world_pos.w);
    varyings.atlas_pixel_scale = f32(atlas_pixel_scale);
    varyings.texcoord_in_pixels = vec2<f32>(texcoord_in_pixels);
    varyings.weight_offset = f32(f32(weight_0_15) * 50.0 - 250.0);  // encodes -250..500 in steps of 50

    // Picking
    varyings.pick_idx = u32(index);
    varyings.glyph_coord = vec2<f32>(corner);

    return varyings;
}


@fragment
fn fs_main(varyings: Varyings) -> FragmentOutput {

    // Get the float texcoord
    let atlas_size = textureDimensions(t_atlas);
    let texcoord = varyings.texcoord_in_pixels  / vec2<f32>(atlas_size);

    // Sample the distance. A value of 0.5 represents the edge of the glyph,
    // with positive values representing the inside.
    let atlas_value = textureSample(t_atlas, s_atlas, texcoord).r;

    // Convert to a more useful measure, where the edge is at 0.0, and the inside is negative.
    // The maximum value at which we can still detect the edge is just below 0.5.
    let distance = (0.5 - atlas_value);

    // Load thickness factors
    let weight_offset = clamp(varyings.weight_offset + u_material.weight_offset, -400.0, 1600.0);
    let weight_thickness = weight_offset * 0.00031;  // empirically derived factor
    let outline_thickness = u_material.outline_thickness;

    // The softness is calculated from the scale of one atlas-pixel in screen space.
    let max_softness = 0.75;
    let softness = clamp(0.0, max_softness, 2.0 / (f32(REF_GLYPH_SIZE) * varyings.atlas_pixel_scale));

    // Turns out that how thick a font looks depends on a number of factors:
    // - In pygfx the size of the font for which the sdf was created affects the output a bit.
    // - In a browser, the type of browser seems to affect the output a bit.
    // - In a browser, the OS matters more (e.g. Windows and MacOS handle aa and pixel alignment differently).
    // - The blurry edge for aa affects the perception of the weight.
    // - White text on black looks more bold than black text on white!
    //
    // Below you see how I gave the cut_off an offset that scales with the softness.
    // This might suggest that it compensates mostly for the aa-effect but that might be a
    // coincidence. All I did was try to bring our result close to the output of the
    // same text, rendered in a browser, where the text is white on a dark bg (I
    // checked against Firefox on MacOS, with retina display). Note that modern
    // browsers compensate for the white-on-dark effect. We cannot, because we don't
    // know what's behind the text, but the user can use weight_offset when the text
    // is darker than the bg. More info at issue #358.
    let cut_off_correction = 0.25 * softness;

    // Calculate cut-off's. Apply min so it's always a valid shape.
    let cut_off = min(0.49, 0.0 + cut_off_correction + weight_thickness + outline_thickness);
    let outline_cutoff = max(0.0, cut_off - outline_thickness);

    // Init opacity value to get the shape of the glyph
    var aa_alpha = 1.0;
    var soften_alpha = 1.0;
    var outline = 0.0;

    $$ if aa
        // We use smoothstep to include alpha blending.
        let outside_ness = smoothstep(cut_off - softness, cut_off + softness, distance);
        aa_alpha = (1.0 - outside_ness);
        // High softness values also result in lower alpha to prevent artifacts under high angles.
        soften_alpha = 1.0 - max(softness / max_softness - 0.1, 0.0);
        // Outline
        let outline_softness = min(softness, 0.5 * outline_thickness);
        outline = smoothstep(outline_cutoff - outline_softness, outline_cutoff + outline_softness, distance);
    $$ else
        // Do a hard transition
        aa_alpha = select(0.0, 1.0, distance < cut_off);
        outline = select(1.0, 0.0, distance < outline_cutoff);
    $$ endif

    // Early exit
    if (aa_alpha <= 0.0) { discard; }

    // For aa we reduce alpha quicker, which looks better to the human eye
    aa_alpha = aa_alpha * aa_alpha;

    // Turn outline really off if not used. Otherwise some of it may leak through in the aa fragments.
    outline = select(0.0, outline, outline_thickness > 0.0);

    // Get color for this fragment
    let base_srgb = u_material.color;
    let outline_srgb = u_material.outline_color;
    let color = mix(srgb2physical(base_srgb.rgb), srgb2physical(outline_srgb.rgb), outline);
    let color_alpha = mix(base_srgb.a, outline_srgb.a, outline);

    // Compose total opacity and the output color
    let opacity = u_material.opacity * color_alpha * aa_alpha * soften_alpha;
    var color_out = vec4<f32>(color, opacity);

    // Debug
    //color_out = vec4<f32>(atlas_value, 0.0, 0.0, 1.0);

    // Wrap up
    apply_clipping_planes(varyings.world_pos);
    var out = get_fragment_output(varyings.position.z, color_out);

    // Move text closer to camera, since its often overlaid on something.
    // The text is moved closer than the outline so that the outline of one character
    // does not hide the text of an other.
    // The depth buffer is a 32 bit float.
    // Setting the incremented depth to be less than 1e-6 caused
    // z-fighting to occur leading to artifacts.
    // See: https://github.com/pygfx/pygfx/pull/776#issuecomment-2149374275
    // And: https://github.com/pygfx/pygfx/pull/774#issuecomment-2147458827
    out.depth = varyings.position.z - 1e-6 * (2.0 - outline);

    $$ if write_pick
    // The wobject-id must be 20 bits. In total it must not exceed 64 bits.
    out.pick = (
        pick_pack(u32(u_wobject.id), 20) +
        pick_pack(varyings.pick_idx, 26) +
        pick_pack(u32(varyings.glyph_coord.x * 511.0), 9) +
        pick_pack(u32(varyings.glyph_coord.y * 511.0), 9)
    );
    $$ endif

    return out;
}
