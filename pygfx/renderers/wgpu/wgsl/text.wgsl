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

// The per-glyph struct. Should match text's GLYPH_DTYPE
struct GlyphData {
    x: f32,
    y: f32,
    s: f32,  // f16 is plenty precise, but is unknown type, I think our current wgsl is not recent enoug    h to support it
    block_index: u32,
    atlas_index_and_format: u32,
}


@group(1) @binding(0)
var<storage,read> s_glyph_info: array<GlyphInfo>;

@group(1) @binding(1)
var<storage,read> s_glyph_data: array<GlyphData>;


@vertex
fn vs_main(in: VertexInput) -> Varyings {

    let screen_factor = u_stdinfo.logical_size.xy / 2.0;

    let raw_index = i32(in.vertex_index);
    let index = raw_index / 6;
    let sub_index = raw_index % 6;

    // Load glyph data
    let glyph_data = s_glyph_data[index];
    let block_index  = i32(glyph_data.block_index);
    let atlas_index = i32(glyph_data.atlas_index_and_format & 0xFFFFu);
    let format_bitmask = (glyph_data.atlas_index_and_format & 0xFFFF0000u) >> 16u;
    let glyph_pos = vec2<f32>(glyph_data.x, glyph_data.y);
    let font_size = f32(glyph_data.s);

    // Get position of the current block
    let block_pos: vec3<f32> = load_s_positions(block_index);

    // Extract actual glyph index and the encoded font props
    let weight_0_15 = (format_bitmask & 0xF000u) >> 12u;  // 4 bits encode -250 .. 500 in steps of 50
    let weight_offset = f32(weight_0_15) * 50.0 - 250.0;
    let slant_0_15 = (format_bitmask & 0x0F00u) >> 8u;  // 4 bits encode -1.75 .. 2 in steps of 0.25
    let slant_offset = f32(slant_0_15) / 4.0 - 1.75;
    //let reserved1 = (format_bitmask & 0x00F0u) >> 4u;
    //let reserved2 = (format_bitmask & 0x000Fu);

    // Load meta-data of the glyph in the atlas
    let glyph_info = s_glyph_info[atlas_index];
    let bitmap_rect = vec4<i32>(glyph_info.origin, glyph_info.size);

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
    let slant_factor = slant_offset * 0.23;  // empirically selected based on NotoSans-Italic
    let slant = vec2<f32>(0.5 - corner.y, 0.0) * slant_factor;

    let pos_corner_factor = corner * vec2<f32>(1.0, -1.0);
    let glyph_vertex_pos = glyph_pos + (pos_offset1 + pos_offset2 * pos_corner_factor + slant) * font_size;


    $$ if is_screen_space

        // We take the object's pos (model pos is origin), move to NDC, and apply the
        // glyph-positioning in logical screen coords. The text position is affected
        // by the world_transform, but the local scale and rotation do not affect the position.
        // We apply these separately in screen space, so the user can scale and rotate the text that way.

        $$ if is_multi_text
            // In multi text geometry, the positions array is used to position blocks in model space.
            let model_pos = block_pos;
            let vertex_pos = glyph_vertex_pos;
        $$ else
            // In single text geometry, the positions array is used for layout in screen space.
            let model_pos = vec3<f32>(0.0, 0.0, 0.0);
            let vertex_pos = block_pos.xy + glyph_vertex_pos;
        $$ endif

        let world_pos = u_wobject.world_transform * vec4<f32>(model_pos, 1.0);
        let ndc_pos = u_stdinfo.projection_transform * u_stdinfo.cam_transform * world_pos;
        let vertex_pos_rotated_and_scaled = u_wobject.rot_scale_transform * vec2<f32>(vertex_pos);
        let delta_ndc = vertex_pos_rotated_and_scaled.xy / screen_factor;

        // Pixel scale is easy
        let atlas_pixel_scale = font_size / f32(REF_GLYPH_SIZE);

    $$ else
        // model-space
        // We take the glyph positions as model pos, move to world and then NDC.

        let model_pos = block_pos + vec3<f32>(glyph_vertex_pos, 0.0);
        let vertex_pos = model_pos.xy;

        let world_pos = u_wobject.world_transform * vec4<f32>(model_pos, 1.0);
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

    // Construct final ndc pos, or degenerate quad if this is an empty slot in the glyph array
    var final_ndc_pos = vec4<f32>(ndc_pos.xy + delta_ndc * ndc_pos.w, ndc_pos.zw);
    if (font_size == 0.0) {
        final_ndc_pos= vec4<f32>(0.0);
    }

    var varyings: Varyings;
    varyings.position = vec4<f32>(final_ndc_pos);
    varyings.world_pos = vec3<f32>(world_pos.xyz / world_pos.w);
    varyings.atlas_pixel_scale = f32(atlas_pixel_scale);
    varyings.bitmap_rect = vec4<i32>(bitmap_rect);
    varyings.glyph_coord = vec2<f32>(corner);
    varyings.weight_offset = f32(weight_offset);

    // Picking
    varyings.pick_idx = u32(index);

    return varyings;
}


@fragment
fn fs_main(varyings: Varyings) -> FragmentOutput {

    // clipping planes
    {$ include 'pygfx.clipping_planes.wgsl' $}

    let l2p:f32 = u_stdinfo.physical_size.x / u_stdinfo.logical_size.x;

    let bitmap_rect = vec4<f32>(varyings.bitmap_rect);

    //  _______     Imagine this being the leftmost pixel in the patch for this glyph.
    // |       |
    // |       |    A = bitmap_rect.x
    // |_______|    B = bitmap_rect.x + 0.5
    // |   |   |    C = bitmap_rect.x + 1.0
    // A   B   C
    //              Sampling from point B and beyond is fine. But in the part between A and B, the sampled value will
    //              also include thet value of the pixel to the left of here (of another glyph). To prevent that,
    //              we clamp the glyph_texcoord.

    // Get the glyph's local texcoord (0..1 in two dimensions), and clamp it.
    // Basically address_mode is CLAMP, but local to this patch.
    let half_pixel_in_tex_coords = 0.5 / bitmap_rect.zw;
    let glyph_texcoord = varyings.glyph_coord;
    let glyph_texcoord_clamped = clamp(varyings.glyph_coord, half_pixel_in_tex_coords, 1.0 - half_pixel_in_tex_coords);

    // The pixels at the edge of the SDF may not be zero (i.e. the furthest distance).
    // But we can assume that the value at A must be zero, and we can make it so.
    let close_to_edge = abs(glyph_texcoord_clamped - glyph_texcoord) / half_pixel_in_tex_coords; // 0..1
    let atlas_value_multiplier = 1.0 - max(close_to_edge.x, close_to_edge.y);

    // Convert to normalized texcoords.
    // Note that this is the first time that we use bitmap_rect.xy; we want to be careful with that value,
    // otherwise roundoff errors can cause slightly different results depending on the position of the glyph in the atlas,
    // which is anoying, because it can e.g. cause image-based tests to fail (for us and our users).
    let texcoord = (bitmap_rect.xy + bitmap_rect.zw * glyph_texcoord_clamped) / vec2<f32>(textureDimensions(t_atlas));

    // Sample the distance. A value of 0.5 represents the edge of the glyph,
    // with positive values representing the inside.
    let atlas_value = f32(textureSample(t_atlas, s_atlas, texcoord).r) * atlas_value_multiplier;

    // Convert to a more useful measure, where the edge is at 0.0, and the inside is negative.
    // The maximum value at which we can still detect the edge is just below 0.5.
    var distance = (0.5 - atlas_value);

    // Load thickness factors
    let weight_offset = clamp(varyings.weight_offset + u_material.weight_offset, -400.0, 1600.0);
    let weight_thickness = weight_offset * 0.00031;  // empirically derived factor
    let outline_thickness = u_material.outline_thickness;

    // The relative size of the text, more or less it's size on screen in physical pixels.
    // Text looked at from an angle (thus getting very scewed) has a low relative_size too.
    let relative_size = varyings.atlas_pixel_scale * f32(REF_GLYPH_SIZE) * l2p;

    // When the relative_size reaches about 10, the text becomes readable.
    // We render unreadable text as little blobs, which reduces aliasing/flicker a bit.
    let text_is_readable = smoothstep(4.0, 8.0, relative_size);
    let alt_distance = length(varyings.glyph_coord - 0.5) * 2.0 - 0.5;
    distance = mix(alt_distance, distance, text_is_readable);

    // The softness defines the width of the aa fall-off region. How much "distance" is used to make this text appear smooth.
    // It is calculated from the scale of one atlas-pixel in screen space, so that the aa is consistent over different text sizes.
    let max_softness = 0.4; // max sdf-distance used to make it soft, if using too much, there's no more room for the outline.
    let softness = clamp(2.0 / relative_size, 0.0, max_softness);

    // Similarly, a smooth transition from front to outline
    let outline_softness = min(softness, 0.5 * outline_thickness);

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

    // Calculate cutoffs
    let max_inner_cutoff = 0.40 - softness;
    let max_outer_cutoff = 0.49 - outline_softness;
    let inner_cutoff = min(0.0 + cut_off_correction + weight_thickness, max_inner_cutoff);
    let outer_cutoff = min(inner_cutoff + outline_thickness, max_outer_cutoff);

    // Init opacity value to get the shape of the glyph
    var aa_alpha = 1.0;
    var outline = 0.0;

    $$ if aa
        // We use smoothstep to include alpha blending.
        let outside_ness = smoothstep(outer_cutoff - softness, (outer_cutoff + softness), distance);
        aa_alpha = (1.0 - outside_ness);
        outline = smoothstep(inner_cutoff - outline_softness, inner_cutoff + outline_softness, distance);
        // Less readable text is made more transparent
    $$ else
        // Do a hard transition
        aa_alpha = select(0.0, 1.0, distance < outer_cutoff);
        outline = select(1.0, 0.0, distance < inner_cutoff);
    $$ endif

    // Early exit
    if (aa_alpha <= 0.0) { discard; }

    // Turn outline really off if not used. Otherwise some of it may leak through in the aa fragments.
    outline = select(0.0, outline, outline_thickness > 0.0);

    // Get color for this fragment
    let base_srgb = u_material.color;
    let outline_srgb = u_material.outline_color;
    let color = mix(srgb2physical(base_srgb.rgb), srgb2physical(outline_srgb.rgb), outline);
    let color_alpha = mix(base_srgb.a, outline_srgb.a, outline);

    // Compose total opacity and the output color
    let opacity = u_material.opacity * color_alpha * aa_alpha;
    var color_out = vec4<f32>(color, opacity);

    // Debug
    //color_out = vec4<f32>(atlas_value, 0.0, 0.0, 1.0);

    var out: FragmentOutput;
    out.color = color_out;

    // Move text closer to camera, since its often overlaid on something.
    // The text is moved closer than the outline so that the outline of one character
    // does not hide the text of an other.
    // The depth buffer is a 32 bit float.
    // Setting the incremented depth to be less than 1e-6 caused
    // z-fighting to occur leading to artifacts.
    // See: https://github.com/pygfx/pygfx/pull/776#issuecomment-2149374275
    // And: https://github.com/pygfx/pygfx/pull/774#issuecomment-2147458827
    // float32 precision is about 5-6 digits
    // The math below is equivalent to (1 - 1e-5 * (2 - outline))
    out.depth = varyings.position.z * (0.99998 + 1e-5 * outline);

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
