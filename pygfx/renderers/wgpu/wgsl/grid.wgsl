// Main shader for rendering grids.

{# Includes #}
{$ include 'pygfx.std.wgsl' $}


struct VertexInput {
    @builtin(vertex_index) index : u32,
};


@vertex
fn vs_main(in: VertexInput) -> Varyings {
    var varyings: Varyings;

    // Grid coordinates to form a quad
    var grid_coords = array<vec2<f32>, 4>(
        vec2<f32>(0.0, 0.0),
        vec2<f32>(1.0, 0.0),
        vec2<f32>(0.0, 1.0),
        vec2<f32>(1.0, 1.0),
    );

    // Select the grid coord for this vertex. We express it with coord1 and
    // coord2, to avoid confusion with xyz world coordinates. By default the
    // plane is in the xz plane (normal to the y-axis).
    let vertex_grid_coord: vec2<f32> = grid_coords[i32(in.index)];
    let coord1 = vertex_grid_coord.x;
    let coord2 = vertex_grid_coord.y;

    // Calculate reference points
    let p0 = (u_wobject.world_transform * vec4<f32>(0.0, 0.0, 0.0, 1.0)).xyz;  // "origin" of the plane
    let p1 = (u_wobject.world_transform * vec4<f32>(1.0, 0.0, 0.0, 1.0)).xyz;  // on the plane
    let p2 = (u_wobject.world_transform * vec4<f32>(0.0, 0.0, 1.0, 1.0)).xyz;  // on the plane
    let p3 = (u_wobject.world_transform * vec4<f32>(0.0, 1.0, 0.0, 1.0)).xyz;  // out of plane!

    // Get vectors for the plane's axii, expressed in world coordinates
    let v1 = normalize(p1 - p0);
    let v2 = normalize(p2 - p0);
    let v3 = normalize(p3 - p0);

    $$ if inf_grid

        // Get description of the grid plane: ax * by  + cz + d == 0
        let abc = v3;  // == the plane's normal
        let d = - dot(abc, p0);

        // Get point on the plane closest to the origin (handy for debugging)
        //let pos_origin = (d / length(abc)) * abc;

        // Get distance between camera and the plane
        let cam_pos = vec3<f32>(u_stdinfo.cam_transform_inv[3].xyz);
        let cam_k = (dot(abc, cam_pos) + d) / length(abc);
        let cam_pos_on_grid = cam_pos - cam_k * abc;
        let distance_cam_to_grid = abs(cam_k); // == distance(cam_pos, cam_pos_on_grid);
        // let distance_cam_to_grid = abs(p0.y - cam_pos.y);  // debug: only works for xz plane.

        // The closer the camera is to the plane, the less space you need to
        // make the horizon look really far away. Scaling too hard will
        // introduce fluctuation artifacts due to inaccurate float arithmatic.
        // With ortho camera the horizon cannot really be made to look far away.
        // So inf grid is not well suited with ortho camera?
        let pos_multiplier = (50.0 * distance_cam_to_grid);

        // Construct position using only the grid's rotation. Scale and offset are overridden.
        let pos = cam_pos_on_grid + (coord1 - 0.5) * v1 * pos_multiplier + (coord2 - 0.5) * v2 * pos_multiplier;
    $$ else

        // The grid's transform defines its place in the world.
        // Add a 2% margin to have space for line width and aa.
        let pos = (u_wobject.world_transform * vec4<f32>((coord1 * 1.04 - 0.02), 0.0, (coord2 * 1.2 - 0.1), 1.0)).xyz;

        // Calculate range, so that in the frag shader the edges can be handled correctly.
        let p_min = (u_wobject.world_transform * vec4<f32>(0.0, 0.0, 0.0, 1.0)).xyz;
        let p_max = (u_wobject.world_transform * vec4<f32>(1.0, 0.0, 1.0, 1.0)).xyz;
        let range_a = vec2<f32>( dot(v1, p_min), dot(v2, p_min) );
        let range_b = vec2<f32>( dot(v1, p_max), dot(v2, p_max) );

    $$ endif

    var pos_ndc = u_stdinfo.projection_transform * u_stdinfo.cam_transform * vec4<f32>(pos, 1.0);

    // Store positions and the view direction in the world
    varyings.position = vec4<f32>(pos_ndc);
    varyings.world_pos = vec3<f32>(ndc_to_world_pos(out.position));
    varyings.gridcoord = vec2<f32>(dot(v1, pos), dot(v2, pos));
    $$ if not inf_grid
    varyings.range_min = vec2<f32>(min(range_a, range_b));
    varyings.range_max = vec2<f32>(max(range_a, range_b));
    $$ endif

    return varyings;
}


@fragment
fn fs_main(varyings: Varyings) -> FragmentOutput {

    // Collect props

    let major_step= vec2<f32>(u_material.major_step);
    let minor_step = vec2<f32>(u_material.minor_step);
    let axis_step = major_step * 10.0;

    var axis_thickness = vec2<f32>(u_material.axis_thickness);
    var major_thickness = vec2<f32>(u_material.major_thickness);
    var minor_thickness = vec2<f32>(u_material.minor_thickness);

    let axis_color = u_material.axis_color;
    let major_color = u_material.major_color;
    let minor_color = u_material.minor_color;

    // Get the grid coordinate
    let uv = vec2<f32>(varyings.gridcoord.xy);

    // Clamped versions
    let uv_clamped_axis = clamp(uv, -0.5 * axis_step, 0.5 * axis_step);
    var uv_clamped_major = vec2<f32>(uv);
    var uv_clamped_minor = vec2<f32>(uv);

    $$ if not inf_grid
        // Handle edges. If we do nothing about edges, then lines that are on
        // the egde will be half as wide, and have no anti-aliasing. To handle
        // this, we make the grid slightly larger in the vertex shader. Here in
        // the frag shader, we first check whether there is a line (almost) on
        // the edge. If so, we include that line by setting the margin to half
        // the step width. Otherwise, the margin is such that the last line is
        // included, but not any lines beyond the edge.

        let range_min = varyings.range_min;
        let range_max = varyings.range_max;

        // How far to the next line beyond the edge, expressed as a fraction of the step.
        // Similar to a modulo, but different round-direction for min and max.
        let major_fract_min = ceil(range_min / major_step) - (range_min / major_step);
        let major_fract_max = (range_max / major_step) - floor(range_max / major_step);
        // On the edge (or close enough)?
        let major_is_on_edge_min = major_fract_min > vec2<f32>(0.999) || major_fract_min < vec2<f32>(0.001);
        let major_is_on_edge_max = major_fract_max > vec2<f32>(0.999) || major_fract_max < vec2<f32>(0.001);
        // Calcaulate margin.
        let major_margin_min = select((0.5 - major_fract_min) * major_step, 0.5 * major_step, major_is_on_edge_min);
        let major_margin_max = select((0.5 - major_fract_max) * major_step, 0.5 * major_step, major_is_on_edge_max);

        // Repeat for minor.
        let minor_fract_min = ceil(range_min / minor_step) - (range_min / minor_step);
        let minor_fract_max = (range_max / minor_step) - floor(range_max / minor_step);
        let minor_is_on_edge_min = minor_fract_min > vec2<f32>(0.999) || minor_fract_min < vec2<f32>(0.001);
        let minor_is_on_edge_max = minor_fract_max > vec2<f32>(0.999) || minor_fract_max < vec2<f32>(0.001);
        let minor_margin_min = select((0.5 - minor_fract_min) * minor_step, 0.5 * minor_step, minor_is_on_edge_min);
        let minor_margin_max = select((0.5 - minor_fract_max) * minor_step, 0.5 * minor_step, major_is_on_edge_max);

        // Get ranges.
        let range_major_min = range_min - major_margin_min;
        let range_major_max = range_max + major_margin_max;
        let range_minor_min = range_min - minor_margin_min;
        let range_minor_max = range_max + minor_margin_max;

        // Hide/show line parallel to the edge.
        uv_clamped_major = clamp(uv, range_major_min, range_major_max);
        uv_clamped_minor = clamp(uv, range_minor_min, range_minor_max);

        // The lines orthogonal to the edge, should simply not be draw beyond the edge.
        if ( uv.x < range_min.x || uv.x > range_max.x ) {
            major_thickness.y = 0.0;
            minor_thickness.y = 0.0;
        }
        if ( uv.y < range_min.y || uv.y > range_max.y ) {
            major_thickness.x = 0.0;
            minor_thickness.x = 0.0;
        }
    $$ endif

    // Calculate grid alphas
    // Note that a step or distance of zero automatically results in the result
    // of the prestineGrid call to be either zero or nan.

    let axis_alpha = pristineGrid(uv, uv_clamped_axis, axis_step, axis_thickness);
    let major_alpha = pristineGrid(uv, uv_clamped_major, major_step, major_thickness);
    let minor_alpha = pristineGrid(uv, uv_clamped_minor, minor_step, minor_thickness);

    var alpha: f32 = 0.0;
    var color = vec4<f32>(0.0);

    if ( axis_alpha > alpha) {
        alpha = axis_alpha;
        color = axis_color;
    }
    if ( major_alpha > alpha * 1.5 ) {
        alpha = major_alpha;
        color = major_color;
    }
    if ( minor_alpha > alpha * 1.5 ) {
        alpha = minor_alpha;
        color = minor_color;
    }

    // ---------------------

    // Make physical color with combined alpha
    let physical_color = srgb2physical(color.rgb);
    let opacity = alpha * color.a * u_material.opacity;
    let out_color = vec4<f32>(physical_color, opacity);

    // We can apply clipping planes, but maybe a grid should not be clipped?
    // apply_clipping_planes(in.world_pos);

    var out = get_fragment_output(varyings.position.z, out_color);
    $$ if write_pick
        // Just the object id for now
        out.pick = pick_pack(u32(u_wobject.id), 20);
    $$ endif
    return out;
}


fn pristineGrid(uv: vec2<f32>, uv_clamped: vec2<f32>, step: vec2<f32>, lineWidth: vec2<f32>) -> f32 {
    // The Best Darn Grid Shader (okt 2023)
    // For details see https://bgolus.medium.com/the-best-darn-grid-shader-yet-727f9278b9d8
    // AK: I removed the black-white-swap logic, because our output is alpha, not
    // luminance. I limited the linewidth instead. Also added support for screen
    // space line width.
    let uv_scaled = uv / step;
    let uv_clamped_scaled = uv_clamped / step; // note that clamped uv will have nonzero derivs at clamp-edges
    let l2p:f32 = u_stdinfo.physical_size.x / u_stdinfo.logical_size.x;
    let ddx: vec2<f32> = dpdx(uv_scaled);
    let ddy: vec2<f32> = dpdy(uv_scaled);
    let uvDeriv = vec2<f32>(length(vec2<f32>(ddx.x, ddy.x)), length(vec2<f32>(ddx.y, ddy.y)));
    $$ if thickness_space == 'screen'
    let targetWidth = min(l2p * lineWidth * uvDeriv, vec2<f32>(0.5));  // lineWidth in screen space
    $$ else
    let targetWidth = min(lineWidth / step, 0.5 / step);  // lineWidth in world space
    $$ endif
    let drawWidth = clamp(targetWidth, uvDeriv, 0.5 / step);
    let lineAA = uvDeriv * 1.5;
    let gridUV = 1.0 - abs(fract(uv_clamped_scaled) * 2.0 - 1.0);
    var grid2: vec2<f32> = smoothstep(drawWidth + lineAA, drawWidth - lineAA, gridUV);
    grid2 = grid2 * clamp(targetWidth / drawWidth, vec2<f32>(0.0), vec2<f32>(1.0));
    grid2 = mix(grid2, targetWidth, clamp(uvDeriv * 2.0 - 1.0, vec2<f32>(0.0), vec2<f32>(1.0)));
    return mix(grid2.x, 1.0, grid2.y);
}
