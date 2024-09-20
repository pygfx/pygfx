// Main shader for rendering grids.

{# Includes #}
{$ include 'pygfx.std.wgsl' $}


struct VertexInput {
    @builtin(vertex_index) index : u32,
};


@vertex
fn vs_main(in: VertexInput) -> Varyings {
    var varyings: Varyings;

    // Calculate reference points
    let p0 = (u_wobject.world_transform * vec4<f32>(0.0, 0.0, 0.0, 1.0)).xyz;  // "origin" of the plane
    let p1 = (u_wobject.world_transform * vec4<f32>(1.0, 0.0, 0.0, 1.0)).xyz;  // on the plane
    let p2 = (u_wobject.world_transform * vec4<f32>(0.0, 1.0, 0.0, 1.0)).xyz;  // on the plane
    let p3 = (u_wobject.world_transform * vec4<f32>(0.0, 0.0, 1.0, 1.0)).xyz;  // out of plane!

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

        // Get position of camera projected on the plane
        let cam_pos = vec3<f32>(u_stdinfo.cam_transform_inv[3].xyz);
        let cam_k = (dot(abc, cam_pos) + d) / length(abc);
        let cam_pos_on_grid = cam_pos - cam_k * abc;

        // The closer the camera is to the plane, the less space you need to
        // make the horizon look really far away. Scaling too hard will
        // introduce fluctuation artifacts due to inaccurate float arithmatic.
        // Scaling too softly results in a non-horizontal horizon. To avoid a
        // compromise and have the best of both, we define a small and a big
        // quad.
        //
        // Note thay with an ortho camera there is no proper horizon, making the
        // inf grid not well suited for use with an ortho camera.
        //
        // The grid coordinates form a big quad, with a small quad inside (without overlap).
        //
        //  x-----------------x
        //  |\\           / / |
        //  | \ \      /  /   |
        //  |  \  \ /   /     |
        //  |   \  x---x      |
        //  |    \ |   | \    |
        //  |      x---x  \   |
        //  |     /   / \  \  |
        //  |   /  /      \ \ |
        //  | / /           \\|
        //  x-----------------x

        // Get scale multiplier for near quad.
        var near_multiplier = vec2<f32>(1.0);
        if is_orthographic() {
            // With an orthographic camera, we use the scale factor (camera
            // width/height), because it best represent the "scene size". The scale
            // factors for x and y will often be the same, but not necessarily, e.g.
            // maintain_aspect=False. We make the assumption that when the scale factors
            // are not the same, we're dealing with a 2D scene, and the camera is
            // looking straight at the grid, without rotation. This seems like a pretty
            // safe assumption.
            near_multiplier = vec2<f32>(
                1.0 / u_stdinfo.projection_transform[0][0],
                1.0 / u_stdinfo.projection_transform[1][1]
            );
        } else {
            // For perspective cameras we use the distance to the plane and assume an aspect ratio of 1.
            let distance_cam_to_grid = abs(cam_k); // == distance(cam_pos, cam_pos_on_grid);
            near_multiplier = vec2<f32>(5.0 * distance_cam_to_grid);
        }

        // The far quad is simply 200x further
        let far_multiplier = 200.0 * near_multiplier;

        let far_scale = 2.0;
        var grid_coords = array<vec2<f32>,8>(
            vec2<f32>(-1.0, -1.0) * near_multiplier,
            vec2<f32>( 1.0, -1.0) * near_multiplier,
            vec2<f32>(-1.0,  1.0) * near_multiplier,
            vec2<f32>( 1.0,  1.0) * near_multiplier,
            vec2<f32>(-1.0, -1.0) * far_multiplier,
            vec2<f32>( 1.0, -1.0) * far_multiplier,
            vec2<f32>(-1.0,  1.0) * far_multiplier,
            vec2<f32>( 1.0,  1.0) * far_multiplier,
        );
        var coord_indices = array<i32, 30>(
            0, 1, 2, 2, 1, 3,  // close
            4, 5, 0, 0, 5, 1,
            5, 7, 1, 1, 7, 3,
            7, 6, 3, 3, 6, 2,
            6, 4, 2, 2, 4, 0,
        );

        // Select the grid coord for this vertex. We express it with coord1 and
        // coord2, to avoid confusion with xyz world coordinates. By default the
        // plane is in the xz plane (normal to the y-axis).
        let index = i32(in.index);
        let vertex_grid_coord: vec2<f32> = grid_coords[coord_indices[index]];
        let coord1 = vertex_grid_coord.x;
        let coord2 = vertex_grid_coord.y;

        // Construct position using only the grid's rotation. Scale and offset are overridden.
        let pos = cam_pos_on_grid + coord1 * v1 + coord2 * v2;

    $$ else

        // Grid coordinates to form a quad
        var grid_coords = array<vec2<f32>, 4>(
            vec2<f32>(0.0, 0.0),
            vec2<f32>(1.0, 0.0),
            vec2<f32>(0.0, 1.0),
            vec2<f32>(1.0, 1.0),
        );
        var coord_indices = array<i32, 6>(0, 1, 2, 2, 1, 3);

        // Select the grid coord for this vertex. We express it with coord1 and
        // coord2, to avoid confusion with xyz world coordinates. By default the
        // plane is in the xz plane (normal to the y-axis).
        let vertex_grid_coord: vec2<f32> = grid_coords[coord_indices[i32(in.index)]];
        let coord1 = vertex_grid_coord.x;
        let coord2 = vertex_grid_coord.y;

        // The grid's transform defines its place in the world.
        // Add a 2% margin to have space for line width and aa.
        let pos = (u_wobject.world_transform * vec4<f32>((coord1 * 1.04 - 0.02), (coord2 * 1.04 - 0.02), 0.0, 1.0)).xyz;

        // Calculate range, so that in the frag shader the edges can be handled correctly.
        let p_min = (u_wobject.world_transform * vec4<f32>(0.0, 0.0, 0.0, 1.0)).xyz;
        let p_max = (u_wobject.world_transform * vec4<f32>(1.0, 1.0, 0.0, 1.0)).xyz;
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

    // Collect thicknesses. These are also used to hide regions of the grid as needed.
    var axis_thickness = vec2<f32>(u_material.axis_thickness);
    var major_thickness = vec2<f32>(u_material.major_thickness);
    var minor_thickness = vec2<f32>(u_material.minor_thickness);

    // Collect step sizes.
    let major_step= vec2<f32>(u_material.major_step);
    let minor_step = vec2<f32>(u_material.minor_step);
    let axis_step = max(major_step * 10.0, axis_thickness * 10.0);

    // Collect colors.
    let axis_color = u_material.axis_color;
    let major_color = u_material.major_color;
    let minor_color = u_material.minor_color;



    // Get the grid coordinate
    let uv = vec2<f32>(varyings.gridcoord.xy);

    // Apply axis limits, so they have one line for each dimension.
    if (abs(uv.x) > 0.5 * axis_step.x) { axis_thickness.x = 0.0; }
    if (abs(uv.y) > 0.5 * axis_step.y) { axis_thickness.y = 0.0; }

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
        if ( uv.x < range_major_min.x || uv.x > range_major_max.x ) { major_thickness.x = 0.0; }
        if ( uv.y < range_major_min.y || uv.y > range_major_max.y ) { major_thickness.y = 0.0; }
        if ( uv.x < range_minor_min.x || uv.x > range_minor_max.x ) { minor_thickness.x = 0.0; }
        if ( uv.y < range_minor_min.y || uv.y > range_minor_max.y ) { minor_thickness.y = 0.0; }

        // The lines orthogonal to the edge, should simply not be draw beyond the edge.
        if ( uv.x < range_min.x || uv.x > range_max.x ) { major_thickness.y = 0.0; minor_thickness.y = 0.0; }
        if ( uv.y < range_min.y || uv.y > range_max.y ) { major_thickness.x = 0.0; minor_thickness.x = 0.0; }

    $$ endif

    // Calculate grid alphas.
    // Note that a step or distance of zero automatically results in the result
    // of the prestine_grid call to be either zero or nan.

    var alpha: f32 = 0.0;
    var color = vec4<f32>(0.0);

    $$ if draw_axis
        let axis_alpha = pristine_grid(uv, axis_thickness, axis_step);
        if ( axis_alpha > alpha) {
            alpha = axis_alpha;
            color = axis_color;
        }
    $$ endif
    $$ if draw_major
        let major_alpha = pristine_grid(uv, major_thickness, major_step);
        if ( major_alpha > alpha * 1.5 ) {
            alpha = major_alpha;
            color = major_color;
        }
    $$ endif
    $$ if draw_minor
        let minor_alpha = pristine_grid(uv, minor_thickness, minor_step);
        if ( minor_alpha > alpha * 1.5 ) {
            alpha = minor_alpha;
            color = minor_color;
        }
    $$ endif

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


fn pristine_grid(uv: vec2<f32>, lineWidth: vec2<f32>, step: vec2<f32>) -> f32 {
    // The Best Darn Grid Shader (okt 2023), by Ben Golus.
    // For details see: https://bgolus.medium.com/the-best-darn-grid-shader-yet-727f9278b9d8
    // We removed the black-white-swap logic, because our output is alpha, not
    // luminance. The targetWidth is limited instead. Also added support for a
    // custom step size, and expressing line width in screen space.

    // Scale uv based on step. Only this and lineWidth need to be scaled.
    let uvScaled = uv / step;

    // Calculate derivatives.
    let ddx: vec2<f32> = dpdx(uvScaled);
    let ddy: vec2<f32> = dpdy(uvScaled);
    let uvDeriv = vec2<f32>(length(vec2<f32>(ddx.x, ddy.x)), length(vec2<f32>(ddx.y, ddy.y)));

    // Get scaled line width, expressed in uv_scaled, supporting both screen and world space.
    $$ if thickness_space == 'screen'
        let l2p:f32 = u_stdinfo.physical_size.x / u_stdinfo.logical_size.x;
        let lineWidthScaled = l2p * lineWidth * uvDeriv;  // lineWidth in screen space
    $$ else
        let lineWidthScaled = lineWidth / step;  // lineWidth in world space
    $$ endif

    // Get target width and draw width.
    let targetWidth = clamp(lineWidthScaled, vec2<f32>(0.0), vec2<f32>(0.5));
    let drawWidth = clamp(targetWidth, uvDeriv, vec2<f32>(0.5));

    // anti-aliasing, anti-moiré, and some black magic.
    let lineAA = uvDeriv * 1.5;  // Also seen in example shader: ``max(uvDeriv, vec2<f32>(1e-6)) * 1.5;``
    let gridUV = 1.0 - abs(fract(uvScaled) * 2.0 - 1.0);
    var grid2: vec2<f32> = smoothstep(drawWidth + lineAA, drawWidth - lineAA, gridUV);
    grid2 = grid2 * clamp(targetWidth / drawWidth, vec2<f32>(0.0), vec2<f32>(1.0));
    grid2 = mix(grid2, targetWidth, clamp(uvDeriv * 2.0 - 1.0, vec2<f32>(0.0), vec2<f32>(1.0)));
    return mix(grid2.x, 1.0, grid2.y);
}
