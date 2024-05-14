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
    // plane is in the xz plane (normal to the y-axis) but this is more or less
    // arbitrary.
    let vertex_grid_coord: vec2<f32> = grid_coords[i32(in.index)];
    let coord1 = vertex_grid_coord.x;
    let coord2 = vertex_grid_coord.y;

    // Calculate reference points
    let p0 = (u_wobject.world_transform * vec4<f32>(0.0, 0.0, 0.0, 1.0)).xyz;  // "origin" of the plane
    let p1 = (u_wobject.world_transform * vec4<f32>(1.0, 0.0, 0.0, 1.0)).xyz;  // on the plane
    let p2 = (u_wobject.world_transform * vec4<f32>(0.0, 0.0, 1.0, 1.0)).xyz;  // on the plane
    let p3 = (u_wobject.world_transform * vec4<f32>(0.0, 1.0, 0.0, 1.0)).xyz;  // out of plane!

    // Get vectors for the plane's axii, expressed in world coordinates
    let v1 = p1 - p0; // todo: normalize?
    let v2 = p2 - p0;
    let v3 = p3 - p0;

    $$ if inf_grid

        // Get description of the grid plane: ax * by  + cz + d == 0
        let abc = normalize(v3);  // == the plane's normal
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

        // The grid's transform defines its place in the world
        let pos = (u_wobject.world_transform * vec4<f32>(coord1, 0.0, coord2, 1.0)).xyz;

    $$ endif

    var pos_ndc = u_stdinfo.projection_transform * u_stdinfo.cam_transform * vec4<f32>(pos, 1.0);

    // Store positions and the view direction in the world
    varyings.position = vec4<f32>(pos_ndc);
    varyings.world_pos = vec3<f32>(ndc_to_world_pos(out.position));
    varyings.gridcoord = vec2<f32>(dot(v1, pos), dot(v2, pos));
    return varyings;
}


@fragment
fn fs_main(varyings: Varyings) -> FragmentOutput {

    // Collect props

    let major_step= vec2<f32>(u_material.major_step);
    let minor_step = vec2<f32>(u_material.minor_step);

    let axis_thickness = vec2<f32>(u_material.axis_thickness);
    let major_thickness = vec2<f32>(u_material.major_thickness);
    let minor_thickness = vec2<f32>(u_material.minor_thickness);

    let axis_color = u_material.axis_color;
    let major_color = u_material.major_color;
    let minor_color = u_material.minor_color;

    // Calculate grid alphas
    // Note that a step or distance of zero automatically results in the result
    // of the prestineGrid call to be either zero or nan.

    let uv = vec2<f32>(varyings.gridcoord.xy);

    let axis_alpha = pristineGrid(clamp(uv, vec2<f32>(-0.5), vec2<f32>(0.5)), vec2<f32>(1.0), axis_thickness);
    let major_alpha = pristineGrid(uv, major_step, major_thickness);
    let minor_alpha = pristineGrid(uv, minor_step, minor_thickness);

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
    //let out_color = vec4<f32>(1.0, 0.0, 0.0, 1.0);

    // We can apply clipping planes, but maybe a grid should not be clipped?
    // apply_clipping_planes(in.world_pos);

    var out = get_fragment_output(varyings.position.z, out_color);
    $$ if write_pick
        // Just the object id for now
        out.pick = pick_pack(u32(u_wobject.id), 20);
    $$ endif
    return out;
}


fn pristineGrid(uv_raw: vec2<f32>, step: vec2<f32>, lineWidth: vec2<f32>) -> f32 {
    // The Best Darn Grid Shader (okt 2023)
    // For details see https://bgolus.medium.com/the-best-darn-grid-shader-yet-727f9278b9d8#5ef5
    // I removed the black-white-swap logic, because our output is alpha, not luminance. I limited the linewidth instead.
    let uv = uv_raw / step;
    let l2p:f32 = u_stdinfo.physical_size.x / u_stdinfo.logical_size.x;
    let ddx: vec2<f32> = dpdx(uv);
    let ddy: vec2<f32> = dpdy(uv);
    let uvDeriv = vec2<f32>(length(vec2<f32>(ddx.x, ddy.x)), length(vec2<f32>(ddx.y, ddy.y)));
    $$ if thickness_space == 'screen'
    let targetWidth = min(l2p * lineWidth * uvDeriv, vec2<f32>(0.5));  // lineWidth in screen space
    $$ else
    let targetWidth = min(lineWidth / step, 0.5 / step);  // lineWidth in world space
    $$ endif
    let drawWidth = clamp(targetWidth, uvDeriv, 0.5 / step);
    let lineAA = uvDeriv * 1.5;
    var gridUV = abs(fract(uv) * 2.0 - 1.0);
    gridUV.x = 1.0 - gridUV.x;
    gridUV.y = 1.0 - gridUV.y;
    var grid2: vec2<f32> = smoothstep(drawWidth + lineAA, drawWidth - lineAA, gridUV);
    grid2 = grid2 * clamp(targetWidth / drawWidth, vec2<f32>(0.0), vec2<f32>(1.0));
    grid2 = mix(grid2, targetWidth, clamp(uvDeriv * 2.0 - 1.0, vec2<f32>(0.0), vec2<f32>(1.0)));
    return mix(grid2.x, 1.0, grid2.y);
}


// Antialias stroke alpha coeff
fn stroke_alpha(distance: f32, linewidth: f32, antialias: f32) -> f32 {
    let t = linewidth / 2.0 - antialias;
    let signed_distance = distance;
    let border_distance = abs(signed_distance) - t;
    var alpha = border_distance / antialias;
    //alpha = exp(-alpha * alpha);
    if ( border_distance > (linewidth / 2.0 + antialias) ) {
        return 0.0;
    } else if ( border_distance < 0.0 ) {
        return 1.0;
    } else {
        return alpha;
    }
}

// Compute the nearest tick from a (normalized) t value
fn get_tick(t: f32, vmin: f32, vmax: f32, step: f32) -> f32 {
    let first_tick = floor((vmin + step / 2.0) / step) * step;
    let  last_tick = floor((vmax + step / 2.0) / step) * step;
    var tick = vmin + t * (vmax-vmin);
    if (tick < (vmin + (first_tick-vmin) / 2.0)) {
        return vmin;
    }
    if (tick > (last_tick + (vmax-last_tick) / 2.0)) {
        return vmax;
    }
    tick = tick + step / 2.0;
    tick = floor(tick / step) * step;
    return min(max(vmin, tick), vmax);
}

// Compute the distance (in physical pixels) between p1 and p2
fn screen_distance(p1: vec4<f32>, p2: vec4<f32>) -> f32 {
    let resolution = vec2<f32>(u_stdinfo.physical_size);
    var p1_ndc = u_stdinfo.projection_transform * u_stdinfo.cam_transform * u_wobject.world_transform * p1;
    p1_ndc = p1_ndc / p1_ndc.w;
    var p2_ndc = u_stdinfo.projection_transform * u_stdinfo.cam_transform * u_wobject.world_transform * p2;
    p2_ndc = p2_ndc / p2_ndc.w;
    let p1_physical = vec2<f32>(0.5 * p1_ndc.xy * resolution);
    let p2_physical = vec2<f32>(0.5 * p2_ndc.xy * resolution);
    return length(p1_physical - p2_physical);
}


fn transform_forward(p: vec2<f32>) -> vec2<f32> {
    return p;
}

fn transform_inverse(p: vec2<f32>) -> vec2<f32> {
    return p;
}


// [-0.5,-0.5]x[0.5,0.5] -> [xmin,xmax]x[ymin,ymax]
fn scale_forward(p: vec2<f32>, limits: vec4<f32>) -> vec2<f32> {
    // limits = xmin,xmax,ymin,ymax
    var p2 = p + vec2<f32>(0.5, 0.5);
    p2 = p2 * vec2<f32>(limits[1] - limits[0], limits[3]-limits[2]);
    p2 = p2 + vec2<f32>(limits[0], limits[2]);
    return p2;
}

// [xmin,xmax]x[ymin,ymax] -> [-0.5,-0.5]x[0.5,0.5]
fn scale_inverse(p: vec2<f32>, limits: vec4<f32>) -> vec2<f32> {
    // limits = xmin,xmax,ymin,ymax
    var p2 = p - vec2<f32>(limits[0], limits[2]);
    p2 = p2 / vec2<f32>(limits[1]-limits[0], limits[3]-limits[2]);
    return p2 - vec2<f32>(0.5, 0.5);
}
