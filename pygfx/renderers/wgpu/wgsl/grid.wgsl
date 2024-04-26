// Main shader for rendering grids.

{# Includes #}
{$ include 'pygfx.std.wgsl' $}


struct VertexInput {
    @builtin(vertex_index) index : u32,
};


@vertex
fn vs_main(in: VertexInput) -> Varyings {
    var varyings: Varyings;
    // Define positions at the four corners of the viewport, at the largest depth

    // Limits as xmin,xmax,ymin,ymax
    let u_limits1 = vec4<f32>(-100.0, 100.0, -100.0, 100.0);


    let cam_pos = vec3<f32>(
        u_stdinfo.cam_transform_inv[3][0],
        u_stdinfo.cam_transform_inv[3][1],
        u_stdinfo.cam_transform_inv[3][2],
        // u_stdinfo.cam_transform_inv[0][3],
        // u_stdinfo.cam_transform_inv[1][3],
        // u_stdinfo.cam_transform_inv[2][3],
    );
    let distance_cam_to_grid = abs(cam_pos.y - 0.0);

    var positions = array<vec3<f32>, 4>(
        vec3<f32>(u_limits1[0], 0.0, u_limits1[2]),
        vec3<f32>(u_limits1[1], 0.0, u_limits1[2]),
        vec3<f32>(u_limits1[0], 0.0, u_limits1[3]),
        vec3<f32>(u_limits1[1], 0.0, u_limits1[3]),
    );

    var coords = array<vec2<f32>, 4>(
        vec2<f32>(-0.5, -0.5),
        vec2<f32>( 0.5, -0.5),
        vec2<f32>(-0.5,  0.5),
        vec2<f32>( 0.5,  0.5),
    );
    // Select the current position
    let pos = positions[i32(in.index)] * distance_cam_to_grid + vec3<f32>(cam_pos.x, 0.0, cam_pos.z);
    let coord = coords[i32(in.index)];

    var pos_ndc = u_stdinfo.projection_transform * u_stdinfo.cam_transform * vec4<f32>(pos, 1.0);// u_wobject.world_transform * pos;

    // Store positions and the view direction in the world
    varyings.position = vec4<f32>(pos_ndc);
    varyings.world_pos = vec3<f32>(ndc_to_world_pos(out.position));
    varyings.gridcoord = vec3<f32>(pos.x, pos.z, 1.0);
    varyings.texcoord = vec2<f32>(coord);
    return varyings;
}


@fragment
fn fs_main(varyings: Varyings) -> FragmentOutput {



    // For now, use variables from the paper

    // Line antialias area (usually 1 pixel)
    let u_antialias = 0.707;
    // Cartesian and projected limits as xmin,xmax,ymin,ymax
    let u_limits1 = vec4<f32>(-100.0, 100.0, -100.0, 100.0);
    let u_limits2 =  vec4<f32>(-99.0, 99.0, -99.0, 99.0);
    // Major and minor grid steps
    let u_major_grid_step = vec2<f32>(1.0, 1.0);
    let u_minor_grid_step = vec2<f32>(0.1, 0.1);
    // Major and minor grid line widths (1.50 pixel, 0.75 pixel)
    let u_major_grid_width = f32(0.01);
    let u_minor_grid_width = f32(0.005);
    // Major grid line color
    let u_major_grid_color = u_material.major_color;
    // Minor grid line color
    let u_minor_grid_color = u_material.minor_color;
    // Texture coordinates (from (-0.5,-0.5) to (+0.5,+0.5)
    let v_texcoord = vec2<f32>(varyings.texcoord);
    // Viewport resolution (in physical pixels)



    // ---------------------

    let NP1 = v_texcoord;
    let P1 = scale_forward(NP1, u_limits1);
    let P2 = transform_inverse(P1);
    // Test if we are within limits but we do not discard the
    // fragment yet because we want to draw a border. Discarding
    // would mean that the exterior would not be drawn.
    var outside = vec2<bool>(false, false);
    if( P2.x < u_limits2[0] ) { outside.x = true; }
    if( P2.x > u_limits2[1] ) { outside.x = true; }
    if( P2.y < u_limits2[2] ) { outside.y = true; }
    if( P2.y > u_limits2[3] ) { outside.y = true; }
    let NP2 = scale_inverse(P2, u_limits2);
    var P: vec2<f32>;
    var tick: f32;
    // Major tick, X axis
    tick = get_tick(NP2.x + 0.5, u_limits2[0], u_limits2[1], u_major_grid_step[0]);
    P = transform_forward(vec2<f32>(tick, P2.y));
    P = scale_inverse(P, u_limits1);
    // float Mx = length(v_size * (NP1 - P));
    // Here we assume the quad is contained in the XZ plane
    let Mx = screen_distance(vec4<f32>(NP1, 0.0, 1.0), vec4<f32>(P, 0.0, 1.0));
    // Minor tick, X axis
    tick = get_tick(NP2.x + 0.5, u_limits2[0], u_limits2[1], u_minor_grid_step[0]);
    P = transform_forward(vec2<f32>(tick, P2.y));
    P = scale_inverse(P, u_limits1);
    // float mx = length(v_size * (NP1 - P));
    // Here we assume the quad is contained in the XZ plane
    let mx = screen_distance(vec4<f32>(NP1, 0.0, 1.0), vec4<f32>(P, 0.0, 1.0));
    // Major tick, Y axis
    tick = get_tick(NP2.y + 0.5, u_limits2[2], u_limits2[3], u_major_grid_step[1]);
    P = transform_forward(vec2<f32>(P2.x, tick));
    P = scale_inverse(P, u_limits1);
    // float My = length(v_size * (NP1 - P));
    // Here we assume the quad is contained in the XZ plane
    let My = screen_distance(vec4<f32>(NP1, 0.0, 1.0), vec4<f32>(P, 0.0, 1.0));
    // Minor tick, Y axis
    tick = get_tick(NP2.y + 0.5, u_limits2[2], u_limits2[3], u_minor_grid_step[1]);
    P = transform_forward(vec2<f32>(P2.x, tick));
    P = scale_inverse(P, u_limits1);
    // float my = length(v_size * (NP1 - P));
    // Here we assume the quad is contained in the XZ plane
    let my = screen_distance(vec4<f32>(NP1, 0.0, 1.0), vec4<f32>(P, 0.0, 1.0));
    var M = min(Mx, My);
    var m = min(mx, my);
    // Here we take care of "finishing" the border lines
    if ( outside.x && outside.y ) {
        if (Mx > 0.5 * (u_major_grid_width + u_antialias)) {
            discard;
        } else if (My > 0.5 * (u_major_grid_width + u_antialias)) {
            discard;
        } else {
            M = max(Mx, My);
        }
    } else if ( outside.x ) {
        if (Mx > 0.5 * (u_major_grid_width + u_antialias)) {
            discard;
        } else {
            M = Mx;
            m = Mx;
        }
    } else if ( outside.y ) {
        if (My > 0.5 * (u_major_grid_width + u_antialias)) {
            discard;
        } else {
            M = My;
            m = My;
        }
    }
    // // Mix major/minor colors to get dominant color
    // let alpha1 = stroke_alpha(M, u_major_grid_width, u_antialias);
    // let alpha2 = stroke_alpha(m, u_minor_grid_width, u_antialias);
    // var alpha = alpha1;
    // var color: vec4<f32> = u_major_grid_color;
    // if( alpha2 > alpha1 * 1.5 ) {
    //     alpha = alpha2;
    //     color = u_minor_grid_color;
    // }
    // // Without extra cost, we can also project a texture
    // // vec4 texcolor = texture2D(u_texture, vec2<f32>(NP2.x, 1.0-NP2.y));


    let uv = vec2<f32>(varyings.gridcoord.xy);
    let thickness = vec2<f32>(u_material.major_thickness, u_material.major_thickness);
    let alpha = pristineGrid(uv, thickness);
    let color = u_material.major_color;


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


fn pristineGrid(uv: vec2<f32>, lineWidth: vec2<f32>) -> f32 {
    // The Best Darn Grid Shader (okt 2023)
    // For details see https://bgolus.medium.com/the-best-darn-grid-shader-yet-727f9278b9d8#5ef5
    // I removed the black-white-swap logic, because our output is alpha, not luminance. I limited the linewidth instead.
    let l2p:f32 = u_stdinfo.physical_size.x / u_stdinfo.logical_size.x;
    let ddx: vec2<f32> = dpdx(uv);
    let ddy: vec2<f32> = dpdy(uv);
    let uvDeriv = vec2<f32>(length(vec2<f32>(ddx.x, ddy.x)), length(vec2<f32>(ddx.y, ddy.y)));
    $$ if thickness_space == 'screen'
    let targetWidth = min(l2p * lineWidth * uvDeriv, vec2<f32>(0.5));  // lineWidth in screen space
    $$ else
    let targetWidth = min(lineWidth, vec2<f32>(0.5));  // lineWidth in world space
    $$ endif
    let drawWidth = clamp(targetWidth, uvDeriv, vec2<f32>(0.5));  // line width in world space
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
