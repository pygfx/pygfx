// Volume rendering via raycasting. Multiple modes supported.

{# Includes #}
{$ include 'pygfx.std.wgsl' $}
$$ if colormap_dim
    {$ include 'pygfx.colormap.wgsl' $}
$$ endif
$$ if mode == 'iso'
    {$ include 'pygfx.light_phong_simple.wgsl' $}
$$ endif
{$ include 'pygfx.volume_common.wgsl' $}


struct VertexInput {
    @builtin(vertex_index) vertex_index : u32,
};


@vertex
fn vs_main(in: VertexInput) -> Varyings {

    // Our geometry is implicitly defined by the volume dimensions.
    var geo = get_vol_geometry();

    // Select what face we're at
    let index = i32(in.vertex_index);
    let i0 = geo.indices[index];

    // Sample position, and convert to world pos, and then to ndc
    let data_pos = vec4<f32>(geo.positions[i0], 1.0);
    let world_pos = u_wobject.world_transform * data_pos;
    let ndc_pos = u_stdinfo.projection_transform * u_stdinfo.cam_transform * world_pos;

    // Prepare inverse matrix
    let ndc_to_data = u_wobject.world_transform_inv * u_stdinfo.cam_transform_inv * u_stdinfo.projection_transform_inv;

    var varyings: Varyings;

    // Store values for fragment shader
    varyings.position = vec4<f32>(ndc_pos);
    varyings.world_pos = vec3<f32>(world_pos.xyz);

    // The position on the face of the cube. We can say that it's the back face,
    // because we cull the front faces.
    // These positions are in data positions (voxels) rather than texcoords (0..1),
    // because distances make more sense in this space. In the fragment shader we
    // can consider it an isotropic volume, because any position, rotation,
    // and scaling of the volume is part of the world transform.
    varyings.data_back_pos = vec4<f32>(data_pos);

    // We calculate the NDC positions for the near and front clipping planes,
    // and transform these back to data coordinates. From these positions
    // we can construct the view vector in the fragment shader, which is then
    // resistant to perspective transforms. It also makes that if the camera
    // is inside the volume, only the part in front in rendered.
    // Note that the w component for these positions should be left intact.
    let ndc_pos1 = vec4<f32>(ndc_pos.xy, -ndc_pos.w, ndc_pos.w);
    let ndc_pos2 = vec4<f32>(ndc_pos.xy, ndc_pos.w, ndc_pos.w);
    varyings.data_near_pos = vec4<f32>(ndc_to_data * ndc_pos1);
    varyings.data_far_pos = vec4<f32>(ndc_to_data * ndc_pos2);

    return varyings;
}


@fragment
fn fs_main(varyings: Varyings) -> FragmentOutput {

    // Get size of the volume
    let sizef = vec3<f32>(textureDimensions(t_img));

    // Determine the stepsize as a float in pixels.
    // This value should be between ~ 0.1 and 1. Smaller values yield better
    // results at the cost of performance. With larger values you may miss
    // small structures (and corners of larger structures) because the step
    // may skip over them.
    // We could make this a user-facing property. But for now we scale between
    // 0.1 and 0.8 based on the (sqrt of the) volume size.
    let relative_step_size = clamp(sqrt(max(sizef.x, max(sizef.y, sizef.z))) / 20.0, 0.1, 0.8);

    // Positions in data coordinates
    let back_pos = varyings.data_back_pos.xyz / varyings.data_back_pos.w;
    let far_pos = varyings.data_far_pos.xyz / varyings.data_far_pos.w;
    let near_pos = varyings.data_near_pos.xyz / varyings.data_near_pos.w;

    // Calculate unit vector pointing in the view direction through this fragment.
    let view_ray = normalize(far_pos - near_pos);

    // Calculate the (signed) distance, from back_pos to the first voxel
    // that must be sampled, expressed in data coords (voxels).
    var dist = dot(near_pos - back_pos, view_ray);
    dist = max(dist, min((-0.5 - back_pos.x) / view_ray.x, (sizef.x - 0.5 - back_pos.x) / view_ray.x));
    dist = max(dist, min((-0.5 - back_pos.y) / view_ray.y, (sizef.y - 0.5 - back_pos.y) / view_ray.y));
    dist = max(dist, min((-0.5 - back_pos.z) / view_ray.z, (sizef.z - 0.5 - back_pos.z) / view_ray.z));

    // Now we have the starting position. This is typically on a front face,
    // but it can also be incide the volume (on the near plane).
    let front_pos = back_pos + view_ray * dist;

    // Decide how many steps to take. If we'd not cul the front faces,
    // that would still happen here because nsteps would be negative.
    let nsteps = i32(-dist / relative_step_size + 0.5);
    if( nsteps < 1 ) { discard; }

    // Get starting position and step vector in texture coordinates.
    let start_coord = (front_pos + vec3<f32>(0.5, 0.5, 0.5)) / sizef;
    let step_coord = ((back_pos - front_pos) / sizef) / f32(nsteps);

    // Render
    let render_out = raycast(sizef, nsteps, start_coord, step_coord);

    // Get world and ndc pos from the calculatex texture coordinate
    let data_pos = render_out.coord * sizef - vec3<f32>(0.5, 0.5, 0.5);
    let world_pos = u_wobject.world_transform * vec4<f32>(data_pos, 1.0);
    let ndc_pos = u_stdinfo.projection_transform * u_stdinfo.cam_transform * world_pos;

    // Maybe we did the work for nothing
    apply_clipping_planes(world_pos.xyz);

    // Get fragment output. Note that the depth arg only affects the
    // blending - setting the depth attribute actually sets the fragment depth.
    let depth = ndc_pos.z / ndc_pos.w;
    var out = get_fragment_output(depth, render_out.color);
    out.depth = depth;

    $$ if write_pick
    // The wobject-id must be 20 bits. In total it must not exceed 64 bits.
    out.pick = (
        pick_pack(u32(u_wobject.id), 20) +
        pick_pack(u32(render_out.coord.x * 16383.0), 14) +
        pick_pack(u32(render_out.coord.y * 16383.0), 14) +
        pick_pack(u32(render_out.coord.z * 16383.0), 14)
    );
    $$ endif
    return out;
}


// ---- The different supported render modes ----


struct RenderOutput {
    color: vec4<f32>,
    coord: vec3<f32>,
};

$$ if mode == 'mip'
    // raycasting function for MIP rendering.
    fn raycast(sizef: vec3<f32>, nsteps: i32, start_coord: vec3<f32>, step_coord: vec3<f32>) -> RenderOutput {
        // Ideas for improvement:
        // * We could textureLoad() the 27 voxels surrounding the initial location
        //   and sample from that in the refinement step. Less texture loads and we
        //   could do linear interpolation also for formats like i16.
        // * Create helper textures at a lower resolution (e.g. min, max) so we can
        //   skip along the ray much faster. By taking smaller steps where needed,
        //   it will be both faster and more accurate.

        let nstepsf = f32(nsteps);

        // Primary loop. The purpose is to find the approximate location where
        // the maximum is.
        var the_ref = -999999.0;
        var the_coord = start_coord;
        var the_value : vec4<f32>;
        for (var iter=0.0; iter<nstepsf; iter=iter+1.0) {
            let coord = start_coord + iter * step_coord;
            let value = sample_vol(coord, sizef);
            let reff = value.r;
            if (reff > the_ref) {
                the_ref = reff;
                the_coord = coord;
                the_value = value;
            }
        }

        // Secondary loop to close in on a more accurate position using
        // a divide-by-two approach.
        var substep_coord = step_coord;
        for (var iter2=0; iter2<4; iter2=iter2+1) {
            substep_coord = substep_coord * 0.5;
            let coord1 = the_coord - substep_coord;
            let coord2 = the_coord + substep_coord;
            let value1 = sample_vol(coord1, sizef);
            let value2 = sample_vol(coord2, sizef);
            let ref1 = value1.r;
            let ref2 = value2.r;
            if (ref1 >= the_ref) {  // deliberate larger-equal
                the_ref = ref1;
                the_coord = coord1;
                the_value = value1;
            } else if (ref2 > the_ref) {
                the_ref = ref2;
                the_coord = coord2;
                the_value = value2;
            }
        }

        // Colormapping
        let color = sampled_value_to_color(the_value);
        // Move to physical colorspace (linear photon count) so we can do math
        $$ if colorspace == 'srgb'
            let physical_color = srgb2physical(color.rgb);
        $$ else
            let physical_color = color.rgb;
        $$ endif
        let opacity = color.a * u_material.opacity;
        let out_color = vec4<f32>(physical_color, opacity);

        // Produce result
        var out: RenderOutput;
        out.color = out_color;
        out.coord = the_coord;
        return out;
    }

$$ elif mode == 'minip'
    // raycasting function for MINimum Intensity Projection rendering.
    fn raycast(sizef: vec3<f32>, nsteps: i32, start_coord: vec3<f32>, step_coord: vec3<f32>) -> RenderOutput {
        // Ideas for improvement:
        // * We could textureLoad() the 27 voxels surrounding the initial location
        //   and sample from that in the refinement step. Less texture loads and we
        //   could do linear interpolation also for formats like i16.
        // * Create helper textures at a lower resolution (e.g. min, max) so we can
        //   skip along the ray much faster. By taking smaller steps where needed,
        //   it will be both faster and more accurate.

        let nstepsf = f32(nsteps);

        // Primary loop. The purpose is to find the approximate location where
        // the maximum is.
        var the_ref = 999999.0;
        var the_coord = start_coord;
        var the_value : vec4<f32>;
        for (var iter=0.0; iter<nstepsf; iter=iter+1.0) {
            let coord = start_coord + iter * step_coord;
            let value = sample_vol(coord, sizef);
            let reff = value.r;
            if (reff < the_ref) {
                the_ref = reff;
                the_coord = coord;
                the_value = value;
            }
        }

        // Secondary loop to close in on a more accurate position using
        // a divide-by-two approach.
        var substep_coord = step_coord;
        for (var iter2=0; iter2<4; iter2=iter2+1) {
            substep_coord = substep_coord * 0.5;
            let coord1 = the_coord - substep_coord;
            let coord2 = the_coord + substep_coord;
            let value1 = sample_vol(coord1, sizef);
            let value2 = sample_vol(coord2, sizef);
            let ref1 = value1.r;
            let ref2 = value2.r;
            if (ref1 <= the_ref) {  // deliberate less-equal
                the_ref = ref1;
                the_coord = coord1;
                the_value = value1;
            } else if (ref2 < the_ref) {
                the_ref = ref2;
                the_coord = coord2;
                the_value = value2;
            }
        }

        // Colormapping
        let color = sampled_value_to_color(the_value);
        // Move to physical colorspace (linear photon count) so we can do math
        $$ if colorspace == 'srgb'
            let physical_color = srgb2physical(color.rgb);
        $$ else
            let physical_color = color.rgb;
        $$ endif
        let opacity = color.a * u_material.opacity;
        let out_color = vec4<f32>(physical_color, opacity);

        // Produce result
        var out: RenderOutput;
        out.color = out_color;
        out.coord = the_coord;
        return out;
    }

$$ elif mode == 'iso'
    fn raycast(sizef: vec3<f32>, nsteps: i32, start_coord: vec3<f32>, step_coord: vec3<f32>) -> RenderOutput {
        // Ideas for improvement:
        // * We could use the scene lighting.
        // * Create helper textures at a lower resolution (e.g. min, max) so we can
        //   skip along the ray much faster. By taking smaller steps where needed,
        //   it will be both faster and more accurate.

        let nstepsf = f32(nsteps);

        // Primary loop. The purpose is to find the approximate location where
        // the surface is.
        let iso_threshold = u_material.threshold;
        let actual_step_coord = u_material.step_size * step_coord;
        var surface_found = false;
        var the_coord = start_coord;
        var the_value : vec4<f32>;
        for (var iter=0.0; iter<nstepsf; iter=iter+1) {
            let coord = start_coord + iter * actual_step_coord;
            let value = sample_vol(coord, sizef);
            let reff = value.r;
            if (reff > iso_threshold) {
                the_coord = coord;
                the_value = value;
                surface_found = true;
                break;
            }
        }

        if surface_found {
            // take smaller steps back to make sure the surface was found
            let substep_coord = -1 * u_material.substep_size * step_coord;
            let substep_start_coord = the_coord;
            let max_iter = 1 / u_material.substep_size;
            for (var iter=1.0; iter<max_iter; iter=iter+1) {
                let coord = substep_start_coord + iter * substep_coord;
                let value = sample_vol(coord, sizef);
                let reff = value.r;

                if (reff < iso_threshold){
                    // stop if the coord is now outside the surface
                    break;
                }

                // update the coord and values
                the_coord = coord;
                the_value = value;

            }

        }
        else {
            // if no surface found, discard the fragment
            discard;
        }


        // Colormapping
        let color = sampled_value_to_color(the_value);
        // Move to physical colorspace (linear photon count) so we can do math
        $$ if colorspace == 'srgb'
            let physical_color = srgb2physical(color.rgb);
        $$ else
            let physical_color = color.rgb;
        $$ endif

        // Compute the normal
        var normal : vec3<f32>;
        var positive_value : vec4<f32>;
        var negative_value : vec4<f32>;
        let gradient_coord = 1.5 * step_coord;

        negative_value = sample_vol(the_coord + vec3<f32>(-gradient_coord[0],0.0,0.0), sizef);
        positive_value = sample_vol(the_coord + vec3<f32>(gradient_coord[0],0.0,0.0), sizef);
        normal[0] = positive_value.r - negative_value.r;

        negative_value = sample_vol(the_coord + vec3(0.0,-gradient_coord[1],0.0), sizef);
        positive_value = sample_vol(the_coord + vec3(0.0,gradient_coord[1],0.0), sizef);
        normal[1] = positive_value.r - negative_value.r;

        negative_value = sample_vol(the_coord + vec3(0.0,0.0,-gradient_coord[2]), sizef);
        positive_value = sample_vol(the_coord + vec3(0.0,0.0,gradient_coord[2]), sizef);
        normal[2] = positive_value.r - negative_value.r;
        normal = normalize(normal);

        // Do the lighting
        let view_direction = normalize(step_coord);
        let is_front = dot(normal, view_direction) > 0.0;
        let lighted_color = lighting_phong(is_front, normal, view_direction, physical_color);

        let opacity = color.a * u_material.opacity;
        let out_color = vec4<f32>(lighted_color, opacity);

        // Produce result
        var out: RenderOutput;
        out.color = out_color;
        out.coord = the_coord;
        return out;
    }

$$ endif