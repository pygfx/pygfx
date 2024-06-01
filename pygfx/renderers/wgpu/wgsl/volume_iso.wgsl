fn render_mode_iso(sizef: vec3<f32>, nsteps: i32, start_coord: vec3<f32>, step_coord: vec3<f32>) -> RenderOutput {
    // Ideas for improvement:
    // * We could textureLoad() the 27 voxels surrounding the initial location
    //   and sample from that in the refinement step. Less texture loads and we
    //   could do linear interpolation also for formats like i16.
    // * Create helper textures at a lower resolution (e.g. min, max) so we can
    //   skip along the ray much faster. By taking smaller steps where needed,
    //   it will be both faster and more accurate.

    let nstepsf = f32(nsteps);

    // Primary loop. The purpose is to find the approximate location where
    // the surface is.
    let iso_threshold = u_material.isosurface_threshold;
    var surface_found = false;
    var the_coord = start_coord;
    var the_value : vec4<f32>;
    for (var iter=0.0; iter<nstepsf; iter=iter+1) {
        let coord = start_coord + iter * step_coord;
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
        var substep_coord = -0.1 * step_coord;
        var substep_start_coord = the_coord;
        for (var iter=1.0; iter<10; iter=iter+1) {
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
    let opacity = color.a * u_material.opacity;
    let out_color = vec4<f32>(physical_color, opacity);

    // Produce result
    var out: RenderOutput;
    out.color = out_color;
    out.coord = the_coord;
    return out;
}