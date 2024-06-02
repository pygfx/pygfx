fn lighting_phong(
    is_front: bool,
    normal: vec3<f32>,
    view_dir: vec3<f32>,
    albeido: vec3<f32>,
) -> vec3<f32> {
    // This is a simple implementation of
    //
    let light_color = srgb2physical(vec3<f32>(1.0, 1.0, 1.0));

    // Light parameters
    let ambient_factor = 0.1;
    let diffuse_factor = 0.7;
    let specular_factor = 0.3;
    let shininess = 30.0;

    // Base vectors
    let view = normalize(view_dir);
    let light = view;
    var reoriented_normal = select(-normal, normal, is_front);  // See pygfx/issues/#105 for details

    // Ambient
    let ambient_color = light_color * ambient_factor;

    // Diffuse (blinn-phong reflection model)
    let lambert_term = saturate(dot(light, reoriented_normal));
    let diffuse_color = diffuse_factor * light_color * lambert_term;

    // Specular
    let halfway = normalize(light + view);  // halfway vector
    var specular_term = pow(saturate(dot(halfway,  reoriented_normal)), shininess);
    specular_term = select(0.0, specular_term, shininess > 0.0);
    let specular_color = specular_factor * specular_term * light_color;

    // Put together
    return albeido * (ambient_color + diffuse_color) + specular_color;
}

fn render_mode_iso(sizef: vec3<f32>, nsteps: i32, start_coord: vec3<f32>, step_coord: vec3<f32>) -> RenderOutput {
    // Ideas for improvement:
    // * We could use the scene lighting.
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
        let substep_coord = -0.1 * step_coord;
        let substep_start_coord = the_coord;
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