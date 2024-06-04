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
