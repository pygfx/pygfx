// todo: now, to prevent byte align issue, we use mat3x4f as map_transform, but it should be mat3x3f.
// We should avoid do conversion in shader, to improve performance and simplify code.
$$ if map_uv is defined
    let map_transform = mat3x3<f32>(u_map.transform[0].xyz, u_map.transform[1].xyz, u_map.transform[2].xyz);
    let map_uv = (map_transform * vec3<f32>(varyings.texcoord{{map_uv or ''}}, 1.0)).xy;
$$ endif

$$ if use_normal_map is defined
    let normal_map_transform = mat3x3<f32>(u_normal_map.transform[0].xyz, u_normal_map.transform[1].xyz, u_normal_map.transform[2].xyz);
    let normal_map_uv = (normal_map_transform * vec3<f32>(varyings.texcoord{{normal_map_uv or ''}}, 1.0)).xy;
$$ endif

$$ if use_specular_map is defined
    let specular_map_transform = mat3x3<f32>(u_specular_map.transform[0].xyz, u_specular_map.transform[1].xyz, u_specular_map.transform[2].xyz);
    let specular_map_uv = (specular_map_transform * vec3<f32>(varyings.texcoord{{specular_map_uv or ''}}, 1.0)).xy;
$$ endif

$$ if use_specular_intensity_map is defined
    let specular_intensity_map_transform = mat3x3<f32>(u_specular_intensity_map.transform[0].xyz, u_specular_intensity_map.transform[1].xyz, u_specular_intensity_map.transform[2].xyz);
    let specular_intensity_map_uv = (specular_intensity_map_transform * vec3<f32>(varyings.texcoord{{specular_intensity_map_uv or ''}}, 1.0)).xy;
$$ endif

$$ if use_ao_map is defined
    let ao_map_transform = mat3x3<f32>(u_ao_map.transform[0].xyz, u_ao_map.transform[1].xyz, u_ao_map.transform[2].xyz);
    let ao_map_uv = (ao_map_transform * vec3<f32>(varyings.texcoord{{ao_map_uv or ''}}, 1.0)).xy;
$$ endif

$$ if use_light_map is defined
    let light_map_transform = mat3x3<f32>(u_light_map.transform[0].xyz, u_light_map.transform[1].xyz, u_light_map.transform[2].xyz);
    let light_map_uv = (light_map_transform * vec3<f32>(varyings.texcoord{{light_map_uv or ''}}, 1.0)).xy;
$$ endif

$$ if use_emissive_map is defined
    let emissive_map_transform = mat3x3<f32>(u_emissive_map.transform[0].xyz, u_emissive_map.transform[1].xyz, u_emissive_map.transform[2].xyz);
    let emissive_map_uv = (emissive_map_transform * vec3<f32>(varyings.texcoord{{emissive_map_uv or ''}}, 1.0)).xy;
$$ endif

$$ if use_metalness_map is defined
    let metalness_map_transform = mat3x3<f32>(u_metalness_map.transform[0].xyz, u_metalness_map.transform[1].xyz, u_metalness_map.transform[2].xyz);
    let metalness_map_uv = (metalness_map_transform * vec3<f32>(varyings.texcoord{{metalness_map_uv or ''}}, 1.0)).xy;
$$ endif

$$ if use_roughness_map is defined
    let roughness_map_transform = mat3x3<f32>(u_roughness_map.transform[0].xyz, u_roughness_map.transform[1].xyz, u_roughness_map.transform[2].xyz);
    let roughness_map_uv = (roughness_map_transform * vec3<f32>(varyings.texcoord{{roughness_map_uv or ''}}, 1.0)).xy;
$$ endif

$$ if use_clearcoat_map is defined
    let clearcoat_map_transform = mat3x3<f32>(u_clearcoat_map.transform[0].xyz, u_clearcoat_map.transform[1].xyz, u_clearcoat_map.transform[2].xyz);
    let clearcoat_map_uv = (clearcoat_map_transform * vec3<f32>(varyings.texcoord{{clearcoat_map_uv or ''}}, 1.0)).xy;
$$ endif

$$ if use_clearcoat_normal_map is defined
    let clearcoat_normal_map_transform = mat3x3<f32>(u_clearcoat_normal_map.transform[0].xyz, u_clearcoat_normal_map.transform[1].xyz, u_clearcoat_normal_map.transform[2].xyz);
    let clearcoat_normal_map_uv = (clearcoat_normal_map_transform * vec3<f32>(varyings.texcoord{{clearcoat_normal_map_uv or ''}}, 1.0)).xy;
$$ endif

$$ if use_clearcoat_roughness_map is defined
    let clearcoat_roughness_map_transform = mat3x3<f32>(u_clearcoat_roughness_map.transform[0].xyz, u_clearcoat_roughness_map.transform[1].xyz, u_clearcoat_roughness_map.transform[2].xyz);
    let clearcoat_roughness_map_uv = (clearcoat_roughness_map_transform * vec3<f32>(varyings.texcoord{{clearcoat_roughness_map_uv or ''}}, 1.0)).xy;
$$ endif

$$ if use_iridescence_map is defined
    let iridescence_map_transform = mat3x3<f32>(u_iridescence_map.transform[0].xyz, u_iridescence_map.transform[1].xyz, u_iridescence_map.transform[2].xyz);
    let iridescence_map_uv = (iridescence_map_transform * vec3<f32>(varyings.texcoord{{iridescence_map_uv or ''}}, 1.0)).xy;
$$ endif

$$ if use_iridescence_thickness_map is defined
    let iridescence_thickness_map_transform = mat3x3<f32>(u_iridescence_thickness_map.transform[0].xyz, u_iridescence_thickness_map.transform[1].xyz, u_iridescence_thickness_map.transform[2].xyz);
    let iridescence_thickness_map_uv = (iridescence_thickness_map_transform * vec3<f32>(varyings.texcoord{{iridescence_thickness_map_uv or ''}}, 1.0)).xy;
$$ endif

$$ if use_anisotropy_map is defined
    let anisotropy_map_transform = mat3x3<f32>(u_anisotropy_map.transform[0].xyz, u_anisotropy_map.transform[1].xyz, u_anisotropy_map.transform[2].xyz);
    let anisotropy_map_uv = (anisotropy_map_transform * vec3<f32>(varyings.texcoord{{anisotropy_map_uv or ''}}, 1.0)).xy;
$$ endif

