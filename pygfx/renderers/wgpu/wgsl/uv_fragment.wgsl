$$ if map_uv is defined
    let map_uv = (u_map.transform * vec3<f32>(varyings.texcoord{{map_uv or ''}}, 1.0)).xy;
$$ endif

$$ if use_normal_map is defined
    let normal_map_uv = (u_normal_map.transform * vec3<f32>(varyings.texcoord{{normal_map_uv or ''}}, 1.0)).xy;
$$ endif

$$ if use_specular_map is defined
    let specular_map_uv = (u_specular_map.transform * vec3<f32>(varyings.texcoord{{specular_map_uv or ''}}, 1.0)).xy;
$$ endif

$$ if use_specular_intensity_map is defined
    let specular_intensity_map_uv = (u_specular_intensity_map.transform * vec3<f32>(varyings.texcoord{{specular_intensity_map_uv or ''}}, 1.0)).xy;
$$ endif

$$ if use_ao_map is defined
    let ao_map_uv = (u_ao_map.transform * vec3<f32>(varyings.texcoord{{ao_map_uv or ''}}, 1.0)).xy;
$$ endif

$$ if use_light_map is defined
    let light_map_uv = (u_light_map.transform * vec3<f32>(varyings.texcoord{{light_map_uv or ''}}, 1.0)).xy;
$$ endif

$$ if use_emissive_map is defined
    let emissive_map_uv = (u_emissive_map.transform * vec3<f32>(varyings.texcoord{{emissive_map_uv or ''}}, 1.0)).xy;
$$ endif

$$ if use_metalness_map is defined
    let metalness_map_uv = (u_metalness_map.transform * vec3<f32>(varyings.texcoord{{metalness_map_uv or ''}}, 1.0)).xy;
$$ endif

$$ if use_roughness_map is defined
    let roughness_map_uv = (u_roughness_map.transform * vec3<f32>(varyings.texcoord{{roughness_map_uv or ''}}, 1.0)).xy;
$$ endif

$$ if use_clearcoat_map is defined
    let clearcoat_map_uv = (u_clearcoat_map.transform * vec3<f32>(varyings.texcoord{{clearcoat_map_uv or ''}}, 1.0)).xy;
$$ endif

$$ if use_clearcoat_normal_map is defined
    let clearcoat_normal_map_uv = (u_clearcoat_normal_map.transform * vec3<f32>(varyings.texcoord{{clearcoat_normal_map_uv or ''}}, 1.0)).xy;
$$ endif

$$ if use_clearcoat_roughness_map is defined
    let clearcoat_roughness_map_uv = (u_clearcoat_roughness_map.transform * vec3<f32>(varyings.texcoord{{clearcoat_roughness_map_uv or ''}}, 1.0)).xy;
$$ endif

$$ if use_iridescence_map is defined
    let iridescence_map_uv = (u_iridescence_map.transform * vec3<f32>(varyings.texcoord{{iridescence_map_uv or ''}}, 1.0)).xy;
$$ endif

$$ if use_iridescence_thickness_map is defined
    let iridescence_thickness_map_uv = (u_iridescence_thickness_map.transform * vec3<f32>(varyings.texcoord{{iridescence_thickness_map_uv or ''}}, 1.0)).xy;
$$ endif

$$ if use_anisotropy_map is defined
    let anisotropy_map_uv = (u_anisotropy_map.transform * vec3<f32>(varyings.texcoord{{anisotropy_map_uv or ''}}, 1.0)).xy;
$$ endif

