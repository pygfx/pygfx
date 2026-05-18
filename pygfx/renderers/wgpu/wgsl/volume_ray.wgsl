// Volume rendering via raycasting. Multiple modes supported.

{# Includes #}
{$ include 'pygfx.std.wgsl' $}
$$ if colormap_dim
    {$ include 'pygfx.colormap.wgsl' $}
$$ endif
{$ include 'pygfx.volume_common.wgsl' $}


struct VertexInput {
    @builtin(vertex_index) vertex_index : u32,
};


struct RenderOutput {
    color: vec4<f32>,  // The final color for the current ray
    coord: vec3<f32>,  // The texture coord (for picking info) for the current ray
    depth: f32,  // The depth to write for the current ray
};


$$ if mode in ['iso']
    // This mode uses lights

    $$ if num_point_lights > 0 or num_spot_lights > 0 or num_dir_lights > 0
        {$ include 'pygfx.light_phong.wgsl' $}

        struct ReflectedLight {
            direct_diffuse: vec3<f32>,
            direct_specular: vec3<f32>,
            indirect_diffuse: vec3<f32>,
            indirect_specular: vec3<f32>,
        };

        fn calculate_light(physical_albedo: vec3f, world_pos: vec3f, surface_normal: vec3f, view_dir: vec3f) -> vec3f {

            // Apply lighting
            var reflected_light: ReflectedLight = ReflectedLight(vec3<f32>(0.0), vec3<f32>(0.0), vec3<f32>(0.0), vec3<f32>(0.0));
            var geometry: GeometricContext;
            geometry.position = world_pos;
            geometry.normal = surface_normal;
            geometry.view_dir = view_dir;

            // The below lines are a copy of 'pygfx.light_phong_fragment.wgsl', but tweaked for volume materials
            var material: BlinnPhongMaterial;
            material.diffuse_color = physical_albedo;
            material.specular_color = srgb2physical(vec3f(0.2863));  // #111, matching default MeshPhongMaterial.specular, and default in ThreeJS
            material.specular_shininess = u_material.shininess;
            material.specular_strength = 1.0;

            // Apply RE_Direct, to populate the reflected_light struct
            {$ include 'pygfx.light_punctual.wgsl' $}

            let ambient_color = u_ambient_light.color.rgb;  // already physical
            let irradiance = ambient_color;
            RE_IndirectDiffuse( irradiance, geometry, material, &reflected_light );

            var physical_color = reflected_light.direct_diffuse + reflected_light.direct_specular + reflected_light.indirect_diffuse + reflected_light.indirect_specular;
            physical_color += srgb2physical(u_material.emissive_color.rgb);

            return physical_color;
        }
    $$ else
        // Previously, the iso render used hardcoded lights, so code that used it likely did not create gfx.DirectionalLight etc. For backwards compatibility
        // we therefore fall back to builtin lights when no (non-ambient) lights are present.
        {$ include 'pygfx.light_phong_simple.wgsl' $}
        fn calculate_light(physical_albedo: vec3f, world_pos: vec3f, surface_normal: vec3f, view_dir: vec3f) -> vec3f {
            return lighting_phong(surface_normal, view_dir, physical_albedo);
        }
    $$ endif

$$endif


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

    // Take care to take into account of the camera flipping any axii
    let cam_sign = sign(u_stdinfo.cam_transform[0][0] * u_stdinfo.cam_transform[1][1] * u_stdinfo.cam_transform[2][2]);

    // We calculate the NDC positions for the near and front clipping planes,
    // and transform these back to data coordinates. From these positions
    // we can construct the view vector in the fragment shader, which is then
    // resistant to perspective transforms. It also makes that if the camera
    // is inside the volume, only the part in front in rendered.
    // Note that the w component for these positions should be left intact.
    let ndc_pos1 = vec4<f32>(ndc_pos.xy, -1.0 * cam_sign * ndc_pos.w, ndc_pos.w);
    let ndc_pos2 = vec4<f32>(ndc_pos.xy, cam_sign * ndc_pos.w, ndc_pos.w);
    varyings.data_near_pos = vec4<f32>(ndc_to_data * ndc_pos1);
    varyings.data_far_pos = vec4<f32>(ndc_to_data * ndc_pos2);

    return varyings;
}


@fragment
fn fs_main(varyings: Varyings) -> FragmentOutput {

    // clipping planes
    {$ include 'pygfx.clipping_planes.wgsl' $}

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
    let render_out: RenderOutput = raycast(sizef, nsteps, start_coord, step_coord);

    do_alpha_test(render_out.color.a);

    // Create fragment output.
    var out: FragmentOutput;
    out.color = render_out.color;
    out.depth = render_out.depth;

    $$ if write_pick
    // The wobject-id must be 20 bits. In total it must not exceed 64 bits.
    out.pick = (
        pick_pack(u32(u_wobject.global_id), 20) +
        pick_pack(u32(render_out.coord.x * 16383.0), 14) +
        pick_pack(u32(render_out.coord.y * 16383.0), 14) +
        pick_pack(u32(render_out.coord.z * 16383.0), 14)
    );
    $$ endif
    return out;
}


// ---- The different supported render modes ----

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

        // Get world and ndc pos from the calculated texture coordinate
        let data_pos = the_coord * sizef - vec3<f32>(0.5, 0.5, 0.5);
        let world_pos = u_wobject.world_transform * vec4<f32>(data_pos, 1.0);
        let ndc_pos = u_stdinfo.projection_transform * u_stdinfo.cam_transform * world_pos;

        // Produce result
        var out: RenderOutput;
        out.color = out_color;
        out.coord = the_coord;
        out.depth = ndc_pos.z / ndc_pos.w;
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

        // Get world and ndc pos from the calculated texture coordinate
        let data_pos = the_coord * sizef - vec3<f32>(0.5, 0.5, 0.5);
        let world_pos = u_wobject.world_transform * vec4<f32>(data_pos, 1.0);
        let ndc_pos = u_stdinfo.projection_transform * u_stdinfo.cam_transform * world_pos;

        // Produce result
        var out: RenderOutput;
        out.color = out_color;
        out.coord = the_coord;
        out.depth = ndc_pos.z / ndc_pos.w;
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
            let physical_albedo = srgb2physical(color.rgb);
        $$ else
            let physical_albedo = color.rgb;
        $$ endif

        // Compute the surface normal
        var normal : vec3<f32>;
        var positive_value : vec4<f32>;
        var negative_value : vec4<f32>;
        let gradient_coord = 1.0 / sizef;

        negative_value = sample_vol(the_coord + vec3<f32>(-gradient_coord[0],0.0,0.0), sizef);
        positive_value = sample_vol(the_coord + vec3<f32>(gradient_coord[0],0.0,0.0), sizef);
        normal[0] = positive_value.r - negative_value.r;

        negative_value = sample_vol(the_coord + vec3(0.0,-gradient_coord[1],0.0), sizef);
        positive_value = sample_vol(the_coord + vec3(0.0,gradient_coord[1],0.0), sizef);
        normal[1] = positive_value.r - negative_value.r;

        negative_value = sample_vol(the_coord + vec3(0.0,0.0,-gradient_coord[2]), sizef);
        positive_value = sample_vol(the_coord + vec3(0.0,0.0,gradient_coord[2]), sizef);
        normal[2] = positive_value.r - negative_value.r;

        // Project normal to world space
        let normal_proj0 =  u_wobject.world_transform * vec4f(0.0, 0.0, 0.0, 1.0);
        let normal_proj1 =  u_wobject.world_transform * vec4f(normal, 1.0);
        normal = normalize(normal_proj1.xyz - normal_proj0.xyz);

        // Project step direction to world space
        let normal_proj2 =  u_wobject.world_transform * vec4f(-step_coord, 1.0);
        let view_dir = normalize(normal_proj2.xyz - normal_proj0.xyz);

        // Flip normal, if needed, see pygfx/issues/#105 for details
        let is_front = dot(normal, view_dir) > 0.0;
        var reoriented_normal = select(-normal, normal, is_front);

        // Get world and ndc pos from the calculated texture coordinate
        let data_pos = the_coord * sizef - vec3<f32>(0.5, 0.5, 0.5);
        let world_pos = u_wobject.world_transform * vec4<f32>(data_pos, 1.0);
        let ndc_pos = u_stdinfo.projection_transform * u_stdinfo.cam_transform * world_pos;

        let physical_color = calculate_light(physical_albedo, world_pos.xyz, reoriented_normal, view_dir);
        let opacity = color.a * u_material.opacity;
        let out_color = vec4<f32>(physical_color, opacity);

        // Produce result
        var out: RenderOutput;
        out.color = out_color;
        out.coord = the_coord;
        out.depth = ndc_pos.z / ndc_pos.w;
        return out;
    }

$$ else
    fn raycast(sizef: vec3<f32>, nsteps: i32, start_coord: vec3<f32>, step_coord: vec3<f32>) -> RenderOutput {
        {{ mode }}__is_not_a_valid_render_mode();
        var out: RenderOutput;
        return out;
    }
$$ endif
