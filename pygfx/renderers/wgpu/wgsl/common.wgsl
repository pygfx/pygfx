// ----- Math

const PI = 3.141592653589793;
const RECIPROCAL_PI = 0.3183098861837907;
fn pow2(x:f32) -> f32 { return x*x; }
fn pow4(x:f32) -> f32 { let x2 = x * x; return x2*x2; }

fn is_orthographic() -> bool {
    return u_stdinfo.projection_transform[2][3] == 0.0;
}

// ----- Blending and lighting placeholders

const alpha_compare_epsilon : f32 = 1e-6;
{{ blending_code }}

$$ if lighting
{{ light_definitions }}
$$ endif

fn refract( light : vec3<f32>, normal : vec3<f32>, eta : f32 ) -> vec3<f32> {
    let cosTheta = dot( -light, normal );
    let rOutPerp = eta * ( light + cosTheta * normal );
    let rOutParallel = -sqrt( abs( 1.0 - dot( rOutPerp, rOutPerp ) ) ) * normal;
    return rOutPerp + rOutParallel;
}

// ----- Transformations

fn ndc_to_world_pos(ndc_pos: vec4<f32>) -> vec3<f32> {
    let ndc_to_world = u_stdinfo.cam_transform_inv * u_stdinfo.projection_transform_inv;
    let world_pos = ndc_to_world * ndc_pos;
    return world_pos.xyz / world_pos.w;
}

// ----- Colors

fn srgb2physical(color: vec3<f32>) -> vec3<f32> {
    // In Python, the below reads as
    // c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
    let f = pow((color + 0.055) / 1.055, vec3<f32>(2.4));
    let t = color / 12.92;
    return select(f, t, color <= vec3<f32>(0.04045));
    // Simplified version with about 0.5% avg error
    // return pow(color, vec3<f32>(2.2));
}

$$ if colormap_dim
    fn sample_colormap(texcoord: {{ colormap_coord_type }}) -> vec4<f32> {

        // Determine colormap texture dimensions
        $$ if colormap_dim == '1d'
            let texcoords_dim = f32(textureDimensions(t_colormap));
        $$ elif colormap_dim == '2d'
            let texcoords_dim = vec2<f32>(textureDimensions(t_colormap));
        $$ elif colormap_dim == '3d'
            let texcoords_dim = vec3<f32>(textureDimensions(t_colormap));
        $$ endif

        // Get final texture coord. With linear interpolation, the colormap's endpoints represent the min and max.
        $$ if colormap_interpolation == 'nearest'
            let tf = texcoord;
        $$ else
            let tf = texcoord * (texcoords_dim - 1.0) / texcoords_dim + 0.5 / texcoords_dim;
        $$ endif

        // Sample in the colormap. We get a vec4 color, but not all channels may be used.
        $$ if not colormap_dim
            let color_value = vec4<f32>(0.0);
        $$ elif colormap_format == 'f32'
            let color_value = textureSample(t_colormap, s_colormap, tf);
        $$ else
            $$ if colormap_dim == '1d'
            let ti = i32(tf * texcoords_dim % texcoords_dim);
            $$ elif colormap_dim == '2d'
            let ti = vec2<i32>(tf * texcoords_dim % texcoords_dim);
            $$ elif colormap_dim == '3d'
            let ti = vec3<i32>(tf * texcoords_dim % texcoords_dim);
            $$ endif
            let color_value = vec4<f32>(textureLoad(t_colormap, ti, 0));
        $$ endif

        // Depending on the number of channels we makeGfxTextureView grayscale, rgb, etc.
        $$ if colormap_nchannels == 1
            let color = vec4<f32>(color_value.rrr, 1.0);
        $$ elif colormap_nchannels == 2
            let color = vec4<f32>(color_value.rrr, color_value.g);
        $$ elif colormap_nchannels == 3
            let color = vec4<f32>(color_value.rgb, 1.0);
        $$ else
            let color = vec4<f32>(color_value.rgb, color_value.a);
        $$ endif
        return color;
    }
$$ endif


// ----- Clipping planes

$$ if not n_clipping_planes
    fn check_clipping_planes(world_pos: vec3<f32>) -> bool { return true; }
    fn apply_clipping_planes(world_pos: vec3<f32>) { }
$$ else
    fn check_clipping_planes(world_pos: vec3<f32>) -> bool {
        var clipped: bool = {{ 'false' if clipping_mode == 'ANY' else 'true' }};
        for (var i=0; i<{{ n_clipping_planes }}; i=i+1) {
            let plane = u_material.clipping_planes[i];
            let plane_clipped = dot( world_pos, plane.xyz ) < plane.w;
            clipped = clipped {{ '||' if clipping_mode == 'ANY' else '&&' }} plane_clipped;
        }
        return !clipped;
    }
    fn apply_clipping_planes(world_pos: vec3<f32>) {
        if (!(check_clipping_planes(world_pos))) { discard; }
    }
$$ endif


// ----- Picking

var<private> p_pick_bits_used : i32 = 0;

fn pick_pack(value: u32, bits: i32) -> vec4<u32> {
    // Utility to pack multiple values into a rgba16uint (64 bits available).
    // Note that we store in a vec4<u32> but this gets written to a 4xu16.
    // See #212 for details.
    //
    // Clip the given value
    let maxval = u32(exp2(f32(bits))) - u32(1);
    let v = max(u32(0), min(value, maxval));
    // Determine bit-shift for each component
    let shift = vec4<i32>(
        p_pick_bits_used, p_pick_bits_used - 16, p_pick_bits_used - 32, p_pick_bits_used - 48,
    );
    // Prepare for next pack
    p_pick_bits_used = p_pick_bits_used + bits;
    // Apply the shift for each component
    let vv = vec4<u32>(v);
    let selector1 = vec4<bool>(shift[0] < 0, shift[1] < 0, shift[2] < 0, shift[3] < 0);
    let pick_new = select( vv << vec4<u32>(shift) , vv >> vec4<u32>(-shift) , selector1 );
    // Mask the components
    let mask = vec4<u32>(65535u);
    let selector2 = vec4<bool>( abs(shift[0]) < 32, abs(shift[1]) < 32, abs(shift[2]) < 32, abs(shift[3]) < 32 );
    return select( vec4<u32>(0u) , pick_new & mask , selector2 );
}