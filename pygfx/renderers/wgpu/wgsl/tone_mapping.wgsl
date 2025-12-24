$$ if tone_mapping_mode == "linear"
    // exposure only
    fn toneMapping(color: vec3<f32>) -> vec3<f32> {
        return saturate(color);
    }
$$ endif

$$ if tone_mapping_mode == "reinhard"
    // source: https://www.cs.utah.edu/docs/techreports/2002/pdf/UUCS-02-001.pdf
    fn toneMapping(color: vec3<f32>) -> vec3<f32> {
        return saturate(color / (vec3<f32>(1.0) + color));
    }
$$ endif

$$ if tone_mapping_mode == "cineon"
    // source: http://filmicworlds.com/blog/filmic-tonemapping-operators/
    fn toneMapping(color: vec3<f32>) -> vec3<f32> {
        // filmic operator by Jim Hejl and Richard Burgess-Dawson
        var mapped_color = max(vec3<f32>(0.0), color - vec3<f32>(0.004));
        return pow((mapped_color * (6.2 * mapped_color + 0.5)) / (mapped_color * (6.2 * mapped_color + 1.7) + 0.06), vec3<f32>(2.2));
    }
$$ endif

$$ if tone_mapping_mode == "aces_filmic"
    // source: https://github.com/selfshadow/ltc_code/blob/master/webgl/shaders/ltc/ltc_blit.fs
    fn RRTAndODTFit(v: vec3<f32>) -> vec3<f32> {
        let a = v * (v + vec3<f32>(0.0245786)) - vec3<f32>(0.000090537);
        let b = v * (vec3<f32>(0.983729) * v + vec3<f32>(0.4329510)) + vec3<f32>(0.238081);
        return a / b;
    }

    // this implementation of ACES is modified to accommodate a brighter viewing environment.
    // the scale factor of 1/0.6 is subjective. see discussion in three.js #19621.

    fn toneMapping(color: vec3<f32>) -> vec3<f32> {
        // sRGB => XYZ => D65_2_D60 => AP1 => RRT_SAT
        let ACESInputMat = mat3x3<f32>(
            vec3<f32>(0.59719, 0.07600, 0.02840),
            vec3<f32>(0.35458, 0.90834, 0.13383),
            vec3<f32>(0.04823, 0.01566, 0.83777)
        );
        // ODT_SAT => XYZ => D60_2_D65 => sRGB
        let ACESOutputMat = mat3x3<f32>(
            vec3<f32>(1.60475, -0.10208, -0.00327),
            vec3<f32>(-0.53108, 1.10813, -0.07276),
            vec3<f32>(-0.07367, -0.00605, 1.07602)
        );
        var mapped_color = color / 0.6;
        mapped_color = ACESInputMat * mapped_color;

        // Apply RRT and ODT
        mapped_color = RRTAndODTFit(mapped_color);
        mapped_color = ACESOutputMat * mapped_color;
        // Clamp to [0, 1]
        return saturate(mapped_color);
    }
$$ endif

$$ if tone_mapping_mode == "agx"
    // Matrices for rec 2020 <> rec 709 color space conversion
    // matrix provided in row-major order so it has been transposed
    // https://www.itu.int/pub/R-REP-BT.2407-201
    const LINEAR_REC2020_TO_LINEAR_SRGB = mat3x3f(
        vec3f( 1.6605, - 0.1246, - 0.0182 ),
        vec3f( - 0.5876, 1.1329, - 0.1006 ),
        vec3f( - 0.0728, - 0.0083, 1.1187 )
    );

    const LINEAR_SRGB_TO_LINEAR_REC2020 = mat3x3f(
        vec3f( 0.6274, 0.0691, 0.0164 ),
        vec3f( 0.3293, 0.9195, 0.0880 ),
        vec3f( 0.0433, 0.0113, 0.8956 )
    );

    // https://iolite-engine.com/blog_posts/minimal_agx_implementation
    // Mean error^2: 3.6705141e-06
    fn agxDefaultContrastApprox( x: vec3<f32> ) -> vec3<f32> {

        let x2 = x * x;
        let x4 = x2 * x2;

        return 15.5 * x4 * x2
            - 40.14 * x4 * x
            + 31.96 * x4
            - 6.868 * x2 * x
            + 0.4298 * x2
            + 0.1191 * x
            - 0.00232;
    }

    // AgX Tone Mapping implementation based on Filament, which in turn is based
    // on Blender's implementation using rec 2020 primaries
    // https://github.com/google/filament/pull/7236
    // Inputs and outputs are encoded as Linear-sRGB.

    fn toneMapping( color: vec3<f32> ) -> vec3<f32> {
        // AgX constants
        let AgXInsetMatrix = mat3x3f(
            vec3f( 0.856627153315983, 0.137318972929847, 0.11189821299995 ),
            vec3f( 0.0951212405381588, 0.761241990602591, 0.0767994186031903 ),
            vec3f( 0.0482516061458583, 0.101439036467562, 0.811302368396859 )
        );
        // explicit AgXOutsetMatrix generated from Filaments AgXOutsetMatrixInv
        let AgXOutsetMatrix = mat3x3f(
            vec3f( 1.1271005818144368, - 0.1413297634984383, - 0.14132976349843826 ),
            vec3f( - 0.11060664309660323, 1.157823702216272, - 0.11060664309660294 ),
            vec3f( - 0.016493938717834573, - 0.016493938717834257, 1.2519364065950405 )
        );

        let AgxMinEv = - 12.47393;  // log2( pow( 2, LOG2_MIN ) * MIDDLE_GRAY )
        let AgxMaxEv = 4.026069;    // log2( pow( 2, LOG2_MAX ) * MIDDLE_GRAY )

        var mapped_color = LINEAR_SRGB_TO_LINEAR_REC2020 * color;
        mapped_color = AgXInsetMatrix * mapped_color;
        // Log2 encoding
        mapped_color = max( mapped_color, vec3f( 1e-10 ) ); // avoid 0 or negative numbers for log2
        mapped_color = log2( mapped_color );
        mapped_color = ( mapped_color - vec3f( AgxMinEv ) ) / vec3f( AgxMaxEv - AgxMinEv );
        mapped_color = clamp( mapped_color, vec3f( 0.0 ), vec3f( 1.0 ) );
        // Apply sigmoid
        // mapped_color = 1.0 / ( 1.0 + exp( - 10.0 * ( mapped_color - 0.5 ) ) );
        mapped_color = agxDefaultContrastApprox( mapped_color );
        // Apply AgX look
        // v = agxLook(v, look);
        mapped_color = AgXOutsetMatrix * mapped_color;
        // Linearize
        mapped_color = pow( max( vec3f( 0.0 ), mapped_color), vec3f( 2.2 ) );
        mapped_color = LINEAR_REC2020_TO_LINEAR_SRGB * mapped_color;
        // Gamut mapping. Simple clamp for now.
        mapped_color = clamp( mapped_color, vec3f( 0.0 ), vec3f( 1.0 ) );
        return mapped_color;

    }
$$ endif

$$ if tone_mapping_mode == "neutral"

    // https://modelviewer.dev/examples/tone-mapping
    fn toneMapping(color: vec3<f32>) -> vec3<f32> {
        let StartCompression = 0.8 - 0.04;
        let Desaturation = 0.15;

        let x = min(color.r, min(color.g, color.b));
        let offset = select(0.04, x - 6.25 * x * x, x < 0.08);

        var mapped_color = color - vec3<f32>(offset);

        let peak = max(mapped_color.r, max(mapped_color.g, mapped_color.b));
        
        if (peak < StartCompression) {
            return mapped_color;
        }
        
        let d = 1.0 - StartCompression;
        let new_peak = 1.0 - d * d / (peak + d - StartCompression);

        mapped_color *= (new_peak / peak);

        let g = 1.0 - 1.0 / (Desaturation * (peak - new_peak) + 1.0);
        
        return mix(mapped_color, vec3<f32>(new_peak), g);
    }

$$ endif