from plot_shadertoy import Shadertoy

shader_code = """

// migrated from: https://www.shadertoy.com/view/mds3DX

let SHAPE_SIZE : f32 = .618;
let CHROMATIC_ABBERATION : f32 = .01;
let ITERATIONS : f32 = 10.;
let INITIAL_LUMA : f32= .5;

let PI : f32 = 3.14159265359;
let TWO_PI : f32 = 6.28318530718;

fn rotate2d(_angle : f32) -> mat2x2<f32> {
    return mat2x2<f32>(cos(_angle),-sin(_angle),sin(_angle),cos(_angle));
}

fn mod( v: vec2<f32>, y : f32 ) -> vec2<f32> {
    return vec2<f32>(v.x - y * floor( v.x / y ), v.y - y * floor( v.y / y ));
}

fn sdPolygon(angle : f32, distance : f32) -> f32 {
    let segment = TWO_PI / 4.0;
    return cos(floor(0.5 + angle / segment) * segment - angle) * distance;
}

fn getColorComponent( st: vec2<f32>, modScale : f32, blur : f32 ) -> f32 {
    let modSt = mod(st, 1. / modScale) * modScale * 2. - 1.;
    let dist = length(modSt);
    let angle = atan2(modSt.x, modSt.y) + sin(i_time * .08) * 9.0;
    let shapeMap = smoothStep(SHAPE_SIZE + blur, SHAPE_SIZE - blur, sin(dist * 3.0) * .5 + .5);
    return shapeMap;
}

fn shader_main(frag_coord: vec2<f32>) -> vec4<f32> {
    var blur = .4 + sin(i_time * .52) * .2;
    var st = (2.* frag_coord - i_resolution.xy) / min(i_resolution.x, i_resolution.y);
    let origSt = st;
    st *= rotate2d(sin(i_time * 0.14) * .3);
    st *= (sin( i_time * 0.15) + 2.0) * .3;
    st *= log(length(st * .428)) * 1.1;

    let modScale = 1.0;
    var color = vec3<f32>(0.0);

    var luma = INITIAL_LUMA;

    for (var i:f32 = 0.0; i < ITERATIONS; i+=1.0) {
        let center = st + vec2<f32>(sin(i_time * .12), cos(i_time * .13));
        let shapeColor = vec3<f32>(
            getColorComponent(center - st * CHROMATIC_ABBERATION, modScale, blur),
            getColorComponent(center, modScale, blur),
            getColorComponent(center + st * CHROMATIC_ABBERATION, modScale, blur)
        ) * luma;
        st *= 1.1 + getColorComponent(center, modScale, .04) * 1.2;
        st *= rotate2d(sin(i_time  * .05) * 1.33);
        color = color + shapeColor;
        color = vec3<f32>(clamp( color.r, 0.0, 1.0 ), clamp( color.g, 0.0, 1.0 ), clamp( color.b, 0.0, 1.0 ));
        luma *= .6;
        blur *= .63;
    }

    let GRADING_INTENSITY = .4;
    let topGrading = vec3<f32>(
        1. + sin(i_time * 1.13 * .3) * GRADING_INTENSITY,
        1. + sin(i_time * 1.23 * .3) * GRADING_INTENSITY,
        1. - sin(i_time * 1.33 * .3) * GRADING_INTENSITY
    );
    let bottomGrading = vec3<f32>(
        1. - sin(i_time * 1.43 * .3) * GRADING_INTENSITY,
        1. - sin(i_time * 1.53 * .3) * GRADING_INTENSITY,
        1. + sin(i_time * 1.63 * .3) * GRADING_INTENSITY
    );
    let origDist = length(origSt);
    let colorGrading = mix(topGrading, bottomGrading, origDist - .5);
    var fragColor = vec4<f32>(pow(color.rgb, colorGrading), 1.);
    // fragColor *= smoothStep(2.1, .7, origDist);
    return fragColor;
}

"""
shader = Shadertoy(shader_code, resolution=(800, 450))

if __name__ == "__main__":
    shader.show()
