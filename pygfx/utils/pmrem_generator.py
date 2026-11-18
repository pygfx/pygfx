import numpy as np
import wgpu
import pygfx as gfx
from pygfx.renderers.wgpu.engine.shared import get_shared
from pygfx.utils.cube_camera import CubeCamera

# Reference: https://learnopengl.com/PBR/IBL/Specular-IBL

prefilter_shader = """
struct Params {
  roughness: f32,
  resolution: f32,
};

@group(0) @binding(0)
var srcTex: texture_cube<f32>;

@group(0) @binding(1)
var s: sampler;

@group(0) @binding(2)
var<uniform> params: Params;

@group(1) @binding(0)
var destTex: texture_storage_2d_array<rgba16float, write>;

const PI: f32 = 3.141592653589793;
const SAMPLE_COUNT: u32 = 4096;

//compute the direction vector for a given cube map face and texel coordinate
fn getCubeDirection(face: u32, uv_: vec2<f32>) -> vec3<f32> {
  let uv = 2.0 * uv_ - 1.0;
  switch (face) {
    case 0u: {
        return vec3f(1.0, -uv.y, -uv.x);  // +X
    }
    case 1u: {
        return vec3f(-1.0, -uv.y, uv.x);  // -X
    }
    case 2u: {
        return vec3f(uv.x, 1.0, uv.y);    // +Y
    }
    case 3u: {
        return vec3f(uv.x, -1.0, -uv.y);  // -Y
    }
    case 4u: {
        return vec3f(uv.x, -uv.y, 1.0);   // +Z
    }
    case 5u: {
        return vec3f(-uv.x, -uv.y, -1.0); // -Z
    }
    default: {
        return vec3f(0.0);
    }
  }
}

// Hammersley sequence
fn hammersley(i: u32, N: u32) -> vec2<f32> {
    var bits = (i << 16u) | (i >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    let radicalInverse = f32(bits) * 2.3283064365386963e-10;
    return vec2f(f32(i)/f32(N), radicalInverse);
}

// GGX importance sampling
fn importanceSampleGGX(xi: vec2<f32>, N: vec3<f32>, roughness: f32) -> vec3<f32> {
  let a = roughness * roughness;
  
  let phi = 2.0 * PI * xi.x;
  let cosTheta = sqrt((1.0 - xi.y) / (1.0 + (a*a - 1.0) * xi.y));
  let sinTheta = sqrt(1.0 - cosTheta * cosTheta);
  
  let H = vec3f(
    cos(phi) * sinTheta,
    sin(phi) * sinTheta,
    cosTheta
  );
  
  let up = select(vec3f(1.0, 0.0, 0.0), vec3f(0.0, 0.0, 1.0), abs(N.z) < 0.999);
  let tangent = normalize(cross(up, N));
  let bitangent = cross(N, tangent);
  
  let sampleVec = tangent * H.x + bitangent * H.y + N * H.z;
  return normalize(sampleVec);
}

// GGX normal distribution function
fn distributionGGX(NdotH: f32, roughness: f32) -> f32 {
  let a = roughness * roughness;
  let a2 = a * a;
  let denom = NdotH * NdotH * (a2 - 1.0) + 1.0;
  return a2 / (PI * denom * denom);
}

// todo: override workgroup_size
const workgroup_size: u32 = 8u;

@compute
@workgroup_size(workgroup_size, workgroup_size)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let face = id.z;

  let uv = (vec2<f32>(id.xy) + 0.5) / vec2<f32>(params.resolution);
  
  let N = normalize(getCubeDirection(face, uv));
  let R = N;
  let V = R;
  
  var prefilteredColor = vec3<f32>(0.0);
  var totalWeight = 0.0;
  
  for (var i: u32 = 0u; i < SAMPLE_COUNT; i = i + 1u) {
    let xi = hammersley(i, SAMPLE_COUNT);
    let H = importanceSampleGGX(xi, N, params.roughness);
    let L = normalize(2.0 * dot(V, H) * H - V);
    
    // let NdotL = max(dot(N, L), 0.0);
    // let NdotL = clamp(dot(N, L), 0.0, 1.0);
    let NdotL = saturate(dot(N, L));
    if (NdotL > 0.0) {
      let NdotH = clamp(dot(N, H), 0.0, 1.0);
      let HdotV = clamp(dot(H, V), 0.0, 1.0);
      
      let D = distributionGGX(NdotH, params.roughness);
      let pdf = D * NdotH / (4.0 * HdotV) + 0.0001;
      let texSize = textureDimensions(srcTex).x;
      let saTexel = 4.0 * PI / (6.0 * f32(texSize) * f32(texSize));
      let saSample = 1.0 / (f32(SAMPLE_COUNT) * pdf + 0.0001);
      let mipLevel = select(0.5 * log2(saSample / saTexel), 0.0, params.roughness == 0.0);

      prefilteredColor += textureSampleLevel(srcTex, s, L, mipLevel).rgb * NdotL;
      totalWeight += NdotL;
    }
  }

  prefilteredColor = prefilteredColor / totalWeight;
  textureStore(destTex, vec2<u32>(id.xy), face, vec4<f32>(prefilteredColor, 1.0));
}
"""  # noqa


def generate_pmrem(env_texture):
    size = env_texture.size[0]

    cube_texture = gfx.Texture(
        dim=2,
        size=(size, size, 6),
        format="rgba16float",
        generate_mipmaps=True,
        usage=wgpu.TextureUsage.COPY_DST | wgpu.TextureUsage.COPY_SRC,
    )

    cube_camera = CubeCamera(cube_texture)
    device = cube_camera.renderer.device
    scene = gfx.Scene()
    background = gfx.Background(None, gfx.BackgroundSkyboxMaterial(map=env_texture))
    scene.add(background)
    cube_camera.render(scene)

    device = get_shared().device
    # Generate mip chain
    return _generate_mip_chain(device, cube_texture)


def _generate_mip_chain(device, cube_texture):
    pipeline = device.create_compute_pipeline(
        layout="auto",
        compute={
            "module": device.create_shader_module(code=prefilter_shader),
            "entry_point": "main",
        },
    )

    params_buffer = device.create_buffer(
        size=8, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST
    )

    sampler = device.create_sampler(
        mag_filter="linear",
        min_filter="linear",
        mipmap_filter="linear",
        address_mode_u="clamp-to-edge",
        address_mode_v="clamp-to-edge",
    )

    size = cube_texture.size[0]

    cube_texture_gpu = cube_texture._wgpu_object

    num_mip_levels = int(np.log2(size)) + 1

    # Create a temp texture
    pmrem_texture_gpu = device.create_texture(
        size=(size, size, 6),
        mip_level_count=num_mip_levels,
        sample_count=1,
        dimension=wgpu.TextureDimension.d2,
        format=wgpu.TextureFormat.rgba16float,
        usage=wgpu.TextureUsage.TEXTURE_BINDING
        | wgpu.TextureUsage.STORAGE_BINDING
        | wgpu.TextureUsage.COPY_SRC,
    )

    bind_group_0 = device.create_bind_group(
        layout=pipeline.get_bind_group_layout(0),
        entries=[
            {
                "binding": 0,
                "resource": cube_texture_gpu.create_view(
                    base_mip_level=0, mip_level_count=num_mip_levels, dimension="cube"
                ),
            },
            {"binding": 1, "resource": sampler},
            {
                "binding": 2,
                "resource": {"buffer": params_buffer, "offset": 0, "size": 8},
            },
        ],
    )

    for mip in range(0, num_mip_levels):
        roughness = mip / (num_mip_levels - 1)
        mip_size = size >> mip
        pmrem_view = pmrem_texture_gpu.create_view(
            base_mip_level=mip, mip_level_count=1, dimension="2d-array"
        )
        device.queue.write_buffer(
            params_buffer, 0, np.array([roughness, mip_size], dtype=np.float32)
        )

        bind_group_1 = device.create_bind_group(
            layout=pipeline.get_bind_group_layout(1),
            entries=[
                {"binding": 0, "resource": pmrem_view},
            ],
        )

        encoder = device.create_command_encoder()
        pass_ = encoder.begin_compute_pass()
        pass_.set_pipeline(pipeline)
        pass_.set_bind_group(0, bind_group_0)
        pass_.set_bind_group(1, bind_group_1)
        pass_.dispatch_workgroups(mip_size // 8, mip_size // 8, 6)
        pass_.end()

        device.queue.submit([encoder.finish()])

    pmrem_texture = gfx.Texture(
        format="rgba16float",
        dim=2,
        size=(size, size, 6),
        generate_mipmaps=True,
        colorspace="physical",
        usage=wgpu.TextureUsage.TEXTURE_BINDING,
    )
    pmrem_texture._wgpu_object = pmrem_texture_gpu
    pmrem_texture._wgpu_mip_level_count = num_mip_levels

    return pmrem_texture, cube_texture
