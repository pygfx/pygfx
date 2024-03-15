import numpy as np
import wgpu

from ....resources._texture import Texture

from .utils import GfxTextureView, GpuCache
from .shared import get_shared


# This cache enables re-using gpu pipelines for calculating mipmaps, so
# that these don't have to be created on each draw, which would make
# things very slow. The number of pipelines in the cache won't be large
# since there are only so many texture formats, but it seems cleaner
# to use our cache mechanism than to store them globally.
MIPMAP_CACHE = GpuCache("mipmap_pipelines")


mipmap_source = """

    @group(0) @binding(0)
    var t_img1 : texture_2d<TYPE>;

    @group(0) @binding(1)
    var t_img2 : texture_storage_2d<FORMAT, write>;

    @compute
    @workgroup_size(1)
    fn c_main(@builtin(global_invocation_id) index2: vec3<u32>) {

        let texSize1 = vec2<f32>(textureDimensions(t_img1));

        // Determine smoothing settings.
        // Sigma 1.0 matches a factor 2 size reduction.
        // support would ideally be 3*sigma, but we can use less to trade performance
        let sigma = 1.0;
        let support = 2;  // 2: 5x5 kernel, 3: 7x7 kernel

        // For the sampling, we work with integer coords. Also use min/max for the edges.
        let ref_index = vec2<f32>(f32(i32(index2.x) * 2), f32(i32(index2.y) * 2));
        let base_index = vec2<i32>(ref_index);
        let min_index = vec2<i32>(0, 0);
        let max_index = vec2<i32>(texSize1 - 1.0);

        // Convolve. Here we apply a Gaussian kernel, the weight is calculated
        // for each pixel individually based on the distance to the ref_index.
        // This means that the textures don't need to align.
        var val: vec4<f32> = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        var weight: f32 = 0.0;
        for (var y:i32 = -support; y <= support; y = y + 1) {
            for (var x:i32 = -support; x <= support; x = x + 1) {
                let step = vec2<i32>(x, y);
                let index = clamp(base_index + step, min_index, max_index);
                let dist = length(ref_index - vec2<f32>(index) - 0.5);
                let t = dist / sigma;
                let w = exp(-0.5 * t * t);
                let pixelVal = vec4<f32>(textureLoad(t_img1, index, 0));
                val = val + pixelVal * w;
                weight = weight + w;
            }
        }
        let value = vec4<TYPE>(val.rgba / weight);
        let outCoord = vec2<i32>(i32(index2.x), i32(index2.y));
        textureStore(t_img2, outCoord, value);
    }
"""


def get_mip_level_count(texture):
    """Get the number of mipmap levels from the texture size."""
    if not isinstance(texture, Texture):
        raise TypeError("Expecting a Texture object.")
    width, height, _ = texture.size
    return int(np.floor(np.log2(max(width, height))) + 1)


def generate_texture_mipmaps(target):
    """Generate mipmaps for the given target. The target can be a
    Texture or GfxTextureView and can be a 2D texture as well as a cube
    texture.
    """

    # If this looks like a cube or stack, generate mipmaps for each individual layer
    if isinstance(target, Texture) and target.dim == 2 and target.size[2] > 1:
        for i in range(target.size[2]):
            generate_texture_mipmaps(GfxTextureView(target, layer_range=(i, i + 1)))
        return

    if isinstance(target, Texture):
        texture = target
        layer = 0
    elif isinstance(target, GfxTextureView):
        view, texture = target, target.texture
        layer = view.layer_range[0]

    generate_mipmaps(texture, layer)


def generate_mipmaps(texture, base_array_layer):
    device = get_shared().device

    pipeline = get_mipmap_pipeline(device, texture)

    command_encoder: "wgpu.GPUCommandEncoder" = device.create_command_encoder()
    bind_group_layout = pipeline.get_bind_group_layout(0)

    dst_size = texture.size[:2]
    prev_view = None

    for i in range(1, texture._wgpu_mip_level_count):
        dst_size = dst_size[0] // 2, dst_size[1] // 2
        bind_group, prev_view = get_bind_group(
            device, texture, bind_group_layout, base_array_layer, i, prev_view
        )

        pass_encoder = command_encoder.begin_compute_pass()
        pass_encoder.set_pipeline(pipeline)
        pass_encoder.set_bind_group(0, bind_group, [], 0, 99)
        pass_encoder.dispatch_workgroups(dst_size[0], dst_size[1])
        pass_encoder.end()

    device.queue.submit([command_encoder.finish()])


def get_bind_group(
    device, texture, bind_group_layout, base_array_layer, mip_level, src_view
):
    # Get the bind group that includes the two subsequent texture views.
    # We cache this on the texture object to avoid re-creating the views
    # and bind-group each time, which results in a significant increase
    # in performance.
    #
    # Note, however, that this does mean we hold on to GPU resources,
    # which may be a bit of a waste if the mipmaps are only created once!
    # So for now we're assuming that mipmaps are generated often.

    key = f"_gfx_bind_group_{base_array_layer}_{mip_level}"
    bind_group = getattr(texture, key, None)
    dst_view = None
    if bind_group is None:
        if src_view is None:
            src_view = texture._wgpu_object.create_view(
                base_mip_level=mip_level - 1,
                mip_level_count=1,
                dimension="2d",
                base_array_layer=base_array_layer,
            )
        dst_view = texture._wgpu_object.create_view(
            base_mip_level=mip_level,
            mip_level_count=1,
            dimension="2d",
            base_array_layer=base_array_layer,
        )
        bind_group = device.create_bind_group(
            layout=bind_group_layout,
            entries=[
                {"binding": 0, "resource": src_view},
                {"binding": 1, "resource": dst_view},
            ],
        )
        setattr(texture, key, bind_group)

    return bind_group, dst_view


def get_mipmap_pipeline(device, texture):
    format = texture._wgpu_object.format
    pipeline = MIPMAP_CACHE.get(format)

    if pipeline is None:
        if format.endswith("srgb"):
            raise RuntimeError("Cannot create mipmaps for srgb textures.")

        type, textype = "f32", "float"
        if format.endswith(("norm", "float")):
            type, textype = "f32", "float"
        elif format.endswith("sint"):
            type, textype = "i32", "sint"
        elif format.endswith("uint"):
            type, textype = "u32", "uint"

        shader = mipmap_source.replace("FORMAT", format).replace("TYPE", type)
        module = device.create_shader_module(code=shader)

        pipeline = device.create_compute_pipeline(
            layout=create_pipeline_layout(device, format, textype),
            compute={
                "module": module,
                "entry_point": "c_main",
            },
        )

        MIPMAP_CACHE.set(format, pipeline)

    # Store a ref of the pipeline, the cache uses weak refs.
    # I.e. the pipeline for this format is in the cache for as long as
    # any textures that use it are alive.
    texture._wgpu_mipmap_pipeline = pipeline

    return pipeline


def create_pipeline_layout(device, format, textype):
    bind_group_layouts = []

    entries = []
    entries.append(
        {
            "binding": 0,
            "visibility": wgpu.ShaderStage.COMPUTE,
            "texture": {"sample_type": textype},
        }
    )
    entries.append(
        {
            "binding": 1,
            "visibility": wgpu.ShaderStage.COMPUTE,
            "storage_texture": {
                "access": "write-only",
                "view_dimension": "2d",
                "format": format,
            },
        }
    )

    bind_group_layout = device.create_bind_group_layout(entries=entries)
    bind_group_layouts.append(bind_group_layout)
    pipeline_layout = device.create_pipeline_layout(
        bind_group_layouts=bind_group_layouts
    )
    return pipeline_layout
