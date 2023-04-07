import math
import wgpu
from ...resources._texture import Texture
from ._utils import GfxTextureView


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


class MipmapsUtil:
    def __init__(self, device) -> None:
        self.device = device

        # Cache pipelines for every texture format used.
        self.pipelines = {}

    def get_mipmap_pipeline(self, format):
        pipeline = self.pipelines.get(format, None)

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
            module = self.device.create_shader_module(code=shader)

            pipeline = self.device.create_compute_pipeline(
                layout=self._create_pipeline_layout(format, textype),
                compute={
                    "module": module,
                    "entry_point": "c_main",
                },
            )

            self.pipelines[format] = pipeline

        return pipeline

    def _create_pipeline_layout(self, format, textype):
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

        bind_group_layout = self.device.create_bind_group_layout(entries=entries)
        bind_group_layouts.append(bind_group_layout)
        pipeline_layout = self.device.create_pipeline_layout(
            bind_group_layouts=bind_group_layouts
        )
        return pipeline_layout

    def generate_mipmaps(
        self,
        wgpu_texture: "wgpu.GPUTexture",
        format,
        mip_level_count,
        base_array_layer=0,
    ):
        pipeline = self.get_mipmap_pipeline(format)

        command_encoder: "wgpu.GPUCommandEncoder" = self.device.create_command_encoder()
        bind_group_layout = pipeline.get_bind_group_layout(0)

        dst_size = wgpu_texture.size[:2]
        src_view = wgpu_texture.create_view(
            base_mip_level=0,
            mip_level_count=1,
            dimension="2d",
            base_array_layer=base_array_layer,
        )

        for i in range(1, mip_level_count):
            dst_size = dst_size[0] // 2, dst_size[1] // 2

            dst_view = wgpu_texture.create_view(
                base_mip_level=i,
                mip_level_count=1,
                dimension="2d",
                base_array_layer=base_array_layer,
            )

            pass_encoder = command_encoder.begin_compute_pass()

            bind_group = self.device.create_bind_group(
                layout=bind_group_layout,
                entries=[
                    {"binding": 0, "resource": src_view},
                    {"binding": 1, "resource": dst_view},
                ],
            )

            pass_encoder.set_pipeline(pipeline)
            pass_encoder.set_bind_group(0, bind_group, [], 0, 99)
            pass_encoder.dispatch_workgroups(dst_size[0], dst_size[1])
            pass_encoder.end()

            src_view = dst_view

        self.device.queue.submit([command_encoder.finish()])


_the_mipmap_util = None


def generate_texture_mipmaps(device, target):
    # If this looks like a cube or stack, generate mipmaps for each individual layer
    if isinstance(target, Texture) and target.dim == 2 and target.size[2] > 1:
        for i in range(target.size[2]):
            generate_texture_mipmaps(
                device, GfxTextureView(target, layer_range=(i, i + 1))
            )
        return

    # Get the util
    global _the_mipmap_util
    if not _the_mipmap_util:
        _the_mipmap_util = MipmapsUtil(device)

    if isinstance(target, Texture):
        texture = target
        wgpu_texture = texture._wgpu_object
        layer = 0
    elif isinstance(target, GfxTextureView):
        view, texture = target, target.texture
        wgpu_texture = texture._wgpu_object
        layer = view.layer_range[0]

    _the_mipmap_util.generate_mipmaps(
        wgpu_texture, wgpu_texture.format, texture._wgpu_mip_level_count, layer
    )


def get_mip_level_count(texture):
    assert isinstance(texture, Texture)
    width, height, _ = texture.size
    return math.floor(math.log2(max(width, height))) + 1
