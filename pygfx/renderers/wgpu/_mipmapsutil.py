import math
import wgpu
from ...resources._texture import Texture
from ._utils import GfxTextureView


mipmap_source = """
    struct VaryingsStruct {
        @builtin( position ) position: vec4<f32>,
        @location( 0 ) texcoord : vec2<f32>
    };

    @group(0) @binding(0)
    var t_img : texture_2d<f32>;

    @vertex
    fn v_main( @builtin( vertex_index ) vertexIndex : u32 ) -> VaryingsStruct {
        var varyings : VaryingsStruct;
        var positions = array< vec2<f32>, 4 >(
            vec2<f32>( -1.0,  1.0 ),
            vec2<f32>(  1.0,  1.0 ),
            vec2<f32>( -1.0, -1.0 ),
            vec2<f32>(  1.0, -1.0 )
        );
        var texcoords = array< vec2<f32>, 4 >(
            vec2<f32>( 0.0, 0.0 ),
            vec2<f32>( 1.0, 0.0 ),
            vec2<f32>( 0.0, 1.0 ),
            vec2<f32>( 1.0, 1.0 )
        );
        varyings.texcoord = texcoords[ vertexIndex ];
        varyings.position = vec4<f32>( positions[ vertexIndex ], 0.0, 1.0 );
        return varyings;
    }

    @fragment
    fn f_main( @location(0) texcoord : vec2<f32> ) -> @location(0) vec4<f32> {

        let texSize = vec2<f32>(textureDimensions(t_img));

        // Determine smoothing settings.
        // Sigma 1.0 matches a factor 2 size reduction.
        // support would ideally be 3*sigma, but we can use less to trade performance
        let sigma = 1.0;
        let support = 2;  // 2: 5x5 kernel, 3: 7x7 kernel

        // The reference index is the subpixel index in the source texture that
        // represents the location of this fragment.
        let ref_index = texcoord * texSize;

        // For the sampling, we work with integer coords. Also use min/max for the edges.
        let base_index = vec2<i32>(ref_index);
        let min_index = vec2<i32>(0, 0);
        let max_index = vec2<i32>(texSize - 1.0);

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
                val = val + textureLoad(t_img, index, 0) * w;
                weight = weight + w;
            }
        }

        return vec4<f32>(val.rgba / weight);
    }
"""


class MipmapsUtil:
    def __init__(self, device) -> None:
        self.device = device

        # Cache pipelines for every texture format used.
        self.pipelines = {}

        self.mipmap_shader_module = self.device.create_shader_module(code=mipmap_source)

    def get_mipmap_pipeline(self, format) -> "wgpu.GPURenderPipeline":
        pipeline = self.pipelines.get(format, None)

        if pipeline is None:
            pipeline = self.device.create_render_pipeline(
                layout=self._create_pipeline_layout(),
                vertex={
                    "module": self.mipmap_shader_module,
                    "entry_point": "v_main",
                    "buffers": [],
                },
                fragment={
                    "module": self.mipmap_shader_module,
                    "entry_point": "f_main",
                    "targets": [{"format": format}],
                },
                primitive={
                    "topology": "triangle-strip",
                    "strip_index_format": "uint32",
                },
            )

            self.pipelines[format] = pipeline

        return pipeline

    def _create_pipeline_layout(self):
        bind_group_layouts = []

        entries = []
        entries.append(
            {
                "binding": 0,
                "visibility": wgpu.ShaderStage.FRAGMENT,
                "texture": {"multisampled": False},
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

        src_view = wgpu_texture.create_view(
            base_mip_level=0,
            mip_level_count=1,
            dimension="2d",
            base_array_layer=base_array_layer,
        )

        for i in range(1, mip_level_count):
            dst_view = wgpu_texture.create_view(
                base_mip_level=i,
                mip_level_count=1,
                dimension="2d",
                base_array_layer=base_array_layer,
            )

            pass_encoder: "wgpu.GPURenderPassEncoder" = (
                command_encoder.begin_render_pass(
                    color_attachments=[
                        {
                            "view": dst_view,
                            "load_op": wgpu.LoadOp.clear,
                            "store_op": wgpu.StoreOp.store,
                            "clear_value": [0, 0, 0, 0],
                        }
                    ]
                )
            )

            bind_group = self.device.create_bind_group(
                layout=bind_group_layout,
                entries=[
                    {"binding": 0, "resource": src_view},
                ],
            )

            pass_encoder.set_pipeline(pipeline)
            pass_encoder.set_bind_group(0, bind_group, [], 0, 99)
            pass_encoder.draw(4, 1, 0, 0)
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
