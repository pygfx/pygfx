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

mipmap_sampler = None

mipmap_vertex_source = """

    struct VarysStruct {
        @builtin( position ) Position: vec4<f32>,
        @location( 0 ) vTex : vec2<f32>
    };
    @vertex
    fn main( @builtin( vertex_index ) vertexIndex : u32 ) -> VarysStruct {
        var Varys : VarysStruct;
        var pos = array< vec2<f32>, 4 >(
            vec2<f32>( -1.0,  1.0 ),
            vec2<f32>(  1.0,  1.0 ),
            vec2<f32>( -1.0, -1.0 ),
            vec2<f32>(  1.0, -1.0 )
        );
        var tex = array< vec2<f32>, 4 >(
            vec2<f32>( 0.0, 0.0 ),
            vec2<f32>( 1.0, 0.0 ),
            vec2<f32>( 0.0, 1.0 ),
            vec2<f32>( 1.0, 1.0 )
        );
        Varys.vTex = tex[ vertexIndex ];
        Varys.Position = vec4<f32>( pos[ vertexIndex ], 0.0, 1.0 );
        return Varys;
    }
"""

mipmap_fragment_source = """

    @group( 0 ) @binding( 0 )
    var imgSampler : sampler;
    @group( 0 ) @binding( 1 )
    var img : texture_2d<f32>;
    @fragment
    fn main( @location( 0 ) vTex : vec2<f32> ) -> @location( 0 ) vec4<f32> {
        return textureSample( img, imgSampler, vTex );
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
        pass_encoder = command_encoder.begin_render_pass(
            color_attachments=[
                {
                    "view": prev_view,
                    "load_op": wgpu.LoadOp.clear,
                    "store_op": wgpu.StoreOp.store,
                    "clear_value": [0, 0, 0, 0],
                }
            ]
        )

        pass_encoder.set_pipeline(pipeline)
        pass_encoder.set_bind_group(0, bind_group)
        pass_encoder.draw(4, 1, 0, 0)
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
    dst_view = texture._wgpu_object.create_view(
        base_mip_level=mip_level,
        mip_level_count=1,
        dimension="2d",
        base_array_layer=base_array_layer,
    )
    if bind_group is None:
        if src_view is None:
            src_view = texture._wgpu_object.create_view(
                base_mip_level=mip_level - 1,
                mip_level_count=1,
                dimension="2d",
                base_array_layer=base_array_layer,
            )
        global mipmap_sampler
        if mipmap_sampler is None:
            mipmap_sampler = device.create_sampler(min_filter="linear")

        bind_group = device.create_bind_group(
            layout=bind_group_layout,
            entries=[
                {"binding": 0, "resource": mipmap_sampler},
                {"binding": 1, "resource": src_view},
            ],
        )
        setattr(texture, key, bind_group)

    return bind_group, dst_view


def get_mipmap_pipeline(device, texture):
    format = texture._wgpu_object.format
    pipeline = MIPMAP_CACHE.get(format)

    if pipeline is None:
        vertex_module = device.create_shader_module(code=mipmap_vertex_source)

        frag_module = device.create_shader_module(code=mipmap_fragment_source)

        pipeline = device.create_render_pipeline(
            layout="auto",
            vertex={
                "module": vertex_module,
                "entry_point": "main",
            },
            fragment={
                "module": frag_module,
                "entry_point": "main",
                "targets": [{"format": format}],
            },
            primitive={
                "topology": "triangle-strip",
                "strip_index_format": "uint32",
            },
        )

        MIPMAP_CACHE.set(format, pipeline)

    # Store a ref of the pipeline, the cache uses weak refs.
    # I.e. the pipeline for this format is in the cache for as long as
    # any textures that use it are alive.
    texture._wgpu_mipmap_pipeline = pipeline

    return pipeline
