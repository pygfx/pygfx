import wgpu  # only for flags/enums
import pyshader
from pyshader import python2shader
from pyshader import f32, i32, ivec2, vec2, vec4, Struct

from ...utils import array_from_shadertype


class RenderTexture:
    """Class used internally to store a render texture and meta data."""

    def __init__(self, format):
        self.format = format
        self.texture = None
        self.texture_view = None
        self.size = (0, 0, 0)

    def set_texture_view(self, texture_view):
        """Set from a texture view. Intended when the texture comes
        from elsewhere. Make sure it matches the format!
        """
        self.texture = None
        self.texture_view = texture_view
        self.size = self.texture_view.size

    def ensure_size(self, device, size):
        """Make sure that the texture has the given size. If necessary,
        recreates the texture and texture-view objects.
        """
        if size != self.size:
            self.size = size
            usage = wgpu.TextureUsage.OUTPUT_ATTACHMENT | wgpu.TextureUsage.COPY_SRC
            if self.format.startswith(("rgb", "bgr")):
                usage |= wgpu.TextureUsage.SAMPLED
            self.texture = device.create_texture(
                size=size, usage=usage, dimension="2d", format=self.format
            )
            self.texture_view = self.texture.create_view()

    @property
    def bytes_per_pixel(self):
        """The number of bytes per pixel."""
        format_map = {
            "depth24plus_stencil8": 4,
            "depth24plus": 3,  # ?
            "depth32": 4,
            "r8": 1,
            "r16": 2,
            "r32": 4,
            "rg8": 2,
            "rg16": 4,
            "rg32": 8,
            "rgba8": 4,
            "rgba16": 8,
            "rgba32": 16,
        }
        for key, val in format_map.items():
            if self.format.startswith(key):
                return val
        else:
            raise ValueError(f"Could not determine bbp of {self.format}")


class PostProcessingStep:
    """
    Utility object to render one texture to another, as a post-processing step.
    """

    def __init__(self, device, *, vertex_shader=None, fragment_shader, uniform_type):
        self._device = device
        self._vertex_shader = (
            default_vertex_shader if vertex_shader is None else vertex_shader
        )
        self._fragment_shader = fragment_shader
        self._hash = None

        self._uniform_type = uniform_type
        self._uniform_data = array_from_shadertype(self._uniform_type)
        self._uniform_buffer = self._device.create_buffer(
            size=self._uniform_data.nbytes,
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
        )

        self._sampler = self._device.create_sampler(
            label="render sampler",
            mag_filter="nearest",
            min_filter="nearest",
        )

    def render(self, src, dst):
        """Render one texture to another. If the src.texture_view and the dst.format
        have not changed, the pipeline is re-used.
        """
        assert isinstance(src, RenderTexture)
        assert isinstance(dst, RenderTexture)
        device = self._device

        # Recreate pipeline?
        hash = src.size, id(src.texture_view), dst.format
        if hash != self._hash:
            self._hash = hash
            self._create_pipeline(
                self._vertex_shader, self._fragment_shader, src.texture_view, dst.format
            )

        command_encoder = device.create_command_encoder()

        # Copy data to tmp buffer
        tmp_buffer = device.create_buffer_with_data(
            data=self._uniform_data,
            usage=wgpu.BufferUsage.COPY_SRC,
        )
        command_encoder.copy_buffer_to_buffer(
            tmp_buffer, 0, self._uniform_buffer, 0, self._uniform_data.nbytes
        )

        render_pass = command_encoder.begin_render_pass(
            color_attachments=[
                {
                    "attachment": dst.texture_view,
                    "resolve_target": None,
                    "load_value": (0, 0, 0, 0),  # LoadOp.load or color
                    "store_op": wgpu.StoreOp.store,
                }
            ],
            depth_stencil_attachment=None,
            occlusion_query_set=None,
        )
        render_pass.set_pipeline(self._render_pipeline)
        render_pass.set_bind_group(0, self._bind_group, [], 0, 99)
        render_pass.draw(4, 1)
        render_pass.end_pass()
        device.default_queue.submit([command_encoder.finish()])

    def _create_pipeline(
        self, vertex_shader, fragment_shader, src_texture_view, dst_format
    ):
        device = self._device

        vs_module = device.create_shader_module(code=vertex_shader)
        fs_module = device.create_shader_module(code=fragment_shader)

        binding_layouts = [
            {
                "binding": 0,
                "visibility": wgpu.ShaderStage.FRAGMENT,
                "type": wgpu.BindingType.sampler,
            },
            {
                "binding": 1,
                "visibility": wgpu.ShaderStage.FRAGMENT,
                "type": wgpu.BindingType.sampled_texture,
                "view_dimension": wgpu.TextureViewDimension.d2,
                "texture_component_type": wgpu.TextureComponentType.float,
                "multisampled": False,
            },
            {
                "binding": 2,
                "visibility": wgpu.ShaderStage.FRAGMENT,
                "type": wgpu.BindingType.uniform_buffer,
            },
        ]
        bindings = [
            {"binding": 0, "resource": self._sampler},
            {"binding": 1, "resource": src_texture_view},
            {
                "binding": 2,
                "resource": {
                    "buffer": self._uniform_buffer,
                    "offset": 0,
                    "size": self._uniform_data.nbytes,
                },
            },
        ]

        bind_group_layout = device.create_bind_group_layout(entries=binding_layouts)
        pipeline_layout = device.create_pipeline_layout(
            bind_group_layouts=[bind_group_layout]
        )
        self._bind_group = device.create_bind_group(
            layout=bind_group_layout, entries=bindings
        )

        self._render_pipeline = device.create_render_pipeline(
            layout=pipeline_layout,
            vertex_stage={"module": vs_module, "entry_point": "main"},
            fragment_stage={"module": fs_module, "entry_point": "main"},
            primitive_topology=wgpu.PrimitiveTopology.triangle_strip,
            color_states=[
                {
                    "format": dst_format,
                    "alpha_blend": (
                        wgpu.BlendFactor.one,
                        wgpu.BlendFactor.zero,
                        wgpu.BlendOperation.add,
                    ),
                    "color_blend": (
                        wgpu.BlendFactor.src_alpha,
                        wgpu.BlendFactor.one_minus_src_alpha,
                        wgpu.BlendOperation.add,
                    ),
                }
            ],
            vertex_state={
                "index_format": wgpu.IndexFormat.uint32,
                "vertex_buffers": [],
            },
        )


class SSAAPostProcessingStep(PostProcessingStep):
    """A texture renderer that supports SSAA."""

    def __init__(self, device):
        super().__init__(
            device, fragment_shader=ssaa_fragment_shader, uniform_type=ssaa_uniform_type
        )

    def render(self, src, dst):

        # Get factor between texture sizes
        factor_x = src.size[0] / dst.size[0]
        factor_y = src.size[1] / dst.size[1]
        factor = (factor_x + factor_y) / 2

        if factor > 1:
            # src has higher res, we can do ssaa
            sigma = 0.5 * factor
            support = min(5, int(sigma * 3))
        else:
            # src has lower res. smooth those pixels a bit
            sigma = 1
            support = 2

        self._uniform_data["size"] = src.size[:2]
        self._uniform_data["sigma"] = sigma
        self._uniform_data["support"] = support

        super().render(src, dst)


# %% Shaders

ssaa_uniform_type = Struct(size=vec2, sigma=f32, support=i32)


@python2shader
def default_vertex_shader(
    index: (pyshader.RES_INPUT, "VertexId", "i32"),
    out_pos: (pyshader.RES_OUTPUT, "Position", vec4),
    v_texcoord: (pyshader.RES_OUTPUT, 0, vec2),
):
    positions = [vec2(0, 1), vec2(0, 0), vec2(1, 1), vec2(1, 0)]
    pos = positions[index]
    v_texcoord = vec2(pos.x, 1.0 - pos.y)  # noqa - shader output
    out_pos = vec4(pos * 2.0 - 1.0, 0.0, 1.0)  # noqa - shader output


@python2shader
def ssaa_fragment_shader(
    v_texcoord: (pyshader.RES_INPUT, 0, vec2),
    s_sam: (pyshader.RES_SAMPLER, (0, 0), ""),
    t_tex: (pyshader.RES_TEXTURE, (0, 1), "2d f32"),
    u_render: (pyshader.RES_UNIFORM, (0, 2), ssaa_uniform_type),
    out_color: (pyshader.RES_OUTPUT, 0, vec4),
):
    # Get info about the smoothing
    sigma = u_render.sigma
    support = min(5, u_render.support)

    # Determine distance between pixels in src texture
    step = vec2(1.0 / u_render.size.x, 1.0 / u_render.size.y)
    # Get texcoord, and round it to the center of the source pixels.
    # Thus, whether the sampler is linear or nearest, we get equal results.
    ori_coord = v_texcoord.xy
    ref_coord = vec2(ivec2(ori_coord / step)) * step + 0.5 * step

    # Convolve. Here we apply a Gaussian kernel, the weight is calculated
    # for each pixel individually based on the distance to the actual texture
    # coordinate. This means that the textures don't even need to align.
    val = vec4(0.0, 0.0, 0.0, 0.0)
    weight = 0.0
    for y in range(-support, support + 1):
        for x in range(-support, support + 1):
            coord = ref_coord + vec2(x, y) * step
            distance = length((ori_coord - coord) / step)  # in src pixels
            t = distance / sigma
            w = exp(-0.5 * t * t)
            val += t_tex.sample(s_sam, coord) * w
            weight += w

    out_color = val / weight  # noqa - shader output
