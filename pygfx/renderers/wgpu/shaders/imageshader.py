import wgpu  # only for flags/enums

from ....objects import Image
from ....materials import ImageBasicMaterial
from ....resources import Texture

from .. import (
    register_wgpu_render_function,
    BaseShader,
    Binding,
    RenderMask,
    to_texture_format,
    GfxSampler,
    GfxTextureView,
    load_wgsl,
)


vertex_and_fragment = wgpu.ShaderStage.VERTEX | wgpu.ShaderStage.FRAGMENT


@register_wgpu_render_function(Image, ImageBasicMaterial)
class ImageShader(BaseShader):
    type = "render"

    def __init__(self, wobject):
        super().__init__(wobject)
        material = wobject.material
        geometry = wobject.geometry

        # In this method we want to set all shader variables (i.e. self['xx']),
        # so that we can properly detect when stuff changes that (may) affect
        # these values, and trigger a recompile.

        # Check grid
        if geometry.grid is None:
            raise ValueError("Image.geometry must have a grid (texture).")
        elif not isinstance(geometry.grid, Texture):
            raise TypeError("Image.geometry.grid must be a Texture.")
        elif geometry.grid.dim != 2:
            raise TypeError("Image.geometry.grid must a 2D texture")

        # Set img_format and climcorrection
        self["climcorrection"] = ""
        fmt = to_texture_format(geometry.grid.format)
        if "norm" in fmt or "float" in fmt:
            self["img_format"] = "f32"
            if "unorm" in fmt:
                self["climcorrection"] = " * 255.0"
            elif "snorm" in fmt:
                self["climcorrection"] = " * 255.0 - 128.0"
        elif "uint" in fmt:
            self["img_format"] = "u32"
        else:
            self["img_format"] = "i32"

        # Set gamma
        self["gamma"] = material.gamma

        # Channels
        self["img_nchannels"] = len(fmt) - len(fmt.lstrip("rgba"))

        # Determine colorspace
        self["colorspace"] = geometry.grid.colorspace
        self["colorrange"] = geometry.grid.colorrange
        self["three_grid_yuv"] = (
            self["colorspace"] in ["yuv420p", "yuv444p"] and geometry.grid.size[2] == 1
        )
        self["use_colormap"] = False

        if geometry.grid.colorspace not in ("srgb", "physical"):
            # A special color-space that we convert to rgb in the shader
            self["img_nchannels"] = 3
        elif material.map is not None:
            self["use_colormap"] = True
            self["colorspace"] = material.map.texture.colorspace

    def get_bindings(self, wobject, shared):
        geometry = wobject.geometry
        material = wobject.material

        bindings = [
            Binding("u_stdinfo", "buffer/uniform", shared.uniform_buffer),
            Binding("u_wobject", "buffer/uniform", wobject.uniform_buffer),
            Binding("u_material", "buffer/uniform", material.uniform_buffer),
        ]

        tex_view = GfxTextureView(geometry.grid)
        sampler = GfxSampler(material.interpolation, "clamp")
        bindings.append(Binding("s_img", "sampler/filtering", sampler, "FRAGMENT"))
        bindings.append(Binding("t_img", "texture/auto", tex_view, vertex_and_fragment))

        if self["three_grid_yuv"]:
            u_tex_view = GfxTextureView(geometry.grid_u)
            v_tex_view = GfxTextureView(geometry.grid_v)
            bindings.append(
                Binding("t_u_img", "texture/auto", u_tex_view, vertex_and_fragment)
            )
            bindings.append(
                Binding("t_v_img", "texture/auto", v_tex_view, vertex_and_fragment)
            )

        if self["use_colormap"]:
            bindings.extend(self.define_img_colormap(material.map))

        bindings = {i: b for i, b in enumerate(bindings)}
        self.define_bindings(0, bindings)

        return {
            0: bindings,
        }

    def get_pipeline_info(self, wobject, shared):
        return {
            "primitive_topology": wgpu.PrimitiveTopology.triangle_strip,
            "cull_mode": wgpu.CullMode.none,
        }

    def get_render_info(self, wobject, shared):
        material = wobject.material

        render_mask = 0
        if wobject.render_mask:
            render_mask = wobject.render_mask
        elif material.is_transparent:
            render_mask = RenderMask.transparent
        else:
            # Determine what passes are needed
            if material.map is not None:
                if self["colormap_nchannels"] in (1, 3):
                    render_mask |= RenderMask.opaque
                else:
                    render_mask |= RenderMask.all
            else:
                if self["img_nchannels"] in (1, 3):
                    render_mask |= RenderMask.opaque
                else:
                    render_mask |= RenderMask.all

        return {
            "indices": (4, 1),
            "render_mask": render_mask,
        }

    def get_code(self):
        return load_wgsl("image.wgsl")
