import wgpu  # only for flags/enums

from ....objects import Background
from ....materials import BackgroundMaterial, BackgroundImageMaterial
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


@register_wgpu_render_function(Background, BackgroundMaterial)
class BackgroundShader(BaseShader):
    type = "render"

    def get_bindings(self, wobject, shared):
        material = wobject.material

        bindings = {}

        # Uniforms
        bindings[0] = Binding("u_stdinfo", "buffer/uniform", shared.uniform_buffer)
        bindings[1] = Binding("u_wobject", "buffer/uniform", wobject.uniform_buffer)
        bindings[2] = Binding("u_material", "buffer/uniform", material.uniform_buffer)

        if isinstance(material, BackgroundImageMaterial) and material.map is not None:
            if not isinstance(material.map, Texture):
                raise TypeError("material.map must be a Texture")
            sampler = GfxSampler("linear", "repeat")
            # Select texture dimension
            if material.map.size[2] == 1:
                tex_view = GfxTextureView(material.map, view_dim="2d")
                self["texture_dim"] = "2d"
            elif material.map.size[2] == 6:
                tex_view = GfxTextureView(material.map, view_dim="cube")
                self["texture_dim"] = "cube"
            else:
                raise ValueError(
                    "BackgroundImageMaterial.map size must be NxMx1 or NxMx6."
                )
            bindings[3] = Binding("r_sampler", "sampler/filtering", sampler, "FRAGMENT")
            bindings[4] = Binding("r_tex", "texture/auto", tex_view, "FRAGMENT")
            # Channels
            fmt = to_texture_format(material.map.format)
            self["texture_nchannels"] = len(fmt) - len(fmt.lstrip("rgba"))
        else:
            self["texture_dim"] = ""

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
        return {
            "indices": (4, 1),
            "render_mask": RenderMask.opaque,
        }

    def get_code(self):
        return load_wgsl("background.wgsl")
