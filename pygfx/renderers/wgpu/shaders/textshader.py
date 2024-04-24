import wgpu  # only for flags/enums

from ....objects import Text
from ....materials import TextMaterial
from ....utils.text._shaper import REF_GLYPH_SIZE

from .. import (
    register_wgpu_render_function,
    load_wgsl,
    BaseShader,
    Binding,
    RenderMask,
    GfxSampler,
    GfxTextureView,
)


@register_wgpu_render_function(Text, TextMaterial)
class TextShader(BaseShader):
    type = "render"

    def __init__(self, wobject):
        super().__init__(wobject)
        geometry = wobject.geometry
        material = wobject.material
        self["screen_space"] = geometry.screen_space
        self["aa"] = material.aa
        self["REF_GLYPH_SIZE"] = REF_GLYPH_SIZE

    def get_bindings(self, wobject, shared):
        geometry = wobject.geometry
        material = wobject.material

        sbuffer = "buffer/read_only_storage"
        bindings = [
            Binding("u_stdinfo", "buffer/uniform", shared.uniform_buffer),
            Binding("u_wobject", "buffer/uniform", wobject.uniform_buffer),
            Binding("u_material", "buffer/uniform", material.uniform_buffer),
            Binding("s_indices", sbuffer, geometry.indices, "VERTEX"),
            Binding("s_positions", sbuffer, geometry.positions, "VERTEX"),
            Binding("s_sizes", sbuffer, geometry.sizes, "VERTEX"),
        ]

        tex = shared.glyph_atlas_texture
        sampler = GfxSampler("linear", "clamp")
        tex_view = GfxTextureView(tex)
        bindings.append(Binding("s_atlas", "sampler/filtering", sampler, "FRAGMENT"))
        bindings.append(Binding("t_atlas", "texture/auto", tex_view, "FRAGMENT"))

        # Let the shader generate code for our bindings
        bindings = {i: b for i, b in enumerate(bindings)}
        self.define_bindings(0, bindings)

        bindings1 = {}
        bindings1[0] = Binding(
            "s_glyph_infos", sbuffer, shared.glyph_atlas_info_buffer, "VERTEX"
        )

        return {
            0: bindings,
            1: bindings1,
        }

    def get_pipeline_info(self, wobject, shared):
        return {
            "primitive_topology": wgpu.PrimitiveTopology.triangle_list,
            "cull_mode": wgpu.CullMode.none,
        }

    def get_render_info(self, wobject, shared):
        material = wobject.material
        n = wobject.geometry.positions.nitems * 6

        render_mask = 0
        if wobject.render_mask:
            render_mask = wobject.render_mask
        elif material.is_transparent:
            render_mask = RenderMask.transparent
        else:
            # Determine needed passes
            if material.color_is_transparent:
                render_mask |= RenderMask.transparent
            else:
                render_mask |= RenderMask.opaque
            if material.outline_color_is_transparent:
                render_mask |= RenderMask.transparent
            else:
                render_mask |= RenderMask.opaque
            if material.aa:
                render_mask |= RenderMask.transparent

        return {
            "indices": (n, 1),
            "render_mask": render_mask,
        }

    def get_code(self):
        return load_wgsl("text.wgsl")
