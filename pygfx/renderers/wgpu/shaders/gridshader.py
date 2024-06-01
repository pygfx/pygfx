import wgpu  # only for flags/enums

from ....objects import Grid
from ....materials import GridMaterial

from .. import (
    register_wgpu_render_function,
    BaseShader,
    Binding,
    RenderMask,
    load_wgsl,
)


@register_wgpu_render_function(Grid, GridMaterial)
class GridShader(BaseShader):
    type = "render"

    def __init__(self, wobject):
        super().__init__(wobject)
        material = wobject.material

        self["thickness_space"] = material.thickness_space
        self["inf_grid"] = material.infinite
        self["draw_axis"] = material._gfx_draw_axis
        self["draw_major"] = material._gfx_draw_major
        self["draw_minor"] = material._gfx_draw_minor

    def get_bindings(self, wobject, shared):
        material = wobject.material

        bindings = {}
        bindings[0] = Binding("u_stdinfo", "buffer/uniform", shared.uniform_buffer)
        bindings[1] = Binding("u_wobject", "buffer/uniform", wobject.uniform_buffer)
        bindings[2] = Binding("u_material", "buffer/uniform", material.uniform_buffer)
        self.define_bindings(0, bindings)

        return {
            0: bindings,
        }

    def get_pipeline_info(self, wobject, shared):
        return {
            "primitive_topology": wgpu.PrimitiveTopology.triangle_list,
            "cull_mode": wgpu.CullMode.none,
        }

    def get_render_info(self, wobject, shared):
        nverts = 30 if self["inf_grid"] else 6
        return {
            "indices": (nverts, 1),
            "render_mask": RenderMask.all,
        }

    def get_code(self):
        return load_wgsl("grid.wgsl")
