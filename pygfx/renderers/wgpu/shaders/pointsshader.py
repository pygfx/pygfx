import wgpu  # only for flags/enums

from ....objects import Points
from ....materials import (
    PointsMaterial,
    PointsGaussianBlobMaterial,
    PointsSpriteMaterial,
)

from .. import (
    register_wgpu_render_function,
    WorldObjectShader,
    Binding,
    RenderMask,
    load_wgsl,
    nchannels_from_format,
)


@register_wgpu_render_function(Points, PointsMaterial)
class PointsShader(WorldObjectShader):

    type = "render"

    def __init__(self, wobject):
        super().__init__(wobject)
        material = wobject.material
        geometry = wobject.geometry

        color_mode = str(material.color_mode).split(".")[-1]
        if isinstance(material, PointsSpriteMaterial):
            self["color_mode"] = "sprite"
        elif color_mode == "auto":
            if material.map is not None:
                self["color_mode"] = "vertex_map"
                self["color_buffer_channels"] = 0
            else:
                self["color_mode"] = "uniform"
                self["color_buffer_channels"] = 0
        elif color_mode == "uniform":
            self["color_mode"] = "uniform"
            self["color_buffer_channels"] = 0
        elif color_mode == "vertex":
            nchannels = nchannels_from_format(geometry.colors.format)
            self["color_mode"] = "vertex"
            self["color_buffer_channels"] = nchannels
            if nchannels not in (1, 2, 3, 4):
                raise ValueError(f"Geometry.colors needs 1-4 columns, not {nchannels}")
        elif color_mode == "vertex_map":
            self["color_mode"] = "vertex_map"
            self["color_buffer_channels"] = 0
            if material.map is None:
                raise ValueError(f"Cannot apply colormap is no material.map is set.")
        else:
            raise RuntimeError(f"Unknown color_mode: '{color_mode}'")

        self["size_mode"] = str(material.size_mode).split(".")[-1]
        self["size_space"] = material.size_space
        self["aa"] = material.aa

    def get_bindings(self, wobject, shared):
        geometry = wobject.geometry
        material = wobject.material

        rbuffer = "buffer/read_only_storage"
        bindings = [
            Binding("u_stdinfo", "buffer/uniform", shared.uniform_buffer),
            Binding("u_wobject", "buffer/uniform", wobject.uniform_buffer),
            Binding("u_material", "buffer/uniform", material.uniform_buffer),
            Binding("s_positions", rbuffer, geometry.positions, "VERTEX"),
        ]

        if self["size_mode"] == "vertex":
            bindings.append(Binding("s_sizes", rbuffer, geometry.sizes, "VERTEX"))

        # Per-vertex color, colormap, or a uniform color?
        if self["color_mode"] == "sprite":
            self["img_nchannels"] = 2
            bindings.extend(
                self.define_img_colormap(material.map, material.map_interpolation)
            )
        elif self["color_mode"] == "vertex":
            bindings.append(Binding("s_colors", rbuffer, geometry.colors, "VERTEX"))
        elif self["color_mode"] == "vertex_map":
            bindings.extend(
                self.define_texcoords_and_colormap(
                    material.map, geometry.texcoords, material.map_interpolation
                )
            )

        self["shape"] = "circle"
        if isinstance(material, PointsGaussianBlobMaterial):
            self["shape"] = "gaussian"

        bindings = {i: b for i, b in enumerate(bindings)}
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
        material = wobject.material

        offset, size = wobject.geometry.positions.draw_range
        offset, size = offset * 6, size * 6

        render_mask = wobject.render_mask
        if not render_mask:
            if material.is_transparent:
                render_mask = RenderMask.transparent
            elif self["color_mode"] == "uniform":
                if material.color_is_transparent:
                    render_mask = RenderMask.transparent
                else:
                    render_mask = RenderMask.all
            else:
                render_mask = RenderMask.all

        return {
            "indices": (size, 1, offset, 0),
            "render_mask": render_mask,
        }

    def get_code(self):
        return self.code_definitions() + self.code_common() + load_wgsl("points.wgsl")
