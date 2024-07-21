import wgpu  # only for flags/enums

from ....resources import Texture
from ....objects import Points
from ....materials import (
    PointsMaterial,
    PointsGaussianBlobMaterial,
    PointsMarkerMaterial,
    PointsSpriteMaterial,
)

from .. import (
    register_wgpu_render_function,
    BaseShader,
    Binding,
    RenderMask,
    load_wgsl,
    nchannels_from_format,
    to_texture_format,
    GfxSampler,
    GfxTextureView,
)


@register_wgpu_render_function(Points, PointsMaterial)
class PointsShader(BaseShader):

    type = "render"

    def __init__(self, wobject):
        super().__init__(wobject)
        material = wobject.material
        geometry = wobject.geometry

        color_mode = str(material.color_mode).split(".")[-1]
        if color_mode == "auto":
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
        elif color_mode == "debug":
            self["color_mode"] = "debug"
            self["color_buffer_channels"] = 0
        else:
            raise RuntimeError(f"Unknown color_mode: '{color_mode}'")

        self["edge_mode"] = material.edge_mode
        self["is_sprite"] = 0  # 0, 1, 2
        if isinstance(material, PointsSpriteMaterial):
            self["is_sprite"] = 1
            if material.sprite is not None:
                self["is_sprite"] = 2  # i.e. is sprite and has a texture

        self["size_mode"] = str(material.size_mode).split(".")[-1]
        self["size_space"] = material.size_space
        self["aa"] = material.aa

        self["draw_line_on_edge"] = False
        if isinstance(material, PointsMarkerMaterial):
            self["draw_line_on_edge"] = True

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
        if self["color_mode"] == "vertex":
            bindings.append(Binding("s_colors", rbuffer, geometry.colors, "VERTEX"))
        elif self["color_mode"] == "vertex_map":
            bindings.append(
                Binding("s_texcoords", rbuffer, geometry.texcoords, "VERTEX")
            )
            bindings.extend(
                self.define_colormap(
                    material.map, geometry.texcoords, material.map_interpolation
                )
            )

        # Process sprite texture. Note that we can *also* have a colormap for the base color.
        if self["is_sprite"] == 2:
            sprite_sampler = GfxSampler("linear", "clamp")
            if not isinstance(material.sprite, Texture):
                raise TypeError("material sprite must be a Texture")
            sprite_view = GfxTextureView(material.sprite)
            if sprite_view.view_dim != "2d":
                raise ValueError("Sprite textures must be 2D")
            fmt = to_texture_format(sprite_view.format)
            if not ("norm" in fmt or "float" in fmt):
                raise ValueError("Sprite textures must be u8norm or float")
            self["sprite_nchannels"] = len(fmt) - len(fmt.lstrip("rgba"))
            bindings += [
                Binding("s_sprite", "sampler/filtering", sprite_sampler, "FRAGMENT"),
                Binding("t_sprite", "texture/auto", sprite_view, "FRAGMENT"),
            ]

        self["shape"] = "circle"
        if isinstance(material, PointsGaussianBlobMaterial):
            self["shape"] = "gaussian"
        elif isinstance(material, PointsMarkerMaterial):
            self["shape"] = material.marker
            custom_sdf = material.custom_sdf
            if custom_sdf is None:
                # Make a nice full square to help the user better design their
                # custom SDF
                custom_sdf = "return max(abs(coord.x), abs(coord.y)) - size * 0.5;"
            self["custom_sdf"] = custom_sdf

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

        render_mask = 0
        if wobject.render_mask:
            render_mask = wobject.render_mask
        elif material.is_transparent:
            render_mask = RenderMask.transparent
        else:
            # Get what passes are needed for the color
            if self["color_mode"] == "uniform":
                if material.color_is_transparent:
                    render_mask |= RenderMask.transparent
                else:
                    render_mask |= RenderMask.opaque
            elif self["color_mode"] == "vertex":
                if self["color_buffer_channels"] in (2, 4):
                    render_mask |= RenderMask.all
                else:
                    render_mask |= RenderMask.opaque
            elif self["color_mode"] == "vertex_map":
                if self["colormap_nchannels"] in (2, 4):
                    render_mask |= RenderMask.all
                else:
                    render_mask |= RenderMask.opaque
            elif self["color_mode"] == "debug":
                render_mask |= RenderMask.all
            else:
                raise RuntimeError(f"Unexpected color mode {self['color_mode']}")
            # Need transparency for aa
            if material.aa:
                render_mask |= RenderMask.transparent
            # More cases
            elif isinstance(material, PointsSpriteMaterial):
                if self["sprite_nchannels"] in [2, 4]:
                    render_mask |= RenderMask.transparent
                else:
                    pass  # mixed with color, so no need to OR with opaque
            elif isinstance(material, PointsMarkerMaterial):
                if material.edge_color_is_transparent:
                    render_mask |= RenderMask.transparent
                else:
                    render_mask |= RenderMask.opaque

        return {
            "indices": (size, 1, offset, 0),
            "render_mask": render_mask,
        }

    def get_code(self):
        return load_wgsl("points.wgsl")
