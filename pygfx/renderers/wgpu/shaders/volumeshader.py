import wgpu  # only for flags/enums

from ....objects import Volume
from ....materials import VolumeSliceMaterial, VolumeRayMaterial
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


class BaseVolumeShader(BaseShader):
    def get_bindings(self, wobject, shared):
        geometry = wobject.geometry
        material = wobject.material  # noqa

        bindings = [
            Binding("u_stdinfo", "buffer/uniform", shared.uniform_buffer),
            Binding("u_wobject", "buffer/uniform", wobject.uniform_buffer),
            Binding("u_material", "buffer/uniform", material.uniform_buffer),
        ]

        # Collect texture and sampler
        if geometry.grid is None:
            raise ValueError("Volume.geometry must have a grid (texture).")
        if not isinstance(geometry.grid, Texture):
            raise TypeError("Volume.geometry.grid must be a Texture")
        if geometry.grid.dim != 3:
            raise TypeError("Volume.geometry.grid must a 3D texture (view)")

        tex_view = GfxTextureView(geometry.grid)
        sampler = GfxSampler(material.interpolation, "clamp")
        self["colorspace"] = geometry.grid.colorspace

        # Sampling type
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

        # Channels
        self["img_nchannels"] = len(fmt) - len(fmt.lstrip("rgba"))

        bindings.append(Binding("s_img", "sampler/filtering", sampler, "FRAGMENT"))
        bindings.append(Binding("t_img", "texture/auto", tex_view, vertex_and_fragment))

        # If a colormap is applied ...
        if material.map is not None:
            bindings.extend(
                self.define_img_colormap(material.map, material.map_interpolation)
            )
            self["colorspace"] = material.map.colorspace

        bindings = {i: b for i, b in enumerate(bindings)}
        self.define_bindings(0, bindings)

        return {
            0: bindings,
        }


@register_wgpu_render_function(Volume, VolumeSliceMaterial)
class VolumeSliceShader(BaseVolumeShader):
    type = "render"

    def get_pipeline_info(self, wobject, shared):
        return {
            "primitive_topology": wgpu.PrimitiveTopology.triangle_list,
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
            "indices": (12, 1),
            "render_mask": render_mask,
        }

    def get_code(self):
        return load_wgsl("volume_slice.wgsl")


@register_wgpu_render_function(Volume, VolumeRayMaterial)
class VolumeRayShader(BaseVolumeShader):
    type = "render"

    def get_bindings(self, wobject, shared):
        self["mode"] = wobject.material.render_mode
        return super().get_bindings(wobject, shared)

    def get_pipeline_info(self, wobject, shared):
        return {
            "primitive_topology": wgpu.PrimitiveTopology.triangle_list,
            "cull_mode": wgpu.CullMode.front,  # the back planes are the ref
        }

    def get_render_info(self, wobject, shared):
        material = wobject.material

        render_mask = wobject.render_mask
        if not render_mask:
            if material.is_transparent:
                render_mask = RenderMask.transparent
            elif self["img_nchannels"] in (1, 3):
                render_mask = RenderMask.opaque
            else:
                render_mask = RenderMask.all

        return {
            "indices": (36, 1),
            "render_mask": render_mask,
        }

    def get_code(self):
        return load_wgsl("volume_ray.wgsl")
