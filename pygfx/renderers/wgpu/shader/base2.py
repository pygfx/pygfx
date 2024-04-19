from ....resources import Buffer, Texture

from ..engine.utils import (
    to_vertex_format,
    to_texture_format,
    GfxSampler,
    GfxTextureView,
)
from .base1 import BaseShader
from ..wgsl import load_wgsl


class WorldObjectShader(BaseShader):
    """A base shader for world objects. Must be subclassed to implement
    a shader for a specific material. This class also implements common
    functions that can be used in all material-specific renderers.
    """

    type = "render"  # must be "compute" or "render"
    needs_bake_function = False

    def __init__(self, wobject, **kwargs):
        super().__init__(**kwargs)

        # Init values that get set when generate_wgsl() is called, using blender.get_shader_kwargs()
        self.kwargs.setdefault("write_pick", True)
        self.kwargs.setdefault("blending_code", "")

        # Init colormap values
        self.kwargs.setdefault("colormap_dim", "")
        self.kwargs.setdefault("colormap_nchannels", 1)
        self.kwargs.setdefault("colormap_format", "f32")

        # Init lighting
        self.kwargs.setdefault("lighting", "")

        # Apply_clip_planes
        self["n_clipping_planes"] = wobject.material.clipping_plane_count
        self["clipping_mode"] = wobject.material.clipping_mode

    def code_common(self):
        if self["colormap_dim"]:
            typemap = {"1d": "f32", "2d": "vec2<f32>", "3d": "vec3<f32>"}
            self.derived_kwargs["colormap_coord_type"] = typemap.get(
                self["colormap_dim"], "f32"
            )
        return load_wgsl("common.wgsl")

    # ----- What subclasses must implement

    def get_bindings(self, wobject, shared):
        """Subclasses must return a dict describing the buffers and
        textures used by this shader.

        The result must be a dict of dicts with binding objects
        (group_slot -> binding_slot -> binding)
        """
        return {
            0: {},
        }

    def get_pipeline_info(self, wobject, shared):
        """Subclasses must return a dict describing pipeline details.

        Fields for a compute shader: empty

        Fields for a render shader:
          * "cull_mode"
          * "primitive_topology"
        """
        return {
            "primitive_topology": 0,
            "cull_mode": 0,
        }

    def get_render_info(self, wobject, shared):
        """Subclasses must return a dict describing render details.

        Fields for a compute shader:
          * "indices" (3 ints)

        Fields for a render shader:
          * "render_mask"
          * "indices" (list of 2 or 4 ints).
        """
        return {
            "indices": (1, 1),
            "render_mask": 0,
        }

    # ----- Colormap stuff

    def define_texcoords_and_colormap(self, texture, texcoords, interpolation="linear"):
        """Define the given texture as the colormap to be used to
        lookup the final color from the (per-vertex or per-face) texcoords.
        In the WGSL the colormap can be sampled using ``sample_colormap()``.
        Returns a list of bindings.
        """
        # TODO: this is an indication that Binding needs its own module. See similar case further down
        from ..engine.pipeline import Binding  # avoid recursive import

        sampler = GfxSampler(interpolation, "repeat")
        self["colormap_interpolation"] = interpolation

        if not isinstance(texture, Texture):
            raise TypeError("texture must be a Texture")
        elif not isinstance(texcoords, Buffer):
            raise ValueError("texture is present, but texcoords must be a buffer")
        texture_view = GfxTextureView(texture)
        # Dimensionality
        self["colormap_dim"] = view_dim = texture_view.view_dim
        if view_dim not in ("1d", "2d", "3d"):
            raise ValueError(f"Unexpected texture dimension: '{view_dim}'")
        # Texture dim matches texcoords
        vert_fmt = to_vertex_format(texcoords.format)
        if view_dim == "1d" and "x" not in vert_fmt:
            pass
        elif not vert_fmt.endswith("x" + view_dim[0]):
            raise ValueError(
                f"texcoords {texcoords.format} does not match texture_view {view_dim}"
            )
        # Sampling type
        fmt = to_texture_format(texture_view.format)
        if "norm" in fmt or "float" in fmt:
            self["colormap_format"] = "f32"
        elif "uint" in fmt:
            self["colormap_format"] = "u32"
        else:
            self["colormap_format"] = "i32"
        # Channels
        self["colormap_nchannels"] = len(fmt) - len(fmt.lstrip("rgba"))
        # Return bindings
        return [
            Binding("s_colormap", "sampler/filtering", sampler, "FRAGMENT"),
            Binding("t_colormap", "texture/auto", texture_view, "FRAGMENT"),
            Binding("s_texcoords", "buffer/read_only_storage", texcoords, "VERTEX"),
        ]

    def define_img_colormap(self, texture, interpolation="linear"):
        """Define the given texture view as the colormap to be used to
        lookup the final color from the image data.
        In the WGSL the colormap can be sampled using ``sample_colormap()``.
        Returns a list of bindings.
        """
        from ..engine.pipeline import Binding  # avoid recursive import

        sampler = GfxSampler(interpolation, "clamp")
        self["colormap_interpolation"] = interpolation

        if not isinstance(texture, Texture):
            raise TypeError("texture must be a Texture")
        texture_view = GfxTextureView(texture)
        # Dimensionality
        self["colormap_dim"] = view_dim = texture_view.view_dim
        if texture_view.view_dim not in ("1d", "2d", "3d"):
            raise ValueError("Unexpected colormap texture dimension")
        # Texture dim matches image channels
        if int(view_dim[0]) != self["img_nchannels"]:
            raise ValueError(
                f"Image channels {self['img_nchannels']} does not match texture_view {view_dim}"
            )
        # Sampling type
        fmt = to_texture_format(texture_view.format)
        if "norm" in fmt or "float" in fmt:
            self["colormap_format"] = "f32"
        elif "uint" in fmt:
            self["colormap_format"] = "u32"
        else:
            self["colormap_format"] = "i32"
        # Channels
        self["colormap_nchannels"] = len(fmt) - len(fmt.lstrip("rgba"))
        # Return bindings
        return [
            Binding("s_colormap", "sampler/filtering", sampler, "FRAGMENT"),
            Binding("t_colormap", "texture/auto", texture_view, "FRAGMENT"),
        ]
