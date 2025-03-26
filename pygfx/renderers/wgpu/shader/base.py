"""
Implements the base shader class. The shader is responsible for providing the
WGSL code, as well as providing the information to connect it to the resources
(buffers and textures) and some details on the pipeline and rendering.
"""

from ....resources import Buffer, Texture
from ..engine.utils import (
    GfxSampler,
    GfxTextureView,
    to_vertex_format,
    to_texture_format,
    hash_from_value,
)
from ..engine.binding import Binding
from .resolve import resolve_shadercode
from .bindings import BindingDefinitions
from .templating import apply_templating


class ShaderInterface:
    """Define what a shader object must look like from the pov of the pipeline."""

    def __init__(self):
        self._hash = None

    @property
    def hash(self):
        """A hash of the current state of the shader. If the hash changed, it's likely that the shader changed."""
        if self._hash is None:
            return self._get_hash()
        else:
            return self._hash

    def _get_hash(self):
        raise NotImplementedError()

    def generate_wgsl(self, **template_vars):
        raise NotImplementedError()

    def get_bindings_info(self, wobject, shared):
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


class BaseShader(ShaderInterface):
    """Base shader object to compose and template shaders using jinja2.

    Templating variables can be passed kwargs, set(and get) as attributes, or passed as kwargs to ``generate_wgsl()``.
    They should only be set in __init__() and get_bindings_info().

    The template variables affect the hash. After `pipeline.py` calls get_bindings_info(), it checks
    the hash to see if the shader has changed, and recompiles the wgsl if it has.
    """

    type = "render"  # must be "compute" or "render"
    needs_bake_function = False

    def __init__(self, wobject, **template_vars):
        super().__init__()

        # The shader templating variables
        self._template_vars_init = {}
        self._template_vars_init.update(template_vars)
        self._template_vars_bindings = {}

        # Start off with the current dict set. This is reset when get_bindings_info() gets called.
        self._template_vars_current = self._template_vars_init

        # Handling binding definitions is handled by a wrapped object.
        self._binding_definitions = BindingDefinitions()

        # Init variables that apply to all world objects. The wobject can be None, mostly for testing.
        if wobject is not None:
            self["n_clipping_planes"] = wobject.material.clipping_plane_count
            self["clipping_mode"] = wobject.material.clipping_mode

        # Init other common variables so we don't need jinja2's defined()
        self["colormap_dim"] = None

    def __setitem__(self, key, value):
        if self._template_vars_current is None:
            raise RuntimeError(
                "Attempt to set template variable outside of _init() or _get_bindings()."
            )
        self._template_vars_current[key] = value

    def __getitem__(self, key):
        try:
            return self._template_vars_bindings[key]
        except KeyError:
            return self._template_vars_init[key]

    def _get_hash(self):
        # Getting the hash also locks template var setting
        self._template_vars_current = None

        # The full name of this shader class.
        fullname = self.__class__.__module__ + "." + self.__class__.__name__

        # If we assume that the shader class produces the same code for
        # a specific set of template-vars, we can use the fullname in the hash
        # instead of the actual code. This assumption is valid in
        # general, but can break down in a few specific situations, the
        # most common one being an interactive session. In this case,
        # the fullname would be something like "__main__.CustomShader".
        # To be on the safe side, we use the full code when the fullname
        # contains only one dot. This may introduce false positives,
        # but that's fine, because this is only a performance
        # optimization.

        name_probably_defines_code = fullname.count(".") >= 2

        if name_probably_defines_code:
            # Faster, but assumes that the produced code only depends on template-vars.
            return hash_from_value(
                [
                    fullname,
                    self._binding_definitions.get_code(),
                    self._template_vars_init,
                    self._template_vars_bindings,
                ]
            )
        else:
            # More reliable (e.g. in an interactive session).
            return hash_from_value(
                [
                    self._binding_definitions.get_code(),
                    self.get_code(),
                    self._template_vars_init,
                    self._template_vars_bindings,
                ]
            )

    def get_bindings_info(self, wobject, shared):
        # We assume that in normal usage (by the pipeline.py logic),
        # this method is called first after initialization,
        # which means that we can handle the _template_vars_current here.
        self._template_vars_current = self._template_vars_bindings

        # Clear the binding-based template variables, and the binding definitions, so that when get_bindings()
        # not-sets a variable (which was previously set), this affects the hash and is detected.
        self._template_vars_bindings.clear()
        self._binding_definitions.clear()

        try:
            return self.get_bindings(wobject, shared)
        finally:
            self._template_vars_current = None
            self._hash = None  # template vars my have changed, so force a recalculation

    def get_bindings(self, wobject, shared):
        # Default implementation returns zero bindings
        return {0: {}}

    def get_code(self):
        """Implement this to compose the total (but still templated)
        shader. This method is called by ``generate_wgsl()``.
        """
        raise NotImplementedError()

    def generate_wgsl(self, **more_template_vars):
        """Generate the final WGSL. Calls get_code() and then resolves
        the templating variables, varyings, and depth output.
        """

        # Compose shader variables
        template_vars = {}
        template_vars.update(self._template_vars_init)
        template_vars.update(self._template_vars_bindings)
        template_vars.update(more_template_vars)
        template_vars["bindings_code"] = self._binding_definitions.get_code()

        # If template variables are used in ``get_code()``, make sure its correct.
        ori_template_vars, self._template_vars_bindings = (
            self._template_vars_bindings,
            template_vars,
        )

        try:
            # Obtain base code
            code1 = self.get_code()
            # Templating and resolving includes
            code2 = apply_templating(code1, **template_vars)
            # Resolve Varyings and FragmentOutput
            code3 = resolve_shadercode(code2)
            return code3
        finally:
            self._template_vars_bindings = ori_template_vars

    def define_bindings(self, bindgroup, bindings_dict):
        """Define a collection of bindings organized in a dict."""
        if self._template_vars_current is not self._template_vars_bindings:
            raise RuntimeError("define_bindings can only be used from get_bindings")

        for index, binding in bindings_dict.items():
            self._binding_definitions.define_binding(bindgroup, index, binding)

    def define_binding(self, bindgroup, index, binding):
        """Define a uniform, buffer, sampler, or texture.

        The binding must be a Binding object. The code that defines the binding
        will be inserted in ``pygfx.std.wgsl``.
        """
        if self._template_vars_current is not self._template_vars_bindings:
            raise RuntimeError("define_bindings can only be used from get_bindings")

        self._binding_definitions.define_binding(bindgroup, index, binding)

    # ----- Colormap stuff

    def define_generic_colormap(self, map, texcoords):
        """Define the given texture as the colormap to be used to
        lookup the final color from the (per-vertex or per-face) texcoords. In
        the WGSL the colormap can be sampled using ``sample_colormap()``.
        Returns a list of bindings.

        For colormaps in mesh, line, points. Supports 1D/2D/3D
        textures, different texture formats.
        """

        filter_mode = f"{map.mag_filter}, {map.min_filter}, {map.mipmap_filter}"
        address_mode = f"{map.wrap_s}, {map.wrap_t}"

        sampler = GfxSampler(filter_mode, address_mode)
        texture_view = self._define_colormap_texture(map.texture)

        # Check that texture dim matches texcoords
        if not isinstance(texcoords, Buffer):
            raise ValueError("texture is present, but texcoords must be a buffer")
        vert_fmt = to_vertex_format(texcoords.format)
        if texture_view.view_dim == "1d" and "x" not in vert_fmt:
            pass
        elif not vert_fmt.endswith("x" + texture_view.view_dim[0]):
            raise ValueError(
                f"texcoords {texcoords.format} does not match texture_view {texture_view.view_dim}"
            )

        # Return bindings
        return [
            Binding("s_colormap", "sampler/filtering", sampler, "FRAGMENT"),
            Binding("t_colormap", "texture/auto", texture_view, "FRAGMENT"),
        ]

    def define_img_colormap(self, map):
        """Define the given texture view as the colormap to be used to
        lookup the final color from the image data.
        In the WGSL the colormap can be sampled using ``sample_colormap()``.
        Returns a list of bindings.
        """

        filter_mode = f"{map.mag_filter}, {map.min_filter}, {map.mipmap_filter}"
        address_mode = f"{map.wrap_s}, {map.wrap_t}"

        sampler = GfxSampler(filter_mode, address_mode)
        texture_view = self._define_colormap_texture(map.texture)

        # Check that texture dim matches image channels
        if int(texture_view.view_dim[0]) != self["img_nchannels"]:
            raise ValueError(
                f"Image channels {self['img_nchannels']} does not match texture_view {texture_view.view_dim}"
            )

        # Return bindings
        return [
            Binding("s_colormap", "sampler/filtering", sampler, "FRAGMENT"),
            Binding("t_colormap", "texture/auto", texture_view, "FRAGMENT"),
        ]

    def _define_colormap_texture(self, texture):
        # Get texture view
        if not isinstance(texture, Texture):
            raise TypeError("texture must be a Texture")
        texture_view = GfxTextureView(texture)
        # Dimensionality
        view_dim = texture_view.view_dim
        if view_dim not in ("1d", "2d", "3d"):
            raise ValueError(f"Unexpected texture dimension: '{view_dim}'")
        self["colormap_dim"] = view_dim
        self["colormap_coord_type"] = {
            "1d": "f32",
            "2d": "vec2<f32>",
            "3d": "vec3<f32>",
        }[view_dim]
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
        return texture_view
