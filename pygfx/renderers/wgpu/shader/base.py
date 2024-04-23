"""
Implements the base shader class. The shader is responsible for
providing the WGSL code, as well as providing the information to connect
it to the resources (buffers and textures) and some details on the
pipeline and rendering.
"""

from typing import Callable, Tuple
import jinja2

from ....resources import Buffer, Texture
from ..engine.utils import (
    GfxSampler,
    GfxTextureView,
    to_vertex_format,
    to_texture_format,
    hash_from_value,
)
from ..engine.binding import Binding
from .resolve import resolve_varyings, resolve_depth_output
from .definitions import BindingDefinitions


loader = jinja2.PrefixLoader({}, delimiter=".")
loader.mapping["pygfx"] = jinja2.PackageLoader("pygfx.renderers.wgpu.wgsl", ".")

jinja_env = jinja2.Environment(
    block_start_string="{$",
    block_end_string="$}",
    variable_start_string="{{",
    variable_end_string="}}",
    line_statement_prefix="$$",
    undefined=jinja2.StrictUndefined,
    loader=loader,
)


def register_wgsl_loader(context, func):
    """Register a source for shader snippets.

    When code is encountered that looks like::

       {$ include 'some_context.name.wgsl' $}

    The loader for "some_context" is looked up and used to load the wgsl to include.
    This function allows registering a loader for your downstream package.

    Parameters
    ----------
    context : str
        The context of the loader.
    func: callable
        The function that will be called when a shader is loaded for the given context.
        The function must accept one positional argument (the name to include).
    """
    if not (isinstance(context, str) and "." not in context):
        raise TypeError("Wgsl load context must be a string witout dots.")
    if not callable(func):
        raise TypeError("The given wgsl load func must be callable.")
    if context in loader.mapping:
        raise RuntimeError(f"A loader is already registered for '{context}'.")
    loader.mapping[context] = func


class ShaderInterface:
    """Define what a shader object must look like from the pov from the pipeline."""

    def __init__(self):
        self._hash = None

    def lock_hash(self):
        self._hash = self._get_hash()

    def unlock_hash(self):
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

    def generate_wgsl(self, **kwargs):
        raise NotImplementedError()

    def get_bindings(self, wobject, shared):
        raise NotImplementedError()

    def get_pipeline_info(self, wobject, shared):
        raise NotImplementedError()

    def get_render_info(self, wobject, shared):
        raise NotImplementedError()


class BaseShader(ShaderInterface):
    """Base shader object to compose and template shaders using jinja2.

    Templating variables can be provided as kwargs, set (and get) as attributes,
    or passed as kwargs to ``generate_wgsl()``.

    The idea is that this class is subclassed, and that methods are
    implemented that return (templated) shader code. The purpose of
    using methods for this is to easier navigate/structure parts of the
    shader. Subclasses should also implement ``get_code()`` that simply
    composes the different parts of the total shader.
    """

    def __init__(self, **kwargs):
        # The shader kwargs
        self.kwargs = kwargs
        # Additional shader kwargs for private stuff. Only use for derived values:
        # values that are already defined by an item in kwargs!!
        self.derived_kwargs = {}
        # The stored hash. If set, the hash is locked, and kwargs cannot
        # be set. This is to prevent the shader implementations setting
        # shader kwargs *after* the pipeline has obtained the hash.
        self._hash = None

        self._binding_definitions = BindingDefinitions()

    def __setitem__(self, key, value):
        if hasattr(self.__class__, key):
            msg = f"Templating variable {key} causes name clash with class attribute."
            raise KeyError(msg)
        if self._hash is not None:
            raise RuntimeError(
                "Shader is trying to set shaders kwargs while they're locked."
            )
        self.kwargs[key] = value

    def __getitem__(self, key):
        return self.kwargs[key]

    def _get_hash(self):
        # The full name of this shader class.
        fullname = self.__class__.__module__ + "." + self.__class__.__name__

        # If we assume that the shader class produces the same code for
        # a specific set of kwargs, we can use the fullname in the hash
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
            # Faster, but assumes that the produced code only depends on kwargs.
            return hash_from_value(
                [fullname, self._binding_definitions.get_code(), self.kwargs]
            )
        else:
            # More reliable (e.g. in an interactive session).
            return hash_from_value(
                [self._binding_definitions.get_code(), self.get_code(), self.kwargs]
            )

    def get_code(self):
        """Implement this to compose the total (but still templated)
        shader. This method is called by ``generate_wgsl()``.
        """
        raise NotImplementedError()

    def generate_wgsl(self, **kwargs):
        """Generate the final WGSL. Calls get_code() and then resolves
        the templating variables, varyings, and depth output.
        """

        # Compose shader kwargs
        shader_kwargs = self.kwargs.copy()
        shader_kwargs.update(kwargs)

        # Set self.kwargs, because self.get_code() might use it.
        ori_kwargs = self.kwargs
        self.kwargs = shader_kwargs

        try:
            code1 = self.get_code()

            # Also add some additional kwargs, some of these may have been set during get_code()
            shader_kwargs.update(self.derived_kwargs)
            shader_kwargs["bindings_code"] = self._binding_definitions.get_code()
            # todo: do this via a dict loader or something?

            err_msg = None
            try:
                t = jinja_env.from_string(code1)
                code2 = t.render(**shader_kwargs)
            except jinja2.UndefinedError as err:
                err_msg = f"Cannot compose shader: {err.args[0]}"

            if err_msg:
                # Don't raise within handler to avoid recursive tb
                raise ValueError(err_msg)

            # Auto-insert varyings struct
            code2 = resolve_varyings(code2)

            # Tweak FragmentOutput if depth is set in frag shader.
            code2 = resolve_depth_output(code2)

            return code2

        finally:
            self.kwargs = ori_kwargs

    def define_bindings(self, bindgroup, bindings_dict):
        """Define a collection of bindings organized in a dict."""
        for index, binding in bindings_dict.items():
            self._binding_definitions.define_binding(bindgroup, index, binding)

    def define_binding(self, bindgroup, index, binding):
        """Define a uniform, buffer, sampler, or texture.

        The binding must be a Binding object. The code that defines the binding
        will be inserted in ``pygfx.std.wgsl``.
        """
        self._binding_definitions.define_binding(bindgroup, index, binding)


class WorldObjectShader(BaseShader):
    """A base shader for world objects. Must be subclassed to implement
    a shader for a specific material. This class also implements common
    functions that can be used in all material-specific renderers.
    """

    type = "render"  # must be "compute" or "render"
    needs_bake_function = False

    def __init__(self, wobject, **kwargs):
        super().__init__(**kwargs)

        # Init values related to blender. The blending_code will include FragmentOutput and get_fragment_output()
        self.kwargs.setdefault("write_pick", True)
        self.kwargs.setdefault("blending_code", "")

        # Init other variables for the environment.
        self.kwargs.setdefault("lighting", "")

        # Init colormap values
        self.kwargs.setdefault("colormap_dim", "")
        self.kwargs.setdefault("colormap_nchannels", 1)
        self.kwargs.setdefault("colormap_format", "f32")

        # Init clip plane values
        self["n_clipping_planes"] = wobject.material.clipping_plane_count
        self["clipping_mode"] = wobject.material.clipping_mode

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

        self["colormap_interpolation"] = interpolation
        sampler = GfxSampler(interpolation, "repeat")
        texture_view = self._define_colormap_texture(texture)

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
            Binding("s_texcoords", "buffer/read_only_storage", texcoords, "VERTEX"),
        ]

    def define_img_colormap(self, texture, interpolation="linear"):
        """Define the given texture view as the colormap to be used to
        lookup the final color from the image data.
        In the WGSL the colormap can be sampled using ``sample_colormap()``.
        Returns a list of bindings.
        """

        self["colormap_interpolation"] = interpolation
        sampler = GfxSampler(interpolation, "clamp")
        texture_view = self._define_colormap_texture(texture)

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
        self.derived_kwargs["colormap_coord_type"] = {
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
