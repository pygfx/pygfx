"""
Implements the base shader class. The shader is responsible for
providing the WGSL code, as well as providing the information to connect
it to the resources (buffers and textures) and some details on the
pipeline and rendering.
"""

import jinja2

from ..engine.utils import hash_from_value

from .resolve import resolve_varyings, resolve_depth_output
from .definitions import BindingDefinitions


jinja_env = jinja2.Environment(
    block_start_string="{$",
    block_end_string="$}",
    variable_start_string="{{",
    variable_end_string="}}",
    line_statement_prefix="$$",
    undefined=jinja2.StrictUndefined,
)


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

        self._definitions = BindingDefinitions()

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
            return hash_from_value([fullname, self.code_definitions(), self.kwargs])
        else:
            # More reliable (e.g. in an interactove session).
            return hash_from_value([self.get_code(), self.kwargs])

    def code_definitions(self):
        """Get the WGSL definitions of types and bindings (uniforms, storage
        buffers, samplers, and textures).
        """
        return self._definitions.get_code()

    def get_code(self):
        """Implement this to compose the total (but still templated)
        shader. This method is called by ``generate_wgsl()``.
        """
        return self.code_definitions()

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
            t = jinja_env.from_string(code1)

            # Also add some additional kwargs, some of these may be set during get_code()
            shader_kwargs.update(self.derived_kwargs)

            err_msg = None
            try:
                code2 = t.render(**shader_kwargs)
            except jinja2.UndefinedError as err:
                err_msg = f"Cannot compose shader: {err.args[0]}"

            if err_msg:
                # Don't raise within handler to avoid recursive tb
                raise ValueError(err_msg)
            else:
                code2 = resolve_varyings(code2)
                code2 = resolve_depth_output(code2)
                return code2
        finally:
            self.kwargs = ori_kwargs

    def define_bindings(self, bindgroup, bindings_dict):
        """Define a collection of bindings organized in a dict."""
        for index, binding in bindings_dict.items():
            self._definitions.define_binding(bindgroup, index, binding)

    def define_binding(self, bindgroup, index, binding):
        """Define a uniform, buffer, sampler, or texture. The produced wgsl
        will be part of the code returned by ``get_definitions()``. The binding
        must be a Binding object.
        """
        self._definitions.define_binding(bindgroup, index, binding)
