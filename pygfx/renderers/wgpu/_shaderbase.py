"""
Implements the base shader class. The shader is responsible for
providing the WGSL code, as well as providing the information to connect
it to the resources (buffers and textures) and some details on the
pipeline and rendering.
"""

import re

import jinja2
import numpy as np

from ...utils import array_from_shadertype
from ...resources import Buffer
from ._utils import (
    to_vertex_format,
    to_texture_format,
    generate_uniform_struct,
    hash_from_value,
)


jinja_env = jinja2.Environment(
    block_start_string="{$",
    block_end_string="$}",
    variable_start_string="{{",
    variable_end_string="}}",
    line_statement_prefix="$$",
    undefined=jinja2.StrictUndefined,
)


varying_types = ["f32", "vec2<f32>", "vec3<f32>", "vec4<f32>"]
varying_types = (
    varying_types
    + [t.replace("f", "i") for t in varying_types]
    + [t.replace("f", "u") for t in varying_types]
)

re_varying_getter = re.compile(r"[\s,\(\[]varyings\.(\w+)", re.UNICODE)
re_varying_setter = re.compile(r"\A\s*?varyings\.(\w+)(\.\w+)?\s*?\=")
builtin_varyings = {"position": "vec4<f32>"}


def resolve_varyings(wgsl):
    """Resolve varyings in the given wgsl:
    * Detect varyings being used.
    * Check that these are also set.
    * Remove assignments of varyings that are not used.
    * Include the Varyings struct.
    """
    assert isinstance(wgsl, str)

    # Split into lines, which is easier to process. Ensure it ends with newline in the end.
    lines = wgsl.splitlines()
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop(-1)
    lines.append("")

    # Prepare dicts that map name to list-of-linenr. And a tupe dict.
    assigned_varyings = {}
    used_varyings = {}
    types = {}  # varying types

    # We try to find the function that first uses the Varyings struct.
    struct_insert_pos = None

    # Go over all lines to:
    # - find the lines where a varying is set
    # - collect the types of these varyings
    for linenr, line in enumerate(lines):
        match = re_varying_setter.match(line)
        if match:
            # Get parts
            name = match.group(1)
            attr = match.group(2)
            # Handle builtin
            if name in builtin_varyings:
                used_varyings[name] = []
                types[name] = builtin_varyings[name]
            # Find type
            type = line[match.end() :].split("(")[0].strip().replace(" ", "")
            if type not in varying_types:
                type = ""
            # Triage
            if attr:
                pass  # Not actually a type but an attribute access
            elif not type:
                raise TypeError(
                    f"Varying {name!r} assignment needs an explicit cast (of a correct type), e.g. `varying.{name} = f32(3.0);`:\n{line}"
                )
            elif name in types and type != types[name]:
                raise TypeError(
                    f"Varying {name!r} assignment does not match expected type {types[name]}:\n{line}"
                )
            else:
                types[name] = type
            # Store position
            assigned_varyings.setdefault(name, []).append(linenr)

    # Go over all lines to:
    # - collect all used varyings
    # - find where the vertex-shader starts
    in_vertex_shader = False
    current_func_linenr = 0
    for linenr, line in enumerate(lines):
        line = line.strip()
        # Detect when we enter a new function
        if line.startswith("fn "):
            current_func_linenr = linenr
            if line.startswith("fn vs_main"):
                in_vertex_shader = True
            else:
                in_vertex_shader = False
        # Remove comments (shader code has no strings that can contain slashes)
        line = line.split("//")[0]
        if "Varyings" in line and struct_insert_pos is None:
            struct_insert_pos = current_func_linenr
        # Everything we find here is a match (prepend a space to allow an easier regexp)
        for match in re_varying_getter.finditer(" " + line):
            name = match.group(1)
            this_varying_is_set_on_this_line = linenr in assigned_varyings.get(name, [])
            if this_varying_is_set_on_this_line:
                pass
            elif in_vertex_shader:
                # If varyings are used in another way than setting, in the vertex shader,
                # we should either consider them "used", or possibly break the shader if
                # the used varying is disabled. So let's just not allow it.
                raise TypeError(
                    f"Varying {name!r} is read in the vertex shader, but only writing is allowed:\n{line}"
                )
            else:
                used_varyings.setdefault(name, []).append(linenr)

    # Check if all used varyings are assigned
    for name in used_varyings:
        if name not in assigned_varyings:
            line = lines[used_varyings[name][0]]
            raise TypeError(f"Varying {name!r} is read, but not assigned:\n{line}")

    # Comment-out the varying setter if its unused elsewhere in the shader
    for name, linenrs in assigned_varyings.items():
        if name not in used_varyings:
            for linenr in linenrs:
                line = lines[linenr]
                indent = line[: len(line) - len(line.lstrip())]
                lines[linenr] = indent + "// unused: " + line[len(indent) :]
                # Deal with multiple-line assignments
                line_s = line.strip()
                while not line_s.endswith(";"):
                    linenr += 1
                    line_s = lines[linenr].strip()
                    unexpected = "fn ", "struct ", "var ", "let ", "}"
                    if line_s.startswith(unexpected) or linenr == len(lines) - 1:
                        raise TypeError(
                            f"Varying {name!r} assignment seems to be missing a semicolon:\n{line}"
                        )
                    lines[linenr] = indent + "// " + line_s

    # Build and insert the struct
    if struct_insert_pos is not None:
        # Maybe we should move up a bit
        if struct_insert_pos > 0:
            if lines[struct_insert_pos - 1].lstrip().startswith("@"):
                struct_insert_pos -= 1
        # First divide into slot-based and builtins
        used_varyings = set(used_varyings)
        used_builtins = used_varyings.intersection(builtin_varyings)
        used_slots = used_varyings.difference(used_builtins)
        used_slots = list(sorted(used_slots))
        # Build struct
        struct_lines = ["struct Varyings {"]
        for slotnr, name in enumerate(used_slots):
            struct_lines.append(f"    @location({slotnr}) {name} : {types[name]},")
        for name in sorted(used_builtins):
            struct_lines.append(f"    @builtin({name}) {name} : {types[name]},")
        struct_lines.append("};\n")
        # Apply indentation and insert
        line = lines[struct_insert_pos]
        indent = line[: len(line) - len(line.lstrip())]
        struct_lines = [indent + line for line in struct_lines]
        lines.insert(struct_insert_pos, "\n".join(struct_lines))
    else:
        assert not used_varyings, "woops, did not expect used_varyings here"

    # Return modified code
    return "\n".join(lines)


re_depth_setter = re.compile(r"\A\s*?out\.depth\s*?\=")


def resolve_depth_output(wgsl):
    """When out.depth is set (in the fragment shader), adjust the FragmentOutput
    to accept depth.
    """
    assert isinstance(wgsl, str)

    # Split into lines, which is easier to process. Ensure it ends with newline in the end.
    lines = wgsl.splitlines()
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop(-1)
    lines.append("")

    # Detect whether the depth is set in the shader. We're going to assume
    # this is in the fragment shader. We check for "out.depth =".
    # Background: by default the depth is based on the geometry (set
    # by vertex shader and interpolated). It is possible for a fragment
    # shader to write the depth instead. If this is done, the GPU cannot
    # do early depth testing; the fragment shader must be run for the
    # depth to be known.
    depth_is_set = False
    struct_linenr = -1
    for linenr, line in enumerate(lines):
        if line.lstrip().startswith("struct FragmentOutput {"):
            struct_linenr = linenr
        elif re_depth_setter.match(line):
            depth_is_set = True
            if struct_linenr >= 0:
                break

    if depth_is_set:
        if struct_linenr < 0:
            raise TypeError("FragmentOutput definition not found.")
        depth_field = "    @builtin(frag_depth) depth : f32,"
        line = lines[struct_linenr]
        indent = line[: len(line) - len(line.lstrip())]
        lines.insert(struct_linenr + 1, indent + depth_field)

    return "\n".join(lines)


class BaseShader:
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

        self._typedefs = {}
        self._binding_codes = {}
        self._uniform_struct_names = {}  # dtype -> name

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

    def lock_hash(self):
        self._hash = self._get_hash()

    def unlock_hash(self):
        self._hash = None

    @property
    def hash(self):
        """A hash of the current state of the shader. If the hash changed,
        it's likely that the shader changed.
        """
        if self._hash is None:
            return self._get_hash()
        else:
            return self._hash

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
        code = ""
        code += "\n".join(self._typedefs.values())
        code += "\n"
        code += "\n".join(self._binding_codes.values())
        return code

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
            self.define_binding(bindgroup, index, binding)

    def define_binding(self, bindgroup, index, binding):
        """Define a uniform, buffer, sampler, or texture. The produced wgsl
        will be part of the code returned by ``get_definitions()``. The binding
        must be a Binding object.
        """
        if binding.type == "buffer/uniform":
            self._define_uniform(bindgroup, index, binding)
        elif binding.type.startswith("buffer"):
            self._define_buffer(bindgroup, index, binding)
        elif binding.type.startswith("sampler"):
            self._define_sampler(bindgroup, index, binding)
        elif binding.type.startswith("texture"):
            self._define_texture(bindgroup, index, binding)
        else:
            raise RuntimeError(
                f"Unknown binding {binding.name} with type {binding.type}"
            )

    def _define_uniform(self, bindgroup, index, binding):
        resource = binding.resource
        if isinstance(resource, dict):
            dtype_struct = array_from_shadertype(resource).dtype
        elif isinstance(resource, Buffer):
            if resource.data.dtype.fields is None:
                raise TypeError(f"define_uniform() needs a structured dtype")
            dtype_struct = resource.data.dtype
        elif isinstance(resource, np.dtype):
            if resource.fields is None:
                raise TypeError(f"define_uniform() needs a structured dtype")
            dtype_struct = resource
        else:
            raise TypeError(f"Unsupported struct type {resource.__class__.__name__}")

        # Get struct name
        struct_hash = str(dtype_struct)

        try:
            structname = self._uniform_struct_names[struct_hash]
            if binding.structname is not None:
                # Do we need to ensure that a dtype corresponds to only one struct?
                assert (
                    structname == binding.structname
                ), "dtype[{struct_hash}] has been defined as struct[{structname}]"
        except KeyError:
            # sometimes, we need a meaningful alias for the struct name.
            if binding.structname is not None:
                structname = binding.structname
                assert (
                    structname not in self._uniform_struct_names.values()
                ), "structname has been used for another dtype"
            else:
                # auto generate struct name
                structname = f"Struct_u_{len(self._uniform_struct_names)+1}"

            self._uniform_struct_names[struct_hash] = structname

        if structname not in self._typedefs:
            struct_code = generate_uniform_struct(dtype_struct, structname)
            self._typedefs[structname] = struct_code

        uniform_type_name = (
            f"array<{structname}, {binding.resource.data.shape[0]}>"  # array of struct
            if isinstance(resource, Buffer) and resource.data.shape  # Buffer.items > 1
            else structname
        )

        code = f"""
        @group({bindgroup}) @binding({index})
        var<uniform> {binding.name}: {uniform_type_name};
        """.rstrip()
        self._binding_codes[binding.name] = code

    def _define_buffer(self, bindgroup, index, binding):
        # Get format, and split in the scalar part and the number of channels
        fmt = to_vertex_format(binding.resource.format)
        if "x" in fmt:
            fmt_scalar, _, nchannels = fmt.partition("x")
            nchannels = int(nchannels)
        else:
            fmt_scalar = fmt
            nchannels = 1

        # Define scalar type: i32, u32 or f32
        # Since the stride must be a multiple of 4 for storage buffers,
        # the supported types is limited until we support structured numpy arrays.
        scalar_type = (
            fmt_scalar.replace("float", "f").replace("uint", "u").replace("sint", "i")
        )
        if not scalar_type.endswith("32"):
            raise ValueError(
                f"Buffer format {fmt} not supported, format must have a stride of 4 bytes: i4, u4 of f4."
            )

        # Define the element types. The element_type2 is the actual type.
        # Because for storage buffers a vec3 has an alignment of 16, we have to
        # be creative for vec3: we bind the buffer as if it was 1D, and convert
        # in the accessor function.
        if nchannels == 1:
            element_type1 = element_type2 = scalar_type
            stride = 4
        elif nchannels == 3:
            element_type1 = scalar_type
            element_type2 = f"vec{nchannels}<{scalar_type}>"
            stride = 4
        else:
            element_type1 = element_type2 = f"vec{nchannels}<{scalar_type}>"
            stride = 4 * nchannels

        stride  # not actually used anymore in wgsl?

        # Produce the binding code and accessor function
        type_modifier = "read" if "read_only" in binding.type else "read_write"
        code = f"""
        @group({bindgroup}) @binding({index})
        var<storage, {type_modifier}> {binding.name}: array<{element_type1}>;
        fn load_{binding.name} (i: i32) -> {element_type2} {{
        """.rstrip()
        if element_type1 == element_type2:
            code += f" return {binding.name}[i];"
        elif nchannels == 2:
            code += f" return {element_type2}( {binding.name}[i * 2], {binding.name}[i * 2 + 1] );"
        elif nchannels == 3:
            code += f" return {element_type2}( {binding.name}[i * 3], {binding.name}[i * 3 + 1], {binding.name}[i * 3 + 2] );"
        else:  # nchannels == 4
            code += f" return {element_type2}( {binding.name}[i * 4], {binding.name}[i * 4 + 1], {binding.name}[i * 4 + 2], {binding.name}[i * 4 + 3] );"
        code += " }"
        self._binding_codes[binding.name] = code

    def _define_sampler(self, bindgroup, index, binding):
        code = f"""
        @group({bindgroup}) @binding({index})
        var {binding.name}: sampler;
        """.rstrip()
        self._binding_codes[binding.name] = code

    def _define_texture(self, bindgroup, index, binding):
        texture = binding.resource  # GfxTextureView
        format = to_texture_format(texture.format)
        if "norm" in format or "float" in format:
            format = "f32"
        elif "uint" in format:
            format = "u32"
        else:
            format = "i32"
        code = f"""
        @group({bindgroup}) @binding({index})
        var {binding.name}: texture_{texture.view_dim}<{format}>;
        """.rstrip()
        self._binding_codes[binding.name] = code
