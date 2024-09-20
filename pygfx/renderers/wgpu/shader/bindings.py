import numpy as np

from ....utils import array_from_shadertype
from ....resources import Buffer

from ..engine.utils import (
    to_vertex_format,
    to_texture_format,
    generate_uniform_struct,
)


class BindingDefinitions:
    """Track definitions of bindings."""

    def __init__(self):
        self._typedefs = {}
        self._binding_codes = {}
        self._uniform_struct_names = {}  # dtype -> name

    def get_code(self):
        """Get the wgsl source code for the collected bindings."""
        code = ""
        code += "\n".join(self._typedefs.values())
        code += "\n"
        code += "\n".join(self._binding_codes.values())
        return code

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

        view_dim = texture.view_dim
        view_dim = view_dim.replace("-", "_")  # 2d-array -> 2d_array

        code = f"""
        @group({bindgroup}) @binding({index})
        var {binding.name}: texture_{view_dim}<{format}>;
        """.rstrip()
        self._binding_codes[binding.name] = code
