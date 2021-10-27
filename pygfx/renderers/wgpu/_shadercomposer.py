import jinja2
import numpy as np
import wgpu

from ...utils import array_from_shadertype
from ...resources import Buffer
from ._conv import to_vertex_format, to_texture_format


jinja_env = jinja2.Environment(
    block_start_string="{$",
    block_end_string="$}",
    variable_start_string="{{",
    variable_end_string="}}",
    line_statement_prefix="$$",
    undefined=jinja2.StrictUndefined,
)


visibility_render = wgpu.ShaderStage.VERTEX | wgpu.ShaderStage.FRAGMENT
visibility_all = (
    wgpu.ShaderStage.VERTEX | wgpu.ShaderStage.FRAGMENT | wgpu.ShaderStage.COMPUTE
)


class Binding:
    """Simple object to hold together some information about a binding, for internal use.

    * name: the name in wgsl
    * type: "buffer/subtype", "sampler/subtype", "texture/subtype", "storage_texture/subtype".
      The subtype: depends on the type:
      BufferBindingType, SamplerBindingType, TextureSampleType, or StorageTextureAccess.
    * resource: Buffer, Texture or TextureView.
    * visibility: wgpu.ShaderStage flag
    * kwargs: could add more specifics in the future.
    """

    def __init__(self, name, type, resource, visibility=visibility_render, **kwargs):
        if isinstance(visibility, str):
            visibility = getattr(wgpu.ShaderStage, visibility)
        self.name = name
        self.type = type
        self.resource = resource
        self.visibility = visibility
        for key, val in kwargs.items():
            setattr(self, key, val)


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
        self.kwargs = kwargs
        self._typedefs = {}
        self._binding_codes = {}

    def __setitem__(self, key, value):
        if hasattr(self.__class__, key):
            msg = f"Templating variable {key} causes name clash with class attribute."
            raise AttributeError(msg)
        self.kwargs[key] = value

    def __getitem__(self, key):
        return self.kwargs[key]

    def get_definitions(self):
        """Get the definitions of types and bindings (uniforms, storage
        buffers, samplers, and textures).
        """
        code = (
            "\n".join(self._typedefs.values())
            + "\n"
            + "\n".join(self._binding_codes.values())
        )
        return code

    def get_code(self):
        """Implement this to compose the total (templated) shader. This method is called
        by ``generate_wgsl()``.
        """
        raise NotImplementedError()

    def generate_wgsl(self, **kwargs):
        """Generate the final WGSL with the templating resolved by jinja2.
        Also accepts templating variables as kwargs.
        """
        code = self.get_code()
        t = jinja_env.from_string(code)

        variables = self.kwargs.copy()
        variables.update(kwargs)

        try:
            return t.render(**variables)
        except jinja2.UndefinedError as err:
            msg = f"Canot compose shader: {err.args[0]}"
        raise ValueError(msg)  # don't raise within handler to avoid recursive tb

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

        structname = "Struct_" + binding.name
        code = f"""
        [[block]]
        struct {structname} {{
        """.rstrip()

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

        # Obtain names of fields that are arrays. This is encoded as an empty field with a
        # name that has the array-fields-names separated with double underscores.
        array_names = []
        for fieldname in dtype_struct.fields.keys():
            if fieldname.startswith("__") and fieldname.endswith("__"):
                array_names.extend(fieldname.replace("__", " ").split())

        # Process fields
        for fieldname, (dtype, offset) in dtype_struct.fields.items():
            if fieldname.startswith("__"):
                continue
            # Resolve primitive type
            primitive_type = dtype.base.name
            primitive_type = primitive_type.replace("float", "f")
            primitive_type = primitive_type.replace("uint", "u")
            primitive_type = primitive_type.replace("int", "i")
            # Resolve actual type (only scalar, vec, mat)
            shape = dtype.shape
            # Detect array
            length = -1
            if fieldname in array_names:
                length = shape[0]
                shape = shape[1:]
            # Obtain base type
            if shape == () or shape == (1,):
                # A scalar
                wgsl_type = align_type = primitive_type
            elif len(shape) == 1:
                # A vector
                n = shape[0]
                if n < 2 or n > 4:
                    raise TypeError(f"Type {dtype} looks like an unsupported vec{n}.")
                wgsl_type = align_type = f"vec{n}<{primitive_type}>"
            elif len(shape) == 2:
                # A matNxM is Matrix of N columns and M rows
                n, m = shape[1], shape[0]
                if n < 2 or n > 4 or m < 2 or m > 4:
                    raise TypeError(
                        f"Type {dtype} looks like an unsupported mat{n}x{m}."
                    )
                align_type = f"vec{m}<primitive_type>"
                wgsl_type = f"mat{n}x{m}<{primitive_type}>"
            else:
                raise TypeError(f"Unsupported type {dtype}")
            # If an array, wrap it
            if length == 0:
                wgsl_type = align_type = None  # zero-length; dont use
            elif length > 0:
                wgsl_type = f"array<{wgsl_type},{length}>"
            else:
                pass  # not an array

            # Check alignment (https://www.w3.org/TR/WGSL/#alignment-and-size)
            if not wgsl_type:
                continue
            elif align_type == primitive_type:
                alignment = 4
            elif align_type.startswith("vec"):
                c = int(align_type.split("<")[0][-1])
                alignment = 8 if c < 3 else 16
            else:
                raise TypeError(f"Cannot establish alignment of wgsl type: {wgsl_type}")
            if offset % alignment != 0:
                # If this happens, our array_from_shadertype() has failed.
                raise TypeError(
                    f"Struct alignment error: {binding.name}.{fieldname} alignment must be {alignment}"
                )

            code += f"\n            {fieldname}: {wgsl_type};"

        code += "\n        };"
        self._typedefs[structname] = code

        code = f"""
        [[group({bindgroup}), binding({index})]]
        var<uniform> {binding.name}: {structname};
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
                f"Buffer format {format} not supported, format must have a stride of 4 bytes: i4, u4 of f4."
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

        # Prepare some names
        typename = "Buffer_" + element_type1.replace("<", "").replace(">", "")
        type_modifier = "read" if "read_only" in binding.type else "read_write"

        # Produce the type definition
        code = f"""
        [[block]]
        struct {typename} {{
            data: [[stride({stride})]] array<{element_type1}>;
        }};
        """.rstrip()
        self._typedefs[typename] = code

        # Produce the binding code and accessor function
        code = f"""
        [[group({bindgroup}), binding({index})]]
        var<storage, {type_modifier}> {binding.name}: {typename};
        fn load_{binding.name} (i: i32) -> {element_type2} {{
        """.rstrip()
        if element_type1 == element_type2:
            code += f" return {binding.name}.data[i];"
        elif nchannels == 2:
            code += f" return {element_type2}( {binding.name}.data[i * 2], {binding.name}.data[i * 2 + 1] );"
        elif nchannels == 3:
            code += f" return {element_type2}( {binding.name}.data[i * 3], {binding.name}.data[i * 3 + 1], {binding.name}.data[i * 3 + 2] );"
        else:  # nchannels == 4
            code += f" return {element_type2}( {binding.name}.data[i * 4], {binding.name}.data[i * 4 + 1], {binding.name}.data[i * 4 + 2], {binding.name}.data[i * 4 + 3] );"
        code += " }"
        self._binding_codes[binding.name] = code

    def _define_sampler(self, bindgroup, index, binding):
        code = f"""
        [[group({bindgroup}), binding({index})]]
        var {binding.name}: sampler;
        """.rstrip()
        self._binding_codes[binding.name] = code

    def _define_texture(self, bindgroup, index, binding):
        texture = binding.resource  # or view
        format = to_texture_format(texture.format)
        if "norm" in format or "float" in format:
            format = "f32"
        elif "uint" in format:
            format = "u32"
        else:
            format = "i32"
        code = f"""
        [[group({bindgroup}), binding({index})]]
        var {binding.name}: texture_{texture.view_dim}<{format}>;
        """.rstrip()
        self._binding_codes[binding.name] = code


class WorldObjectShader(BaseShader):
    """A base shader for world objects. This class implements common functions
    that can be used in all material-specific renderers.
    """

    def __init__(self, wobject, **kwargs):
        super().__init__(**kwargs)

        self.kwargs["n_clipping_planes"] = len(wobject.material.clipping_planes)
        self.kwargs["clipping_mode"] = wobject.material.clipping_mode

    def common_functions(self):

        if not self.kwargs["n_clipping_planes"]:
            clipping_plane_code = """
            fn apply_clipping_planes(world_pos: vec3<f32>) { }  // zero planes
            """
        else:
            clipping_plane_code = """
            fn apply_clipping_planes(world_pos: vec3<f32>) {
                var clipped: bool = {{ 'false' if clipping_mode == 'ANY' else 'true' }};
                for (var i=0; i<{{ n_clipping_planes }}; i=i+1) {
                    let plane = u_material.clipping_planes[i];
                    let plane_clipped = dot( world_pos, plane.xyz ) < plane.w;
                    clipped = clipped {{ '||' if clipping_mode == 'ANY' else '&&' }} plane_clipped;
                }
                if (clipped) { discard; }
            }
            """

        world_pos_code = """
        fn ndc_to_world_pos(ndc_pos: vec4<f32>) -> vec3<f32> {
            let ndc_to_world = u_stdinfo.cam_transform_inv * u_stdinfo.projection_transform_inv;
            let world_pos = ndc_to_world * ndc_pos;
            return world_pos.xyz / world_pos.w;
        }
        """

        return clipping_plane_code + world_pos_code
