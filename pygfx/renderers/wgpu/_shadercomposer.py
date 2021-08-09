import jinja2
import numpy as np

jinja_env = jinja2.Environment(
    block_start_string="{$",
    block_end_string="$}",
    variable_start_string="{{",
    variable_end_string="}}",
    line_statement_prefix="$$",
    undefined=jinja2.StrictUndefined,
)


class BaseShader:
    """Base shader object to compose and template shaders using jinja2."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self._uniform_codes = {}

    def __setitem__(self, key, value):
        self.kwargs[key] = value

    def __getitem__(self, key):
        return self.kwargs[key]

    def get_definitions(self):
        return "\n".join(self._uniform_codes.values())

    def get_code(self):
        raise NotImplementedError()

    def generate_wgsl(self, **kwargs):
        code = self.get_code()
        t = jinja_env.from_string(code)

        variables = self.kwargs.copy()
        variables.update(kwargs)

        try:
            return t.render(**variables)
        except jinja2.UndefinedError as err:
            msg = f"Canot compose shader: {err.args[0]}"
        raise ValueError(msg)  # don't raise within handler to avoid recursive tb

    def define_uniform(self, bindgroup, index, name, struct):

        structname = "Struct_" + name
        code = f"""
        [[block]]
        struct {structname} {{
        """.rstrip()

        if isinstance(struct, dict):
            dtype_struct = np.dtype([(key,) + val for key, val in struct.items()])
        elif isinstance(struct, np.dtype):
            if struct.fields is None:
                raise TypeError(f"define_uniform() needs a structured dtype")
            dtype_struct = struct
        else:
            raise TypeError(f"Unsupported struct type {struct.__class__.__name__}")

        for fieldname, (dtype, offset) in dtype_struct.fields.items():
            # Resolve primitive type
            primitive_type = dtype.base.name
            primitive_type = primitive_type.replace("float", "f")
            primitive_type = primitive_type.replace("uint", "u")
            primitive_type = primitive_type.replace("int", "i")
            # Resolve actual type (only scalar, vec, mat)
            shape = dtype.shape
            if shape == () or shape == (1,):
                wgsl_type = primitive_type
            elif len(shape) == 1:
                wgsl_type = f"vec{shape[0]}<{primitive_type}>"
            elif len(shape) == 2:
                # matNxM is Matrix of N columns and M rows
                wgsl_type = f"mat{shape[1]}x{shape[0]}<{primitive_type}>"
            else:
                raise TypeError("Unsupported type {dtype}")
            # Check alignment (https://www.w3.org/TR/WGSL/#alignment-and-size)
            if wgsl_type == primitive_type:
                alignment = 4
            elif wgsl_type.startswith("vec"):
                c = int(wgsl_type.split("<")[0][-1])
                alignment = 8 if c < 3 else 16
            elif wgsl_type.startswith("mat"):
                c = int(wgsl_type.split("<")[0][-1])
                alignment = 8 if c < 3 else 16
            else:
                raise TypeError(f"Unsupported wgsl type: {wgsl_type}")
            if offset % alignment != 0:
                raise TypeError(
                    f"Struct alignment error: {name}.{fieldname} alignment must be {alignment}"
                )

            code += f"\n            {fieldname}: {wgsl_type};"

        code += f"""
        }};

        [[group({bindgroup}), binding({index})]]
        var {name}: {structname};
        """
        self._uniform_codes[name] = code
