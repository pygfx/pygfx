import wgpu

from .. import RenderFunctionRegistry


registry = RenderFunctionRegistry()


def register_wgpu_render_function(wobject_cls, material_cls):
    """Decorator to register a WGPU render function."""

    def _register_wgpu_renderer(f):
        registry.register(wobject_cls, material_cls, f)
        return f

    return _register_wgpu_renderer


def to_vertex_format(format):
    """Convert pygfx' own format to the wgpu format."""
    if format in wgpu.VertexFormat:
        return format
    elif format in wgpu.IndexFormat:
        return format

    # Get primitive type
    primitives = {
        "i1": "sint8",
        "u1": "uint8",
        "i2": "sint16",
        "u2": "uint16",
        "i4": "sint32",
        "u4": "uint32",
        "f2": "float16",
        "f4": "float32",
    }
    primitive = primitives[format[-2:]]

    if len(format) == 2:
        return primitive
    elif len(format) == 4 and format[1] == "x":  # e.g. 3xf4
        if format[0] == "1":
            return primitive
        elif format[0] in "234":
            return primitive + "x" + str(format[0])
        raise ValueError(f"Unexpected tuple size in index/vertex format '{format}'")
    else:
        raise ValueError(f"Unexpected length of index/vertex format '{format}'")


def to_texture_format(format):
    """Convert pygfx' own format to the wgpu format."""
    if format in wgpu.TextureFormat:
        return format

    # Note how we use normalized types (float in the shader) where we can,
    # because these types can work with an interpolating sampler.
    primitives = {
        "i1": "8snorm",
        "u1": "8unorm",
        "i2": "16sint",
        "u2": "16uint",
        "i4": "32sint",
        "u4": "32uint",
        "f2": "16float",
        "f4": "32float",
    }
    primitive = primitives[format[-2:]]

    if len(format) == 2:
        return "r" + primitive
    elif len(format) == 4 and format[1] == "x":  # e.g. 3xf4
        if format[0] == "1":
            return "r" + primitive
        elif format[0] == "2":
            return "rg" + primitive
        elif format[0] == "3":
            return "rgb" + primitive
        elif format[0] == "4":
            return "rgba" + primitive
        else:
            raise ValueError(f"Unexpected tuple size in texture format '{format}'")
    else:
        raise ValueError(f"Unexpected length of texture format '{format}'")
