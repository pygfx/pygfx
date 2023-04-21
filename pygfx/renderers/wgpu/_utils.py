"""
Utils for the wgpu renderer.
"""

import weakref

import wgpu

from .._base import RenderFunctionRegistry

registry = RenderFunctionRegistry()


def register_wgpu_render_function(wobject_cls, material_cls):
    """Decorator for WGPU rendering functions.

    Parameters
    ----------
    wobject_cls : WorldObject
        The world object that this function knows how to render.
    material_cls : Material
        The world object that this function knows how to render.

    """

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


def to_wgsl_vertex_type(format):
    primitives = {
        "i1": "i8",
        "u1": "u8",
        "i2": "i16",
        "u2": "u16",
        "i4": "i32",
        "u4": "u32",
        "f2": "f16",
        "f4": "f32",
    }

    primitive = primitives[format[-2:]]
    if len(format) == 2:
        return primitive
    elif len(format) == 4 and format[1] == "x":  # e.g. 3xf4
        return f"vec{format[0]}<{primitive}>"


def generate_uniform_struct(dtype_struct, structname):
    code = f"""
        struct {structname} {{
    """.rstrip()

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
                raise TypeError(f"Type {dtype} looks like an unsupported mat{n}x{m}.")
            align_type = f"vec{m}<{primitive_type}>"
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
                f"Struct alignment error: {structname}.{fieldname} alignment must be {alignment}"
            )

        code += f"\n            {fieldname}: {wgsl_type},"

    code += "\n        };"

    return code


class GpuCache:
    """A chache for GPU objects."""

    _caches = {}

    @classmethod
    def get_cache_stats(cls):
        """Get a dict mapping cache names to item counts."""
        return {name: cache.get_count() for name, cache in GpuCache._caches.items()}

    def __init__(self, name):
        assert isinstance(name, str)
        assert name not in GpuCache._caches
        GpuCache._caches[name] = self

        self._objects = weakref.WeakValueDictionary()

    def get_count(self):
        """Get the number of (alive) objects in the cache."""
        return len(list(self._objects.values()))

    def get(self, hash):
        """Get the cached object or None."""
        return self._objects.get(hash, None)

    def set(self, hash, ob):
        """Store the given object under the given hash.
        Note that the cache does not have a (strong) ref to the object.
        """
        self._objects[hash] = ob


class GfxSampler:
    """Simple wrapper for a GPUSampler. Should be considered read-only."""

    def __init__(self, filter="nearest", address_mode="clamp"):
        self.filter = filter
        self.address_mode = address_mode
        self._wgpu_object = None


class GfxTextureView:
    """Simple wrapper for a GPUTextureView. Should be considered read-only."""

    def __init__(self, texture, *, view_dim=None, layer_range=None, aspect=None):
        format = texture.format

        # Check view_dim
        default_view_dim = f"{texture.dim}d"
        if view_dim is None:
            view_dim = default_view_dim
        elif isinstance(view_dim, int):
            view_dim = f"{view_dim}d"

        # Check layer_range (is half-open range)
        default_layer_range = 0, texture.size[2]
        if layer_range is None:
            layer_range = default_layer_range
        else:
            assert isinstance(layer_range, tuple) and len(layer_range) == 2

        # Check aspect
        default_aspect = wgpu.TextureAspect.all
        if aspect is None:
            aspect = default_aspect
        else:
            assert aspect in wgpu.TextureAspect

        # Is this a default view on the texture?
        self.is_default_view = (
            view_dim == default_view_dim
            and layer_range == default_layer_range
            and aspect == default_aspect
        )

        # Store attributes
        self.texture = texture
        self.format = format
        self.view_dim = view_dim
        self.aspect = aspect
        self.layer_range = layer_range
        self._wgpu_object = None
