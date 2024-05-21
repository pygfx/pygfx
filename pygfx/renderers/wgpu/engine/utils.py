"""
Utils for the wgpu renderer.
"""

import json
import weakref

import wgpu

from ....utils.renderfunctionregistry import RenderFunctionRegistry


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
    # Note: in practice we use this to decorate shader classes, but the point is
    # that it registers a callable that - when called - produces one or more
    # BaseShader instances.

    def _register_wgpu_renderer(f):
        registry.register(wobject_cls, material_cls, f)
        return f

    return _register_wgpu_renderer


def nchannels_from_format(format):
    """Return the number of channels from a vertex-buffer format.

    Channels as in elements per item. I.e. will be 1 and 4 for a grayscale and
    rgba color buffer, respectively.
    """
    return int(to_vertex_format(format).partition("x")[2] or "1")


def to_index_format(format):
    """Convert any pygfx-allowed buffer format to the wgpu.IndexFormat."""
    if format in wgpu.IndexFormat:
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

    # Ignore dimensions; consider flattened array (indices is commonly Nx3).
    # Also ignore the sign, providing a signed array is fine as long as it does not contain negative numbers.
    fmt = primitive.replace("s", "u")

    if fmt in wgpu.IndexFormat:
        return fmt
    else:
        raise ValueError(f"Unexpected index format '{format}'")


def to_vertex_format(format):
    """Convert any pygfx-allowed buffer format to the wgpu.VertexFormat."""
    if format in wgpu.VertexFormat:
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

    fmt = ""
    if len(format) == 2:
        fmt = primitive
    elif len(format) == 4 and format[1] == "x":  # e.g. 3xf4
        if format[0] == "1":
            fmt = primitive
        elif format[0] in "234":
            fmt = primitive + "x" + str(format[0])

    if fmt in wgpu.VertexFormat:
        return fmt
    else:
        raise ValueError(f"Unexpected vertex format '{format}'")


def to_texture_format(format):
    """Convert any pygfx-allowed texture format to the wgpu.TextureFormat."""
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

    fmt = ""
    if len(format) == 2:
        fmt = "r" + primitive
    elif len(format) == 4 and format[1] == "x":  # e.g. 3xf4
        if format[0] == "1":
            fmt = "r" + primitive
        elif format[0] == "2":
            fmt = "rg" + primitive
        elif format[0] == "3":
            fmt = "rgb" + primitive
        elif format[0] == "4":
            fmt = "rgba" + primitive

    if fmt in wgpu.TextureFormat:
        return fmt
    elif fmt.replace("rgb", "rgba") in wgpu.TextureFormat:
        return fmt  # We support rgb by wrapping in rgba textures, see renderers.wgpu/engine/update.py
    else:
        raise ValueError(f"Unexpected texture format '{format}'")


def generate_uniform_struct(dtype_struct, structname):
    """Generate wgsl code from a uniform struct defined with a numpy dtype."""
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


class JsonEncoderWithWgpuSupport(json.JSONEncoder):
    def default(self, ob):
        if isinstance(ob, wgpu.GPUObjectBase):
            return ob.__class__.__name__ + "@" + hex(id(ob))


jsonencoder = JsonEncoderWithWgpuSupport()


def hash_from_value(value):
    """Simple way to create a hash from a (possibly composite) object.
    Assumes JSON encodable objects and GPU objects.
    """
    # Encode the value to string using json. The JSON encoder is so fast that
    # its hard to come up with something that can serialze to str faster.
    s = jsonencoder.encode(value)

    # Return hash (an int). For debugging purposes it can be helpul to return s instead.
    return hash(s)


class GpuCaches:
    """A collection of gpu caches."""

    def get_stats(self):
        """Get a dict mapping cache names to item counts."""
        d = {}
        for name, ob in self.__dict__.items():
            if isinstance(ob, GpuCache):
                d[name] = ob.get_stats()
        return d

    def enable(self):
        """Enable all caches."""
        for ob in self.__dict__.values():
            if isinstance(ob, GpuCache):
                ob.enable()

    def disable(self):
        """Disable all caches."""
        for ob in self.__dict__.values():
            if isinstance(ob, GpuCache):
                ob.disable()


gpu_caches = GpuCaches()


class GpuCache:
    """A cache for GPU objects."""

    def __init__(self, name):
        assert isinstance(name, str)
        assert not hasattr(gpu_caches, name)
        setattr(gpu_caches, name, self)

        self._objects = weakref.WeakValueDictionary()
        self._enabled = True
        self.hits = 0
        self.misses = 0

    def get_stats(self):
        """Get the number of (alive) objects in the cache."""
        return len(list(self._objects.values())), self.hits, self.misses

    def enable(self):
        """Enable this cache."""
        self._enabled = True

    def disable(self):
        """Disable this cache."""
        self._enabled = False

    def get(self, key):
        """Get the cached object or None."""
        if self._enabled:
            try:
                ob = self._objects[key]
            except KeyError:
                ob = None
                self.misses += 1
            else:
                self.hits += 1
        else:
            ob = None
        return ob

    def set(self, key, ob):
        """Store the given object under the given key.
        Note that the cache does not have a (strong) ref to the object.
        """
        self._objects[key] = ob


class GfxSampler:
    """Simple wrapper for a GPUSampler. Should be considered read-only."""

    def __init__(self, filter="nearest", address_mode="clamp"):
        self.filter = filter
        self.address_mode = address_mode
        self._wgpu_object = None


class GfxTextureView:
    """Simple wrapper for a GPUTextureView. Should be considered read-only."""

    def __init__(
        self, texture, *, view_dim=None, layer_range=None, mip_range=None, aspect=None
    ):
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

        # Check mip_range (is half-open range)
        if mip_range is None:
            mip_range = None
        else:
            assert isinstance(mip_range, tuple) and len(mip_range) == 2

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
        self.layer_range = layer_range
        self._mip_range = mip_range
        self.aspect = aspect
        self._wgpu_object = None

    @property
    def mip_range(self):
        # Calculated, since _wgpu_mip_level_count is set later
        if self._mip_range is None:
            return 0, self.texture._wgpu_mip_level_count
        else:
            return self._mip_range
