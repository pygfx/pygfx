import wgpu

from ....resources import Buffer
from ....utils import logger


from .utils import to_texture_format, GfxSampler, GfxTextureView
from .update import ensure_wgpu_object, ALTTEXFORMAT


visibility_render = wgpu.ShaderStage.VERTEX | wgpu.ShaderStage.FRAGMENT
visibility_all = (
    wgpu.ShaderStage.VERTEX | wgpu.ShaderStage.FRAGMENT | wgpu.ShaderStage.COMPUTE
)


class Binding:
    """Simple object to hold together some information about a binding, for internal use.

    Parameters:
        name: the name in wgsl
        type: "buffer/subtype", "sampler/subtype", "texture/subtype", "storage_texture/subtype".
            The subtype depends on the type:
            BufferBindingType, SamplerBindingType, TextureSampleType, or StorageTextureAccess.
        resource: the Buffer, GfxTextureView, or GfxSampler object.
        visibility: wgpu.ShaderStage flag
        structname: the custom wgsl struct name, if any. otherwise will auto-generate.

    Some tips on terminology:

    * A "binding" is a generic term for an object that defines how a
        resource (buffer or texture) is bound to a shader. In this subpackage it
        likely means a Binding object like this.
    * This binding can be represented with a binding_descriptor and
        binding_layout_descriptor. These are dicts to be passed to wgpu.
    * A list of these binding_layout_descriptor's can be passed to create_bind_group_layout.
    * A list of these binding_layout's can be passed to create_bind_group.
    * Multiple bind_group_layout's can be combined into a pipeline_layout.

    """

    def __init__(
        self, name, type, resource, visibility=visibility_render, structname=None
    ):
        if isinstance(visibility, str):
            visibility = getattr(wgpu.ShaderStage, visibility)
        self.name = name
        self.type = type
        self.resource = resource
        self.visibility = visibility

        self.structname = structname

    def _require_usage_flags(self, resource, usage_flags):
        if resource._wgpu_object is None:
            resource._wgpu_usage |= usage_flags
        elif not (resource._wgpu_usage & usage_flags):
            logger.warning(
                "{resource} requires usage {usage_flags}, but has already been created."
            )

    def get_bind_group_descriptors(self, slot):
        """Get the descriptors (dicts) for creating a binding_descriptor
        and binding_layout_descriptor. A list of these descriptors are
        combined into a bind_group and bind_group_layout.
        """
        resource = self.resource
        subtype = self.type.partition("/")[2]

        if self.type.startswith("buffer/"):
            assert isinstance(resource, Buffer)
            usage_flags = 0
            if "uniform" in self.type:
                usage_flags |= wgpu.BufferUsage.UNIFORM
            elif "storage" in self.type:
                usage_flags |= wgpu.BufferUsage.STORAGE
                if "indices" in self.name:
                    usage_flags |= wgpu.BufferUsage.INDEX
                else:
                    usage_flags |= wgpu.BufferUsage.VERTEX
            self._require_usage_flags(resource, usage_flags)
            binding = {
                "binding": slot,
                "resource": {
                    "buffer": ensure_wgpu_object(resource),
                    "offset": 0,
                    "size": resource.nbytes,
                },
            }
            # Determine min_binding_size. We can set it to None, so it won't do
            # an early check, causing runtime checks to be needed. If we want to
            # specify it, the needed value is different for shader array
            # variables that have a static length vs ones that have a length
            # defined at runtime. Then there's caching, which isn't really used
            # well when min_binding_size jumps around. On the other hand, in early
            # versions we had a comment here that suggested that the caching
            # caused re-use of binding layouts, but them not being compatible.
            # So let's try this: set the min_binding_size for uniform arrays,
            # and set to default for storage buffers.
            # Also see https://github.com/pygfx/pygfx/issues/855
            if "uniform" in self.type:
                min_binding_size = resource.nbytes
            else:
                min_binding_size = None  # None or resource.itemsize
            binding_layout = {
                "binding": slot,
                "visibility": self.visibility,
                "buffer": {
                    "type": getattr(wgpu.BufferBindingType, subtype),
                    "has_dynamic_offset": False,
                    "min_binding_size": min_binding_size,
                },
            }
        elif self.type.startswith("sampler/"):
            assert isinstance(resource, GfxSampler)
            binding = {
                "binding": slot,
                "resource": ensure_wgpu_object(resource),
            }
            binding_layout = {
                "binding": slot,
                "visibility": self.visibility,
                "sampler": {
                    "type": getattr(wgpu.SamplerBindingType, subtype),
                },
            }
        elif self.type.startswith("texture/"):
            assert isinstance(resource, GfxTextureView)
            self._require_usage_flags(
                resource.texture, wgpu.TextureUsage.TEXTURE_BINDING
            )
            binding = {
                "binding": slot,
                "resource": ensure_wgpu_object(resource),
            }
            dim = resource.view_dim
            dim = getattr(wgpu.TextureViewDimension, dim, dim)
            sample_type = getattr(wgpu.TextureSampleType, subtype, subtype)
            # Derive sample type from texture
            if sample_type == "auto":
                fmt = to_texture_format(resource.format)
                fmt = ALTTEXFORMAT.get(fmt, [fmt])[0]
                if "float" in fmt or "norm" in fmt:
                    sample_type = wgpu.TextureSampleType.float
                    # For float32 wgpu does not allow the sampler to be filterable,
                    # except when the native-only feature
                    # TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES is set,
                    # which wgpu-py does by default.
                    # if "32float" in fmt:
                    #     sample_type = wgpu.TextureSampleType.unfilterable_float
                elif "uint" in fmt:
                    sample_type = wgpu.TextureSampleType.uint
                elif "sint" in fmt:
                    sample_type = wgpu.TextureSampleType.sint
                elif "depth" in fmt:
                    sample_type = wgpu.TextureSampleType.depth
                else:
                    raise ValueError("Could not determine texture sample type.")
            binding_layout = {
                "binding": slot,
                "visibility": self.visibility,
                "texture": {
                    "sample_type": sample_type,
                    "view_dimension": dim,
                    "multisampled": False,
                },
            }
        elif self.type.startswith("storage_texture/"):
            assert isinstance(resource, GfxTextureView)
            self._require_usage_flags(
                resource.texture, wgpu.TextureUsage.STORAGE_BINDING
            )
            binding = {
                "binding": slot,
                "resource": ensure_wgpu_object(resource),
            }
            dim = resource.view_dim
            dim = getattr(wgpu.TextureViewDimension, dim, dim)
            fmt = to_texture_format(resource.format)
            fmt = ALTTEXFORMAT.get(fmt, [fmt])[0]
            binding_layout = {
                "binding": slot,
                "visibility": self.visibility,
                "storage_texture": {
                    "access": getattr(wgpu.StorageTextureAccess, subtype),
                    "format": fmt,
                    "view_dimension": dim,
                },
            }
        elif self.type.startswith("shadow_texture/"):
            # a shadow_texture's resource is wgpu.GPUTextureView
            assert isinstance(resource, wgpu.GPUTextureView)
            binding = {"binding": slot, "resource": resource}

            binding_layout = {
                "binding": slot,
                "visibility": wgpu.ShaderStage.FRAGMENT,
                "texture": {
                    "sample_type": wgpu.TextureSampleType.depth,
                    "view_dimension": subtype,
                    "multisampled": False,
                },
            }
        elif self.type.startswith("shadow_sampler/"):
            # a shadow_sampler's resource is wgpu.GPUSampler
            assert isinstance(resource, wgpu.GPUSampler)
            binding = {"binding": slot, "resource": resource}

            binding_layout = {
                "binding": slot,
                "visibility": wgpu.ShaderStage.FRAGMENT,
                "sampler": {"type": wgpu.SamplerBindingType.comparison},
            }
        else:
            raise RuntimeError(f"Unexpected binding type: '{self.type}'")

        return binding, binding_layout
