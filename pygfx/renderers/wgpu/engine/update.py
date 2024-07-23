"""
Functions to update resources.
"""

import wgpu

from ....resources import Texture, Buffer

from .utils import to_texture_format, GfxSampler, GfxTextureView
from .mipmapsutil import get_mip_level_count, generate_texture_mipmaps
from .shared import get_shared


# Alternative texture formats that we support by padding channels as needed.
# Maps virtual_format -> (wgpu_format, pad_value, nbytes)
ALTTEXFORMAT = {
    "rgb8snorm": ("rgba8snorm", 127, 1),
    "rgb8unorm": ("rgba8unorm", 255, 1),
    "rgb8sint": ("rgba8sint", 127, 1),
    "rgb8uint": ("rgba8uint", 255, 1),
    "rgb16sint": ("rgba16sint", 32767, 2),
    "rgb16uint": ("rgba16uint", 65535, 2),
    "rgb32sint": ("rgba32sint", 2147483647, 4),
    "rgb32uint": ("rgba32uint", 4294967295, 4),
    "rgb16float": ("rgba16float", 1, 2),
    "rgb32float": ("rgba32float", 1, 4),
}


def update_resource(resource):
    """Update the contents of a buffer or texture."""
    if isinstance(resource, Buffer):
        return _update_buffer(resource)
    elif isinstance(resource, Texture):
        return _update_texture(resource)
    else:
        raise ValueError(f"Invalid resource type: {resource.__class__.__name__}")


# Note on how buffer and texture updates work:
#
# * When a resource is first created, it's _wgpu_object attribute is None
#   and its _wgpu_flags is unset.
# * When the resource is actually being used somewhere, it will end up
#   in the logic that creates a pipeline object (e.g. in _pipeline.py), which
#   sets the appropriate usage flags (because that code knows how the resource
#   is used) and then uses ensure_wgpu_object to create the object.
# * Resources that need to be synced are tracked in the resource_registry,
#   but only go into it when they have their _wgpu_object set (i.e. when
#   the resource actually exists on the GPU).
# * Right before the renderer performs a draw, it queries the registry
#   and calls update_resource on each.


def _update_buffer(buffer):
    wgpu_buffer = buffer._wgpu_object
    assert wgpu_buffer is not None

    # Collect chunks to update
    chunk_descriptions = buffer._gfx_get_chunk_descriptions()
    if not chunk_descriptions:
        return

    # Prepare for data uploads
    device = get_shared().device
    bytes_per_item = buffer.itemsize

    # Upload any pending data
    for offset, size in chunk_descriptions:
        chunk_data = buffer._gfx_get_chunk_data(offset, size)
        device.queue.write_buffer(
            wgpu_buffer, bytes_per_item * offset, chunk_data, 0, chunk_data.nbytes
        )
        # todo: investigate using a staging buffer/belt https://github.com/gfx-rs/wgpu-rs/blob/master/src/util/belt.rs


def _update_texture(texture):
    wgpu_texture = texture._wgpu_object
    assert wgpu_texture is not None

    # Collect chunks to update
    chunk_descriptions = texture._gfx_get_chunk_descriptions()
    if not chunk_descriptions:
        return

    # Prepare for data uploads
    device = get_shared().device
    fmt = to_texture_format(texture.format)
    pad_value = 0.0
    extra_bytes = 0
    if texture._wgpu_emulate_rgb:
        _, pad_value, extra_bytes = ALTTEXFORMAT[fmt]

    bytes_per_pixel = (
        texture.nbytes // (texture.size[0] * texture.size[1] * texture.size[2])
        + extra_bytes
    )

    # Upload any pending data
    for offset, size in chunk_descriptions:
        chunk_data = texture._gfx_get_chunk_data(offset, size, pad_value)
        device.queue.write_texture(
            {"texture": wgpu_texture, "origin": offset, "mip_level": 0},
            chunk_data,
            {"bytes_per_row": size[0] * bytes_per_pixel, "rows_per_image": size[1]},
            size,
        )

    if texture.generate_mipmaps:
        generate_texture_mipmaps(texture)


def ensure_wgpu_object(resource):
    """Make sure that the resource (buffer texture, sampler, textureView)
    has a wgpu object attached to it. Returns the native wgpu object.
    """
    if resource._wgpu_object is not None:
        return resource._wgpu_object

    device = get_shared().device

    if isinstance(resource, Buffer):
        if resource.data is not None:
            resource._wgpu_usage |= wgpu.BufferUsage.COPY_DST
        resource._wgpu_object = device.create_buffer(
            size=resource.nbytes, usage=resource._wgpu_usage
        )
        # Mark the resource for sync at the registry (but only if it has pending updates)
        resource._gfx_mark_for_sync()

    elif isinstance(resource, Texture):
        fmt = to_texture_format(resource.format)
        if fmt in ALTTEXFORMAT:
            fmt = ALTTEXFORMAT[fmt][0]
        if resource.data is not None:
            resource._wgpu_usage |= wgpu.TextureUsage.COPY_DST
        usage = resource._wgpu_usage
        if resource.generate_mipmaps:
            usage |= wgpu.TextureUsage.STORAGE_BINDING  # mipmap needs this
            resource._wgpu_mip_level_count = get_mip_level_count(resource)
        resource._wgpu_object = device.create_texture(
            size=resource.size,
            usage=usage,
            dimension=f"{resource.dim}d",
            format=fmt,
            mip_level_count=resource._wgpu_mip_level_count,
            sample_count=1,  # could allow more to implement msaa
        )
        # Mark the resource for sync at the registry (but only if it has pending updates)
        resource._gfx_mark_for_sync()

    elif isinstance(resource, GfxTextureView):
        wgpu_texture = ensure_wgpu_object(resource.texture)
        if resource.is_default_view:
            resource._wgpu_object = wgpu_texture.create_view()
        else:
            fmt = to_texture_format(resource.format)
            fmt = ALTTEXFORMAT.get(fmt, [fmt])[0]
            resource._wgpu_object = wgpu_texture.create_view(
                format=fmt,
                dimension=resource.view_dim,
                aspect=resource.aspect,
                base_mip_level=resource.mip_range[0],
                mip_level_count=(resource.mip_range[1] - resource.mip_range[0]),
                base_array_layer=resource.layer_range[0],
                array_layer_count=(resource.layer_range[1] - resource.layer_range[0]),
            )

    elif isinstance(resource, GfxSampler):
        amodes = resource.address_mode.replace(",", " ").split() or ["clamp"]
        while len(amodes) < 3:
            amodes.append(amodes[-1])
        filters = resource.filter.replace(",", " ").split() or ["nearest"]
        while len(filters) < 3:
            filters.append(filters[-1])
        ammap = {"clamp": "clamp-to-edge", "mirror": "mirror-repeat"}
        resource._wgpu_object = device.create_sampler(
            address_mode_u=ammap.get(amodes[0], amodes[0]),
            address_mode_v=ammap.get(amodes[1], amodes[1]),
            address_mode_w=ammap.get(amodes[2], amodes[2]),
            mag_filter=filters[0],
            min_filter=filters[1],
            mipmap_filter=filters[2],
            # lod_min_clamp -> use default 0
            # lod_max_clamp -> use default inf
            # compare -> only not-None for comparison samplers!
        )

    return resource._wgpu_object
