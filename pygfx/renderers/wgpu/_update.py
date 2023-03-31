"""
Functions to update resources.
"""

import wgpu

from ._utils import to_texture_format, GfxSampler, GfxTextureView
from ._shared import Shared
from ._mipmapsutil import get_mipmaps_util, get_mip_level_count


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


def update_resource(device, resource, kind):
    if kind == "buffer":
        return update_buffer(device, resource)
    elif kind == "texture":
        return update_texture(device, resource)
    else:
        raise ValueError(f"Invalid resource kind: {kind}")


def update_buffer(device, resource):
    buffer = getattr(resource, "_wgpu_buffer", (-1, None))[1]

    # todo: dispose an old buffer? / reuse an old buffer?

    pending_uploads = resource._pending_uploads
    resource._pending_uploads = []
    bytes_per_item = resource.nbytes // resource.nitems

    # Create buffer if needed
    if buffer is None or buffer.size != resource.nbytes:
        resource._wgpu_usage |= wgpu.BufferUsage.COPY_DST
        buffer = device.create_buffer(size=resource.nbytes, usage=resource._wgpu_usage)

    resource._wgpu_buffer = resource.rev, buffer
    if not pending_uploads:
        return

    # Upload any pending data
    for offset, size in pending_uploads:
        subdata = resource._get_subdata(offset, size)
        # A: map the buffer, writes to it, then unmaps. But we don't offer a mapping API in wgpu-py
        # B: roll data in new buffer, copy from there to existing buffer
        # tmp_buffer = device.create_buffer_with_data(
        #     data=subdata,
        #     usage=wgpu.BufferUsage.COPY_SRC,
        # )
        # boffset, bsize = bytes_per_item * offset, bytes_per_item * size
        # encoder.copy_buffer_to_buffer(tmp_buffer, 0, buffer, boffset, bsize)
        # C: using queue. This may be sugar for B, but it may also be optimized.
        device.queue.write_buffer(
            buffer, bytes_per_item * offset, subdata, 0, subdata.nbytes
        )
        # D: A staging buffer/belt https://github.com/gfx-rs/wgpu-rs/blob/master/src/util/belt.rs


def update_texture(device, resource):
    texture = getattr(resource, "_wgpu_texture", (-1, None))[1]
    pending_uploads = resource._pending_uploads
    resource._pending_uploads = []

    fmt = to_texture_format(resource.format)
    pixel_padding = None
    if fmt in ALTTEXFORMAT:
        fmt, pixel_padding, extra_bytes = ALTTEXFORMAT[fmt]

    needs_mipmaps = resource.generate_mipmaps

    mip_level_count = get_mip_level_count(resource) if needs_mipmaps else 1

    # Create texture if needed
    if texture is None:  # todo: or needs to be replaced (e.g. resized)
        resource._wgpu_usage |= wgpu.TextureUsage.COPY_DST

        usage = resource._wgpu_usage
        if needs_mipmaps is True:
            # current mipmap generation requires RENDER_ATTACHMENT
            usage |= wgpu.TextureUsage.RENDER_ATTACHMENT
            resource._mip_level_count = mip_level_count

        texture = device.create_texture(
            size=resource.size,
            usage=usage,
            dimension=f"{resource.dim}d",
            format=fmt,
            mip_level_count=mip_level_count,
            sample_count=1,  # msaa?
        )  # todo: let resource specify mip_level_count and sample_count

    bytes_per_pixel = resource.nbytes // (
        resource.size[0] * resource.size[1] * resource.size[2]
    )
    if pixel_padding is not None:
        bytes_per_pixel += extra_bytes

    resource._wgpu_texture = resource.rev, texture
    if not pending_uploads:
        return

    # Upload any pending data
    for offset, size in pending_uploads:
        subdata = resource._get_subdata(offset, size, pixel_padding)
        # B: using a temp buffer
        # tmp_buffer = device.create_buffer_with_data(data=subdata,
        #     usage=wgpu.BufferUsage.COPY_SRC,
        # )
        # encoder.copy_buffer_to_texture(
        #     {
        #         "buffer": tmp_buffer,
        #         "offset": 0,
        #         "bytes_per_row": size[0] * bytes_per_pixel,  # multiple of 256
        #         "rows_per_image": size[1],
        #     },
        #     {
        #         "texture": texture,
        #         "mip_level": 0,
        #         "origin": offset,
        #     },
        #     copy_size=size,
        # )
        # C: using the queue, which may be doing B, but may also be optimized,
        #    and the bytes_per_row limitation does not apply here
        device.queue.write_texture(
            {"texture": texture, "origin": offset, "mip_level": 0},
            subdata,
            {"bytes_per_row": size[0] * bytes_per_pixel, "rows_per_image": size[1]},
            size,
        )

        if needs_mipmaps:
            generate_mipmaps(device, texture, fmt, mip_level_count, offset[2])


def generate_mipmaps(device, texture_gpu, format, mip_level_count, base_array_layer=0):
    mipmaps_util = get_mipmaps_util(device)
    mipmaps_util.generate_mipmaps(
        texture_gpu, format, mip_level_count, base_array_layer
    )


def ensure_wgpu_object(resource):
    wgpu_object = getattr(resource, "_wgpu_object", None)
    if wgpu_object is not None:
        return wgpu_object

    device = Shared.get_instance().device

    # elif isinstance(resource, Buffer):
    #
    # elif isinstance(resource, Texture):
    #
    if isinstance(resource, GfxTextureView):
        # todo: ensure_wgpu_object(resource.texture)
        update_texture(device, resource.texture)
        resource.texture._wgpu_object = resource.texture._wgpu_texture[1]
        wgpu_texture = resource.texture._wgpu_object

        if resource.is_default_view:
            resource._wgpu_object = wgpu_texture.create_view()
        else:
            fmt = to_texture_format(resource.format)
            fmt = ALTTEXFORMAT.get(fmt, [fmt])[0]
            resource._wgpu_object = wgpu_texture.create_view(
                format=fmt,
                dimension=resource.view_dim,
                aspect=resource.aspect,
                base_mip_level=0,
                mip_level_count=resource.texture._mip_level_count,  # ??
                base_array_layer=0,
                array_layer_count=resource.texture.size[2],
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
