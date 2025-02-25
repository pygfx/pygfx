"""
Video YUV
=========

This example demonstrates how to show video data in a very efficient way,
eliminating as much data-copies and color conversion operations as we can.

Reading
-------

Many videos are encoded using yuv420p. Therefore, loading the video in
that format, instead of requesting rgb frames, saves computation and
data copies upon reading.

Uploading
---------

The YUV data can also be uploaded directly to the texture, albeit in
three chunks. RGBA frames can be send in one chunk (although it is
considerably larger). For RGB data, the data must first be converted
to RGBA, which is costly. (This packing is normally done automatically
by the pygfx Texture).

Size
----

Frames in yuv420p format are half the size of RGB frames (and even less
compared to RGBA) so the upload itself is cheaper, and memory
consumption on the CPU and GPU are reduced.

Implementation details
----------------------

We demonstrate 2 different storage formats for planar YUV textures:

1. In the first, we use a 2-layer grayscale texture. The u and v planes
   fit in a single layer since they're half the size (in both dimensions).
2. In the second, we use 3 different grayscale textures. The u and v textures
   are half the size in each dimension making for more efficent storage.

While the differences are subtle, the first option reduces the overall
total number of Textures that are maintained in the scene, while the second
can save some storage space on the GPU.

Benchmarks
----------

This script can be done to benchmark for different formats. We've performed
benchmarks against videos encoded with yuv420p and we've found that
that using yuv420p data is certainly the fastest. Using rgba is a bit
slower, although it consumes more memory, so its disadvantage may
increase for heavy workloads. Reading the data as RGB has terrible
performance. We believe that this is caused by the fact that RGB images must
be converted to RGBA pior to uploading them to the GPU.
This is unfortunate, because it's the format that many
default to.
See https://github.com/pygfx/pygfx/pull/873 for details.


By: Mark Harfouche and Almar Klein
Date: Nov 2024
"""

# sphinx_gallery_pygfx_docs = 'screenshot'
# sphinx_gallery_pygfx_test = 'run'

import os
import time

import wgpu
import numpy as np
from rendercanvas.auto import RenderCanvas, loop
from rendercanvas.offscreen import OffscreenRenderCanvas
import pygfx as gfx
import imageio
import av


if "PYTEST_CURRENT_TEST" not in os.environ:
    import argparse

    parser = argparse.ArgumentParser(description="Video YUV Demo")
    parser.add_argument(
        "--format",
        type=str,
        default="yuv420p",
        help=(
            "The format in which the data will be decoded out of FFMPEG. "
            "Choose from 'rgb24', 'rgba', 'yuv420p', 'yuv444p'."
        ),
    )
    parser.add_argument(
        "--filename",
        type=str,
        help="Filename of the video file. If unset, we will use a stock video from imageio.",
    )
    parser.add_argument(
        "--colorrange",
        type=str,
        default="limited",
        help="Choose from 'limited', 'full'. Only valid for yuv420p and yuv444p",
    )
    parser.add_argument(
        "--offscreen",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Render to offscreen canvas. This can be used to benchmark the shader pipeline performance. "
            "If False, the frame rate's upper bound will be limited to that of the "
            "GUI framework's. Typically this is 30, 60, or 120 fps."
        ),
    )
    parser.add_argument(
        "--three-grid-yuv",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use three distinct grids for YUV components.",
    )
    args = parser.parse_args()
    FORMAT = args.format
    COLORRANGE = args.colorrange
    OFFSCREEN = args.offscreen
    THREE_GRID_YUV = args.three_grid_yuv

    if args.filename:
        FILENAME = args.filename
    else:
        FILENAME = None
else:
    OFFSCREEN = False
    COLORRANGE = "limited"
    FORMAT = "yuv444p"
    THREE_GRID_YUV = False
    FILENAME = None


if FILENAME is None:
    # yuv444p
    if FORMAT == "yuv444p":
        FILENAME = imageio.core.Request("imageio:cockatoo.mp4", "r?").filename
    else:  # FORMAT in ["yuv420p", "yuv420p-3plane"]:
        FILENAME = imageio.core.Request("imageio:cockatoo_yuv420.mp4", "r?").filename


print(f"Reading video from {FILENAME}")
print(f"Format: {FORMAT}")
print(f"Color range: {COLORRANGE}")
print(f"Offscreen: {OFFSCREEN}")
print(f"Three grid YUV: {THREE_GRID_YUV}")


def video_width_height():
    with av.open(FILENAME) as container:
        return (container.streams[0].width, container.streams[0].height)


def benchmark_video_read():
    start_time = time.perf_counter()
    with av.open(FILENAME) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        frames_read = 0
        for packet in container.demux(stream):
            for frame in stream.decode(packet):
                if FORMAT != frame.format.name:
                    frame = frame.reformat(format=FORMAT)
                frame.to_ndarray()
                frames_read += 1

            if frames_read > 100:
                break
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    frames_per_second = frames_read / elapsed_time
    return frames_per_second


if "PYTEST_CURRENT_TEST" not in os.environ:
    # A mini benchmark to show the limits of just reading in the video from storage
    print(
        f"Reading video in {FORMAT} format: {benchmark_video_read():.2f} frames per second"
    )


def loop_video():
    while True:
        with av.open(FILENAME) as container:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            for packet in container.demux(stream):
                for frame in stream.decode(packet):
                    # Reformat if necessary. If format is the same, reformat is a no-op.
                    # Otherwise, this includes a data copy and some computations.
                    # In the case of yuv420 -> rgb24, the image also takes more memory.
                    if FORMAT != frame.format.name:
                        frame = frame.reformat(format=FORMAT)
                    # Cast to numpy array. This should not involve a data copy.
                    yield frame.to_ndarray()


w, h = video_width_height()
frame_generator = loop_video()

# Create image object

if FORMAT == "yuv420p" and THREE_GRID_YUV:
    tex = gfx.Texture(
        size=(w, h),
        dim=2,
        colorspace="yuv420p",
        colorrange=COLORRANGE,
        format="r8unorm",
        usage=wgpu.TextureUsage.COPY_DST,
    )
    u_tex = gfx.Texture(
        size=(w // 2, h // 2),
        dim=2,
        colorspace="yuv420p",
        colorrange=COLORRANGE,
        format="r8unorm",
        usage=wgpu.TextureUsage.COPY_DST,
    )
    v_tex = gfx.Texture(
        size=(w // 2, h // 2),
        dim=2,
        colorspace="yuv420p",
        colorrange=COLORRANGE,
        format="r8unorm",
        usage=wgpu.TextureUsage.COPY_DST,
    )
elif FORMAT == "yuv420p":  # and not three_plane_yuv
    # For planar yuv420, we use a 2-layer grayscale texture. The u and v planes
    # fit in a single layer since they're half the size (in both dimensions).
    tex = gfx.Texture(
        size=(w, h, 2),
        dim=2,
        colorspace="yuv420p",
        colorrange=COLORRANGE,
        format="r8unorm",
        usage=wgpu.TextureUsage.COPY_DST,
    )
elif FORMAT == "yuv444p" and THREE_GRID_YUV:
    tex = gfx.Texture(
        size=(w, h),
        dim=2,
        colorspace="yuv444p",
        colorrange=COLORRANGE,
        format="r8unorm",
        usage=wgpu.TextureUsage.COPY_DST,
    )
    u_tex = gfx.Texture(
        size=(w, h),
        dim=2,
        colorspace="yuv444p",
        colorrange=COLORRANGE,
        format="r8unorm",
        usage=wgpu.TextureUsage.COPY_DST,
    )
    v_tex = gfx.Texture(
        size=(w, h),
        dim=2,
        colorspace="yuv444p",
        colorrange=COLORRANGE,
        format="r8unorm",
        usage=wgpu.TextureUsage.COPY_DST,
    )
elif FORMAT == "yuv444p":
    tex = gfx.Texture(
        size=(w, h, 3),
        dim=2,
        colorspace="yuv444p",
        colorrange=COLORRANGE,
        format="r8unorm",
        usage=wgpu.TextureUsage.COPY_DST,
    )
else:
    # For rgb/rgba we use an rgba texture (there is no rgb texture format).
    tex = gfx.Texture(
        size=(w, h, 1),
        dim=2,
        colorspace="srgb",
        format="rgba8unorm",
        usage=wgpu.TextureUsage.COPY_DST,
    )

im = gfx.Image(
    gfx.Geometry(grid=tex),
    gfx.ImageBasicMaterial(clim=(0, 255), interpolation="nearest"),
)
if FORMAT in ["yuv420p", "yuv444p"] and THREE_GRID_YUV:
    im.geometry.grid_u = u_tex
    im.geometry.grid_v = v_tex

# Setup the rest of the viz

CanvasClass = OffscreenRenderCanvas if OFFSCREEN else RenderCanvas
canvas = CanvasClass(size=(w // 2, h // 2), max_fps=999, vsync=False)
renderer = gfx.renderers.WgpuRenderer(canvas, show_fps=True)
scene = gfx.Scene()
scene.add(im)
camera = gfx.OrthographicCamera(w, h)
camera.local.position = w // 2, h // 2, 0
camera.local.scale_y = -1
controller = gfx.PanZoomController(camera, register_events=renderer)
stats = gfx.Stats(viewport=renderer)


def animate():
    # Get next frame to upload
    data = next(frame_generator)

    if FORMAT == "yuv420p":
        y = data[:h]
        u = data[h : h + h // 4].reshape(h // 2, w // 2)
        v = data[h + h // 4 :].reshape(h // 2, w // 2)
        if THREE_GRID_YUV:
            # All planes are contiguous, so there are zero data copies.
            tex.send_data((0, 0), y)
            u_tex.send_data((0, 0), u)
            v_tex.send_data((0, 0), v)
        else:
            # Send the three planes to the texture.
            # The y-plane goes to the first layer.
            # The u-plane and v-plane go to the second layer, side-by side.
            # Note that the u and v planes are just a quarter of the size
            # of the y-plane.
            tex.send_data((0, 0, 0), y)
            tex.send_data((0, 0, 1), u)
            tex.send_data((w // 2, 0, 1), v)
    elif FORMAT == "yuv444p":
        if THREE_GRID_YUV:
            tex.send_data((0, 0), data[0])
            u_tex.send_data((0, 0), data[1])
            v_tex.send_data((0, 0), data[2])
        else:
            tex.send_data((0, 0, 0), data)
    elif FORMAT == "rgba":
        # The data is already rgba, so we can just send it as one blob.
        # That blob is more than twice the size of the yuv420 data though.
        tex.send_data((0, 0, 0), data)
    else:
        # We need to copy the rgb data to rgba, beause wgpu does not have rgb
        # textures. Note that you can create a texture with rgb data, and then the
        # Texture makes a copy automatically upon upload, but send_data
        # (intentionally) does not support this.
        rgba = np.full((*data.shape[:2], 4), 255, dtype=np.uint8)
        rgba[:, :, :3] = data
        tex.send_data((0, 0, 0), rgba)

    with stats:
        renderer.render(scene, camera, flush=False)
    stats.render()

    canvas.request_draw()


if __name__ == "__main__":
    renderer.request_draw(animate)
    if OFFSCREEN:
        # Just render as fast as we can, for 5 secs, to measure fps
        etime = time.time() + 10
        while time.time() < etime:
            canvas.draw()
    else:
        # Enter normal canvas event loop
        loop.run()
    wgpu.diagnostics.pygfx_adapter_info.print_report()
