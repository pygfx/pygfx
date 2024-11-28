"""
Video yuv
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

Benchmarks
----------

This script can be done to benchmark for different formats. We've found
that using yuv420p data is certainly the fastest. Using rgba is a bit
slower, although it consumes more memory, so its disadvantage may
increase for heavy workloads. Reading the data as RGB has terrible
performance. This is unfortunate, because it's the format that many
default to.
See https://github.com/pygfx/pygfx/pull/873 for details.

"""

# sphinx_gallery_pygfx_docs = 'screenshot'
# sphinx_gallery_pygfx_test = 'run'

import os
import time

import wgpu
import numpy as np
from wgpu.gui.auto import WgpuCanvas, run
from wgpu.gui.offscreen import WgpuCanvas as OffscreenCanvas
import pygfx as gfx
import imageio
import av


# Set OFFSCREEN to True if the FPS in the visible Window is capped at 60 or 120.
# The script will then render to an offscreen canvas for a few seconds instead.

# Set the filename of the file. We use imageio to get a stock video.
# TODO: when https://github.com/imageio/imageio/pull/1103 is merged and a new release is out, we can use the yuv420 video instead.
# Local: FILENAME = "/Users/almar/dev/py/imageio-binaries/images/cockatoo_yuv420.mp4"  # yuv420p
# FILENAME = imageio.core.Request("imageio:cockatoo.mp4", "r?").filename  # yuv444p

# Set the format to read the frames in. If this matches the video's storage format,
# no conversion will have to be done. But it also affects how the data is handled in Pygfx.
# Set to "rgb24", "rgba", or "yuv420p"
if "PYTEST_CURRENT_TEST" not in os.environ:
    import argparse

    parser = argparse.ArgumentParser(description="Video YUV Demo")
    parser.add_argument(
        "--format",
        type=str,
        default="yuv420p",
        help="Choose from 'rgb24', 'rgba', 'yuv420p'",
    )
    parser.add_argument(
        "--filename",
        type=str,
        help="Filename of the video file",
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
        help="Render to offscreen canvas",
    )
    args = parser.parse_args()
    FORMAT = args.format
    COLORRANGE = args.colorrange
    OFFSCREEN = args.offscreen

    if args.filename:
        FILENAME = args.filename
    else:
        # yuv444p
        if FORMAT == "yuv444p":
            FILENAME = imageio.core.Request("imageio:cockatoo.mp4", "r?").filename
        else:  # FORMAT == "yuv420p":
            FILENAME = imageio.core.Request(
                "imageio:cockatoo_yuv420.mp4", "r?"
            ).filename
else:
    OFFSCREEN = False
    FORMAT = "yuv420p"
    FILENAME = imageio.core.Request("imageio:cockatoo_yuv420p.mp4", "r?").filename


def video_width_height():
    container = av.open(FILENAME)
    return (container.streams[0].width, container.streams[0].height)


def loop_video():
    while True:
        container = av.open(FILENAME)
        for frame in container.decode(video=0):
            # Reformat if necessary. If format is the same, reformat is a no-op.
            # Otherwise, this includes a data copy and some computations.
            # In the case of yuv420 -> rgb24, the image also takes more memory.
            frame = frame.reformat(format=FORMAT)
            # Cast to numpy array. This should not involve a data copy.
            yield frame.to_ndarray()


w, h = video_width_height()
frame_generator = loop_video()

# Create image object

if FORMAT == "yuv420p":
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
    gfx.ImageBasicMaterial(clim=(0, 255)),
)

# Setup the rest of the viz

CanvasClass = OffscreenCanvas if OFFSCREEN else WgpuCanvas
canvas = CanvasClass(size=(w // 2, h // 2), max_fps=999, vsync=False)
renderer = gfx.renderers.WgpuRenderer(canvas, show_fps=True)
scene = gfx.Scene()
scene.add(im)
camera = gfx.OrthographicCamera(w, h)
camera.local.position = w // 2, h // 2, 0
camera.local.scale_y = -1
stats = gfx.Stats(viewport=renderer)


def animate():
    # Get next frame to upload
    data = next(frame_generator)

    if FORMAT == "yuv420p":
        # Send the three planes to the texture. The y-plane goes to the first layer.
        # The u-plane and v-plane go to the second layer, side-by side.
        # Note that the u and v planes are just a quarter of the size of the y-plane.
        # All planes are contiguous, so there are zero data copies.
        y = data[:h]
        u = data[h : h + h // 4].reshape(h // 2, w // 2)
        v = data[h + h // 4 :].reshape(h // 2, w // 2)
        tex.send_data((0, 0, 0), y)
        tex.send_data((0, 0, 1), u)
        tex.send_data((w // 2, 0, 1), v)
    elif FORMAT == "yuv444p":
        y = data[0]
        u = data[1]
        v = data[2]
        tex.send_data((0, 0, 0), y)
        tex.send_data((0, 0, 1), u)
        tex.send_data((0, 0, 2), v)
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
        run()
