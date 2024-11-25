""" """

# sphinx_gallery_pygfx_docs = 'screenshot'
# sphinx_gallery_pygfx_test = 'run'

import time

import wgpu
import numpy as np
from wgpu.gui.auto import WgpuCanvas, run
from wgpu.gui.offscreen import WgpuCanvas as OffscreenCanvas
import pygfx as gfx
import imageio
import av


OFFSCREEN = False  # Set to True if FPS is capped at 60 or 120
FILENAME = imageio.core.Request("imageio:cockatoo.mp4", "r?").filename
FORMAT = "yuv420p"  # Set to yuv420p or rgb24


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
            frame.reformat(format=FORMAT)
            # Cast to numpy array. This should not involve a data copy.
            yield frame.to_ndarray()


w, h = video_width_height()
frame_generator = loop_video()

# Create image object

if FORMAT == "yuv420p":
    # For planar yuv420, we use a 2-layer luminance texture. The u and v planes
    # fit in a single layer since they're half the size (in both dimensions).
    tex = gfx.Texture(
        size=(w, h, 2),
        dim=2,
        colorspace="yuv420p",
        format="r8unorm",
        usage=wgpu.TextureUsage.COPY_DST,
    )
else:
    # For rgb we use an rgba texture (there is no rgb texture format).
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
canvas = CanvasClass(max_fps=999, vsync=False)
renderer = gfx.renderers.WgpuRenderer(canvas, show_fps=True)
scene = gfx.Scene()
scene.add(im)
camera = gfx.OrthographicCamera()
camera.show_object(scene)
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
    else:
        # We need to copy the rgb data to rgba, beause wgpu does not have rgb
        # textures. Note that you can set textures with rgb data, and then the
        # Texture makes a copy automatically upon upload, similar to the below.
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
        etime = time.time() + 5
        while time.time() < etime:
            canvas.draw()
    else:
        # Enter normal canvas event loop
        run()
