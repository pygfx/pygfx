"""
Validate send_data
==================

Show an image with points overlaid.
Both the image and points are uploaded with send_data.
"""

# sphinx_gallery_pygfx_docs = 'screenshot'
# sphinx_gallery_pygfx_test = 'compare'

import imageio.v3 as iio
import wgpu
from wgpu.gui.auto import WgpuCanvas, run
import pygfx as gfx
import numpy as np


# Add image

image_data = iio.imread("imageio:astronaut.png")[:, :, 1]

tex = gfx.Texture(
    size=(512, 512, 1),
    dim=2,
    format=wgpu.TextureFormat.r8unorm,
    usage=wgpu.TextureUsage.COPY_DST,
)

tex.send_data((0, 0, 0), image_data)
tex.send_data((80, 100, 0), np.full((50, 50), 255, np.uint8))
tex.send_data((100, 80, 0), np.full((50, 50), 0, np.uint8))
del image_data

image = gfx.Image(
    gfx.Geometry(grid=tex),
    gfx.ImageBasicMaterial(clim=(0, 255)),
)


# Add points

xx = [182, 180, 161, 153, 191, 237, 293, 300, 272, 267, 254]
yy = [145, 131, 112, 59, 29, 14, 48, 91, 136, 137, 172]
position_data = np.array([(x, y, 1) for x, y in zip(xx, yy)], np.float32)

buf = gfx.Buffer(
    nitems=len(xx),
    nbytes=len(xx) * 4 * 3,
    format="3xf4",
    usage=wgpu.BufferUsage.COPY_DST,
)

buf.send_data(0, position_data)
buf.send_data(10, np.array([[125, 105]], np.float32))
del position_data

points = gfx.Points(
    gfx.Geometry(positions=buf),
    gfx.PointsMaterial(color="cyan", size=15),
)


# Setup for rendering

canvas = WgpuCanvas()
renderer = gfx.renderers.WgpuRenderer(canvas)
scene = gfx.Scene()
scene.add(image, points)

camera = gfx.PerspectiveCamera(0)
camera.local.scale_y = -1
camera.show_rect(-10, 522, -10, 522)

controller = gfx.PanZoomController(camera, register_events=renderer)

canvas.request_draw(lambda: renderer.render(scene, camera))

if __name__ == "__main__":
    print(__doc__)
    run()
