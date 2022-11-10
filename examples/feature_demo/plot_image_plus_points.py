"""
Image with Points Overlaid
==========================

Show an image with points overlaid.
"""

import imageio.v3 as iio
from wgpu.gui.auto import WgpuCanvas, run
import pygfx as gfx

canvas = WgpuCanvas()
renderer = gfx.renderers.WgpuRenderer(canvas)
scene = gfx.Scene()

# %% add image

im = iio.imread("imageio:astronaut.png")

image = gfx.Image(
    gfx.Geometry(grid=gfx.Texture(im, dim=2)),
    gfx.ImageBasicMaterial(clim=(0, 255)),
)
scene.add(image)

# %% add points

xx = [182, 180, 161, 153, 191, 237, 293, 300, 272, 267, 254]
yy = [145, 131, 112, 59, 29, 14, 48, 91, 136, 137, 172]

points = gfx.Points(
    gfx.Geometry(positions=[(x, y, 0) for x, y in zip(xx, yy)]),
    gfx.PointsMaterial(color=(0, 1, 1, 1), size=10),
)
points.position.z = 1  # move points in front of the image
scene.add(points)

camera = gfx.OrthographicCamera(512, 512)
camera.position.set(256, 256, 0)
camera.scale.y = -1


if __name__ == "__main__":
    canvas.request_draw(lambda: renderer.render(scene, camera))
    run()
