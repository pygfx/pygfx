"""

Physical Color
==============

PyGfx by default assumes that all colors are in the sRGB colorspace.
This example shows how you can also provide colors in physical colorspace
(a.k.a. linear-srgb). This example shows 3 images:
* A normal image in sRGB.
* An image in physical colorspace, shows up too dark.
* An image in physical colorspace, rendered correctly.
"""

import imageio.v3 as iio
from wgpu.gui.auto import WgpuCanvas, run
import pygfx as gfx
import numpy as np

canvas = WgpuCanvas()
renderer = gfx.renderers.WgpuRenderer(canvas)
scene = gfx.Scene()


def read_srgb_color_image():
    return iio.imread("imageio:astronaut.png").astype(np.float32) / 255


def read_physical_color_image():
    # We fake it by reading a simple srgb image and gamma decoding it
    return read_srgb_color_image() ** 2.2


im1 = read_srgb_color_image()
im2 = read_physical_color_image()

material = gfx.ImageBasicMaterial(clim=(0, 1))

image1 = gfx.Image(
    gfx.Geometry(grid=gfx.Texture(im1, dim=2)),
    material,
)
image1.position.x = 0

image2 = gfx.Image(
    gfx.Geometry(grid=gfx.Texture(im2, dim=2)),
    material,
)
image2.position.x = 550

image3 = gfx.Image(
    gfx.Geometry(grid=gfx.Texture(im2, dim=2, colorspace="physical")),
    material,
)
image3.position.x = 1100

scene.add(image1, image2, image3)

camera = gfx.OrthographicCamera(1650, 550)
camera.position.set(805, 256, 0)
camera.scale.y = -1


if __name__ == "__main__":
    canvas.request_draw(lambda: renderer.render(scene, camera))
    run()
