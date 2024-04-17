"""
Colormap Image
==============

Example demonstrating different colormap dimensions on an image.
"""

# sphinx_gallery_pygfx_docs = 'screenshot'
# sphinx_gallery_pygfx_test = 'run'

import numpy as np
import imageio.v3 as iio
from wgpu.gui.auto import WgpuCanvas, run
import pygfx as gfx


canvas = WgpuCanvas(size=(900, 400))
renderer = gfx.renderers.WgpuRenderer(canvas)
scene = gfx.Scene()
camera = gfx.OrthographicCamera(1800, 550)
camera.local.y = 256
camera.local.scale_y = -1
camera.local.x = 1736 / 2


# === 1D colormap
#
# For the image data we use the green channel of a common image. For
# the colormap data we use a colormap that goes from red to green to
# blue (1D). A classic image colormap use-case.

img_data1 = iio.imread("imageio:astronaut.png")[:, :, 1]

colormap1 = gfx.cm.magma
image1 = gfx.Image(
    gfx.Geometry(grid=gfx.Texture(img_data1, dim=2)),
    gfx.ImageBasicMaterial(clim=(0, 255), map=colormap1),
)

image1.local.x = 0
scene.add(image1)


# === 2D colormap
#
# For the image data we use a mesh with two channels, which represent
# the x and y coordinate in the colormap. The mesh simply runs from 0
# to 1, but we add a little swirl to it. For the colormap data we use
# the astronaut image again (2D). Note that while that image was
# previously used as image data, it is now used as a colormap.

xx, yy = np.meshgrid(np.linspace(0, 1, 512), np.linspace(0, 1, 512))
xx += np.sin(yy * 10) * 0.1
img_data2 = np.stack([xx, yy], 2).astype(np.float32)
colormap_data2 = img_data1

colormap2 = gfx.Texture(colormap_data2, dim=2)
image2 = gfx.Image(
    gfx.Geometry(grid=gfx.Texture(img_data2, dim=2)),
    gfx.ImageBasicMaterial(clim=(0, 1), map=colormap2),
)

image2.local.x = 612
scene.add(image2)


# === 3D colormap
#
# For the image data we use a mesh with 3 channels that represent
# positions in 3D, forming a cylinder. For the colormap we used a CT
# volume (3D). In effect we get an image that shows a rolled-out
# cylindrical slice through the volume.

rr, zz = np.meshgrid(np.linspace(0, 2 * np.pi, 512), np.linspace(0, 1, 512))
xx = np.sin(rr) * 0.4 + 0.5
yy = np.cos(rr) * 0.4 + 0.5
img_data3 = np.stack([xx, yy, zz], 2).astype(np.float32)
colormap_data3 = iio.imread("imageio:stent.npz").astype(np.float32) / 1500

colormap3 = gfx.Texture(colormap_data3, dim=3)
image3 = gfx.Image(
    gfx.Geometry(grid=gfx.Texture(img_data3, dim=2)),
    gfx.ImageBasicMaterial(clim=(0, 1), map=colormap3),
)

image3.local.x = 1224
scene.add(image3)


def animate():
    renderer.render(scene, camera)
    canvas.request_draw()


if __name__ == "__main__":
    canvas.request_draw(animate)
    run()
