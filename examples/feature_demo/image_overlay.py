"""
Image with another image overlaid
=================================

Show an image with another image overlaid with alpha blending.
"""

# sphinx_gallery_pygfx_docs = 'screenshot'
# sphinx_gallery_pygfx_test = 'run'

import numpy as np
import imageio.v3 as iio
from wgpu.gui.auto import WgpuCanvas, run
import pygfx as gfx

canvas = WgpuCanvas()
renderer = gfx.renderers.WgpuRenderer(canvas)
scene = gfx.Scene()

# add image

im = iio.imread("imageio:coffee.png")

image = gfx.Image(
    gfx.Geometry(grid=gfx.Texture(im, dim=2)),
    gfx.ImageBasicMaterial(clim=(0, 255)),
)
scene.add(image)

# make overlay using red values

# empty array for overlay, shape is [nrows, ncols, RGBA]
overlay = np.zeros(shape=(*im.shape[:2], 4), dtype=np.float32)

# set the blue values of some pixels with an alpha > 1
overlay[im[:, :, -1] > 200] = np.array([0.0, 0.0, 1.0, 0.6]).astype(np.float32)

overlay_image = gfx.Image(
    gfx.Geometry(grid=gfx.Texture(overlay, dim=2)), gfx.ImageBasicMaterial(clim=(0, 1))
)

# place on top of image
overlay_image.world.z = 1

scene.add(overlay_image)

# put the original image below
image_original = gfx.Image(
    gfx.Geometry(grid=gfx.Texture(im, dim=2)),
    gfx.ImageBasicMaterial(clim=(0, 255)),
)

image_original.world.y = im.shape[0] + 10

scene.add(image_original)

camera = gfx.PerspectiveCamera(0)
camera.show_object(scene)
camera.local.scale_y = -1
camera.zoom = 1.2

controller = gfx.PanZoomController(camera, register_events=renderer)

if __name__ == "__main__":
    canvas.request_draw(lambda: renderer.render(scene, camera))
    run()
