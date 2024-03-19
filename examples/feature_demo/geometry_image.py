"""
Geometry Image
==============

Show an image.
"""

# sphinx_gallery_pygfx_docs = 'screenshot'
# sphinx_gallery_pygfx_test = 'run'

import imageio.v3 as iio
from wgpu.gui.auto import WgpuCanvas, run
import pygfx as gfx


canvas = WgpuCanvas()
renderer = gfx.renderers.WgpuRenderer(canvas)
scene = gfx.Scene()

im = iio.imread("imageio:astronaut.png")[:, :, 1]

image = gfx.Image(
    gfx.Geometry(grid=gfx.Texture(im, dim=2)),
    gfx.ImageBasicMaterial(clim=(0, 255)),
)
scene.add(image)

camera = gfx.OrthographicCamera(512, 512)
camera.local.position = (256, 256, 0)
camera.local.scale_y = -1


if __name__ == "__main__":
    canvas.request_draw(lambda: renderer.render(scene, camera))
    run()
