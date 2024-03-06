"""
Simple Colormap
===============

Show an image with a simple colormap.

* You should see two rectangles with a colormap.
* The top goes smoothly from red to green to blue.
* The bottom shows three equally thick bands (red, green blue).
"""

# test_example = true
# sphinx_gallery_pygfx_render = True

import numpy as np
from wgpu.gui.auto import WgpuCanvas, run
import pygfx as gfx


renderer = gfx.renderers.WgpuRenderer(WgpuCanvas(size=(600, 600)))

im = np.repeat(np.linspace(0, 1, 100).reshape(1, -1), 48, 0).astype(np.float32)
colormap = gfx.Texture(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], np.float32), dim=1)

image1 = gfx.Image(
    gfx.Geometry(grid=gfx.Texture(im, dim=2)),
    gfx.ImageBasicMaterial(clim=(0, 1), map=colormap, map_interpolation="nearest"),
)

image2 = gfx.Image(
    gfx.Geometry(grid=gfx.Texture(im, dim=2)),
    gfx.ImageBasicMaterial(clim=(0, 1), map=colormap, map_interpolation="linear"),
)
image2.local.y = 52

scene = gfx.Scene()
scene.add(image1, image2)

camera = gfx.OrthographicCamera()
camera.show_rect(-4, 103, -4, 103)

renderer.request_draw(lambda: renderer.render(scene, camera))


if __name__ == "__main__":
    run()
