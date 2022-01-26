"""
Show an image with a simple colormap.

* You should see a square image with 3 equally sized vertical bands.
* The bands should be red, green, and blue.
"""

import numpy as np
from wgpu.gui.auto import WgpuCanvas, run
import pygfx as gfx


canvas = WgpuCanvas()
renderer = gfx.renderers.WgpuRenderer(canvas)
scene = gfx.Scene()

im = np.repeat(np.linspace(0, 1, 99).reshape(1, -1), 99, 0).astype(np.float32)

colormap_data = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], np.float32)
colormap = gfx.Texture(colormap_data, dim=1).get_view(filter="nearest")

image = gfx.Image(
    gfx.Geometry(grid=gfx.Texture(im, dim=2)),
    gfx.ImageBasicMaterial(clim=(0, 1), map=colormap),
)
scene.add(image)

camera = gfx.OrthographicCamera(99, 99)
camera.position.set(50, 50, 0)


if __name__ == "__main__":
    canvas.request_draw(lambda: renderer.render(scene, camera))
    run()
