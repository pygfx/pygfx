"""
Image Material
==============

Show an image displayed the correct way.

* The green dots should be at the corners that are darker/brighter.
* The green dots should be in the center of these pixels.
* The darker corner is in the bottom left.
"""

# sphinx_gallery_pygfx_docs = 'screenshot'
# sphinx_gallery_pygfx_test = 'compare'

import numpy as np
from wgpu.gui.auto import WgpuCanvas, run
import pygfx as gfx


canvas = WgpuCanvas()
renderer = gfx.renderers.WgpuRenderer(canvas)
scene = gfx.Scene()

im = np.array(
    [
        [0, 1, 1, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 2],
    ],
    np.float32,
)


image = gfx.Image(
    gfx.Geometry(grid=gfx.Texture(im, dim=2)),
    gfx.ImageBasicMaterial(clim=(0, 2), interpolation="nearest"),
)
scene.add(image)

points = gfx.Points(
    gfx.Geometry(positions=[[0, 0, 1], [3, 3, 1]]),
    gfx.PointsMaterial(color=(0, 1, 0, 1), size=20),
)
scene.add(points)

camera = gfx.OrthographicCamera(10, 10)

canvas.request_draw(lambda: renderer.render(scene, camera))


if __name__ == "__main__":
    print(__doc__)
    run()
