"""
Validate image 2
================

Show an image displayed the correct way.

* The green dots should be at the corners that are darker/brighter.
* The green dots should be in the center of these pixels.
* The darker corner is in the bottom left.

This shows the same image for different interpolation methods,
and is repeated for integer and float data types, because these
result in different shaders (integers other than uint8 cannot use a sampler).
"""

# sphinx_gallery_pygfx_docs = 'screenshot'
# sphinx_gallery_pygfx_test = 'compare'

import numpy as np
from rendercanvas.auto import RenderCanvas, loop
import pygfx as gfx


canvas = RenderCanvas()
renderer = gfx.renderers.WgpuRenderer(canvas)
scene = gfx.Scene()

data = np.array(
    [
        [0, 1, 1, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 2],
    ],
    np.uint8,
)

for dy, dtype in enumerate(["float32", "uint32"]):
    typed_data = data.astype(dtype)

    for dx, interpolation in enumerate(["nearest", "linear", "cubic"]):
        image = gfx.Image(
            gfx.Geometry(grid=gfx.Texture(typed_data, dim=2)),
            gfx.ImageBasicMaterial(clim=(0, 2), interpolation=interpolation),
        )
        image.local.position = dx * 5, dy * 5, 0
        scene.add(image)

# Points go on the first (float / nearest-neighbour interpolated) image
points = gfx.Points(
    gfx.Geometry(positions=[[0, 0, 1], [3, 3, 1]]),
    gfx.PointsMaterial(color=(0, 1, 0, 1), size=20, aa=True),
)
scene.add(points)

camera = gfx.OrthographicCamera()
camera.show_rect(-1, 14, -1, 10)

canvas.request_draw(lambda: renderer.render(scene, camera))


if __name__ == "__main__":
    print(__doc__)
    loop.run()
