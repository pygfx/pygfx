"""
Validate volume slice
=====================

Same as validate_image2.py, but for volume slices.
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
).reshape(1, 4, 4)

# Using different dtypes:
#
# float -> r32float, sampler
# uint8 -> r8unorm, sampler
# uin32 -> r32uint, no sampler

for dy, dtype in enumerate(["float32", "uint8", "uint32"]):
    typed_data = data.astype(dtype)

    for dx, interpolation in enumerate(["nearest", "linear", "cubic"]):
        volume = gfx.Volume(
            gfx.Geometry(grid=gfx.Texture(typed_data, dim=3)),
            gfx.VolumeSliceMaterial(clim=(0, 2), interpolation=interpolation),
        )
        volume.local.position = dx * 5, dy * 5, 0
        scene.add(volume)

# Points go on the first (float / nearest-neighbour interpolated) slice
points = gfx.Points(
    gfx.Geometry(positions=[[0, 0, 1], [3, 3, 1]]),
    gfx.PointsMaterial(color=(0, 1, 0, 1), size=20, aa=True),
)
scene.add(points)

camera = gfx.OrthographicCamera()
camera.show_rect(-1, 14, -1, 14)

canvas.request_draw(lambda: renderer.render(scene, camera))


if __name__ == "__main__":
    print(__doc__)
    loop.run()
