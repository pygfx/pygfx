"""
Points with different sizes
===========================

* On the left are points with increasing size, without aa, they don't get smaller than one physical pixel.
* On the right are poonts with increasing size, with aa, really small points diminish with alpha.
* Note that due to the set pixel_ratio, a thickness of 2 is 1 physical pixel.
* The bottom row demonstrates per-vertex sizing.
"""

# sphinx_gallery_pygfx_docs = 'screenshot'
# sphinx_gallery_pygfx_test = 'compare'

import numpy as np
from wgpu.gui.auto import WgpuCanvas, run
import pygfx as gfx


canvas = WgpuCanvas(size=(800, 600))
renderer = gfx.WgpuRenderer(canvas, pixel_ratio=0.5, pixel_filter=False)

x = np.linspace(0, 4 * np.pi, 20)
y = np.sin(x)

positions = np.array([x, y, np.zeros_like(x)], np.float32).T.copy()
geometry = gfx.Geometry(positions=positions, sizes=(10 - 4 * (positions[:, 1] + 1)))

scene = gfx.Scene()

for aa in [False, True]:
    y = 4
    for size in [0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16, 0]:
        line = gfx.Points(
            geometry,
            gfx.PointsMaterial(size=size, color=(1.0, 1.0, 1.0), aa=aa),
        )
        y += 2
        line.local.y = -y
        line.local.x = [-8, 8][aa]
        scene.add(line)
        if size == 0:
            line.material.size_mode = "vertex"

camera = gfx.OrthographicCamera()
camera.show_object(scene, scale=0.7)

canvas.request_draw(lambda: renderer.render(scene, camera))

if __name__ == "__main__":
    print(__doc__)
    run()
