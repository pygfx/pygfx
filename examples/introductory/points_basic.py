"""
Rendering Points
================


Render Points
"""

# sphinx_gallery_pygfx_docs = 'screenshot'
# sphinx_gallery_pygfx_test = 'run'

import numpy as np
from wgpu.gui.auto import WgpuCanvas, run
import pygfx as gfx


canvas = WgpuCanvas()
renderer = gfx.WgpuRenderer(canvas)

scene = gfx.Scene()

positions = np.random.normal(0, 0.5, (100, 3)).astype(np.float32)
sizes = np.random.rand(100).astype(np.float32) * 50
colors = np.random.rand(100, 4).astype(np.float32)
geometry = gfx.Geometry(positions=positions, sizes=sizes, colors=colors)

material = gfx.PointsMaterial(color_mode="vertex", size_mode="vertex")
points = gfx.Points(geometry, material)
scene.add(points)

scene.add(gfx.Background.from_color((0.2, 0.0, 0, 1), (0, 0.0, 0.2, 1)))

camera = gfx.NDCCamera()


if __name__ == "__main__":
    canvas.request_draw(lambda: renderer.render(scene, camera))
    run()
