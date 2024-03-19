"""
Line Drawing Performance
========================

Display a line depicting a noisy signal consisting of a lot of points.
"""

# sphinx_gallery_pygfx_docs = 'screenshot'
# sphinx_gallery_pygfx_test = 'run'

import numpy as np
from wgpu.gui.auto import WgpuCanvas, run
import pygfx as gfx


canvas = WgpuCanvas()
renderer = gfx.WgpuRenderer(canvas)

scene = gfx.Scene()

x = np.linspace(0, 100, 10_000, dtype=np.float32)
y = np.sin(x) * 30 + np.random.normal(0, 5, len(x)).astype(np.float32)

positions = np.column_stack([x, y, np.zeros_like(x)])
line = gfx.Line(
    gfx.Geometry(positions=positions),
    gfx.LineMaterial(thickness=2.0, color=(0.0, 0.7, 0.3, 1.0)),
)
scene.add(line)

camera = gfx.OrthographicCamera(110, 110)
camera.local.position = (50, 0, 0)


if __name__ == "__main__":
    canvas.request_draw(lambda: renderer.render(scene, camera))
    run()
