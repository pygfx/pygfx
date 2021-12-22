"""
Display a lot of line objects. Because of the architecture of wgpu,
this is still performant.
"""

import time  # noqa

import numpy as np
from wgpu.gui.auto import WgpuCanvas, run
import pygfx as gfx


canvas = WgpuCanvas()
renderer = gfx.WgpuRenderer(canvas, show_fps=True)

canvas._target_fps = 9999  # don't limit fps

scene = gfx.Scene()

# Define number of vertices
cols = 20
rows = 50
nvertices = 30000
use_thin_lines = True

print(nvertices * rows * cols, "vertices in total")

x = np.linspace(0.05, 0.95, nvertices, dtype=np.float32)

for row in range(rows):
    for col in range(cols):
        y = np.sin(x * 25) * 0.45 + np.random.normal(0, 0.02, len(x)).astype(np.float32)
        positions = np.column_stack([x, y, np.zeros_like(x)])
        geometry = gfx.Geometry(positions=positions)
        if use_thin_lines:
            material = gfx.LineThinMaterial(color=(col / cols, row / rows, 0.5, 1.0))
        else:
            material = gfx.LineMaterial(
                thickness=0.2 + 2 * row / rows, color=(col / cols, row / rows, 0.5, 1.0)
            )
        line = gfx.Line(geometry, material)
        line.position.x = col
        line.position.y = row
        scene.add(line)

camera = gfx.OrthographicCamera(cols, rows)
camera.maintain_aspect = False
controls = gfx.PanZoomControls(camera.position.clone())
controls.pan(gfx.linalg.Vector3(cols / 2, rows / 2, 0))
controls.add_default_event_handlers(canvas, camera)


def animate():
    controls.update_camera(camera)
    t0 = time.perf_counter()  # noqa
    renderer.render(scene, camera)
    # print(time.perf_counter() - t0)
    canvas.request_draw()


if __name__ == "__main__":
    canvas.request_draw(animate)
    run()
