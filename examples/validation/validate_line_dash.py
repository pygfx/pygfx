"""
Dashing
=======

* Demoing densely sampled lines with dashing.
* Thicker lines show the same pattern (i.e. dash phase increases with thickness).
* Dash offset is expressed in same space as the pattern.
"""

# sphinx_gallery_pygfx_docs = 'screenshot'
# sphinx_gallery_pygfx_test = 'compare'

import numpy as np
from wgpu.gui.auto import WgpuCanvas, run
import pygfx as gfx


canvas = WgpuCanvas(size=(1000, 1000))
renderer = gfx.WgpuRenderer(canvas)

x = np.linspace(0, 4 * np.pi, 1000)
y = np.sin(x)


positions = np.array([x * 100, y * 100, np.zeros_like(x)], np.float32).T.copy()

geometry = gfx.Geometry(positions=positions)

line1 = gfx.Line(
    geometry,
    gfx.LineMaterial(
        thickness=12,
        dash_pattern=[0, 2, 2, 2],
        color=(0.0, 1.0, 1.0, 0.4),
        dash_offset=0,
    ),
)

line2 = gfx.Line(
    geometry,
    gfx.LineMaterial(
        thickness=12,
        dash_pattern=[0, 2, 2, 2],
        color=(0.0, 1.0, 1.0, 0.4),
        dash_offset=2,
    ),
)
line3 = gfx.Line(
    geometry,
    gfx.LineMaterial(
        thickness=24,
        dash_pattern=[0, 2, 2, 2],
        color=(0.0, 1.0, 1.0, 0.4),
        dash_offset=3,
    ),
)

line4 = gfx.Line(
    geometry,
    gfx.LineMaterial(
        thickness=24,
        dash_pattern=[0, 2, 2, 2],
        color=(0.0, 1.0, 1.0, 0.4),
        dash_offset=6,
    ),
)

for line in (line1, line2, line3, line4):
    line.material.thickness_space = "screen"

line1.local.position = 0, 0, 0
line2.local.position = 0, -500, 0
line3.local.position = 0, -1000, 0
line4.local.position = 0, -1500, 0

scene = gfx.Scene()
scene.add(line1, line2, line3, line4)

camera = gfx.OrthographicCamera()
camera.show_object(scene)

controller = gfx.OrbitController(camera, register_events=renderer)

canvas.request_draw(lambda: renderer.render(scene, camera))

if __name__ == "__main__":
    print(__doc__)
    run()
