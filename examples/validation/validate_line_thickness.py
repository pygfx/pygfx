"""
Lines with different thicknesses
================================

* The top line is drawn with the thin line material, a classic gl one-pixel line.
* On the left are lines with increasing thickness, without aa, they don't get thinner than one physical pixel.
* On the right are lines with increasing thickness, with aa, really thin lines diminish with alpha.
* Note that due to the set pixel_ratio, a thickness of 2 is 1 physical pixel.
* When dashes are present, they will eventually diminish as the line thickness reduces.
"""

# sphinx_gallery_pygfx_docs = 'screenshot'
# sphinx_gallery_pygfx_test = 'compare'

import numpy as np
from wgpu.gui.auto import WgpuCanvas, run
import pygfx as gfx


canvas = WgpuCanvas(size=(800, 600))
renderer = gfx.WgpuRenderer(canvas, pixel_ratio=0.5, pixel_filter=False)

x = np.linspace(0, 4 * np.pi, 100)
y = np.sin(x)

positions = np.array([x, y, np.zeros_like(x)], np.float32).T.copy()
geometry = gfx.Geometry(positions=positions)

line0 = gfx.Line(
    geometry,
    gfx.LineThinMaterial(color=(1.0, 1.0, 1.0)),
)

scene = gfx.Scene()
scene.add(gfx.Background.from_color("black"))
scene.add(line0)

for x_offset, mode in enumerate(["noaa", "aa", "dashed"]):
    y = 4
    for thickness in [0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]:
        line = gfx.Line(
            geometry,
            gfx.LineMaterial(thickness=thickness, color=(1.0, 1.0, 1.0)),
        )
        y += 2
        line.local.y = -y
        line.local.x = x_offset * 14
        scene.add(line)
        if mode == "noaa":
            line.material.aa = False
        elif mode == "dashed":
            line.material.dash_pattern = [2, 8]

camera = gfx.OrthographicCamera()
camera.show_object(scene, scale=0.7)

canvas.request_draw(lambda: renderer.render(scene, camera))

if __name__ == "__main__":
    print(__doc__)
    run()
