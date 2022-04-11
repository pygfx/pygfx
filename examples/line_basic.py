"""
Some basic line drawing.
"""

import numpy as np
from wgpu.gui.auto import WgpuCanvas, run
import pygfx as gfx


canvas = WgpuCanvas()
renderer = gfx.WgpuRenderer(canvas)
renderer_svg = gfx.SvgRenderer(640, 480, "~/line.svg")

scene = gfx.Scene()
positions = [[200 + np.sin(i) * i * 6, 200 + np.cos(i) * i * 6, 0] for i in range(20)]
positions += [[400 - np.sin(i) * i * 6, 200 + np.cos(i) * i * 6, 0] for i in range(20)]
positions += [
    [450, 400, 0],
    [375, 400, 0],
    [300, 400, 0],
    [400, 370, 0],
    [300, 340, 0],
]

# Spiral away in z (to make the depth buffer less boring)
for i in range(len(positions)):
    positions[i][2] = i

line = gfx.Line(
    gfx.Geometry(positions=positions),
    gfx.LineMaterial(thickness=12.0, color=(0.8, 0.7, 0.0)),
)
scene.add(line)

camera = gfx.OrthographicCamera(600, 500)
camera.position.set(300, 250, 0)

controls = gfx.PanZoomControls(camera.position.clone())
controls.add_default_event_handlers(renderer, camera)


def animate():
    controls.update_camera(camera)
    renderer.render(scene, camera)


if __name__ == "__main__":
    renderer_svg.render(scene, camera)
    canvas.request_draw(animate)
    run()
