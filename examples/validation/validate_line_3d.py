"""
Cubes drawn with dashed lines
=============================

* The red cubes have thickness in screen space, and should all look the same.
* The blue cubes have thickness in world space, and should look the same.
* The green cubes have thickness in model space, and should have increasing thickness.
"""

import numpy as np
from wgpu.gui.auto import WgpuCanvas, run
import pygfx as gfx


canvas = WgpuCanvas(size=(1000, 1000))
renderer = gfx.WgpuRenderer(canvas)
renderer.blend_mode = "weighted"

positions = [
    [0, 0, 0],
    [1, 1, 1],
    [2, -0.5, 1],
    [3, 0, 0],
    [3, 0, 1],
    [4, 0, 1],
    [4, 0, 1],
    [5, 0, 2],
]
colors = [
    [1, 0, 0],
    [1, 1, 0],
    [0, 1, 0],
    [0, 1, 1],
    [0, 0, 1],
    [1, 0, 1],
    [1, 0, 0],
    [1, 1, 0],
]

positions = np.array(positions, np.float32)
colors = np.array(colors, np.float32)

geometry = gfx.Geometry(positions=positions, colors=colors)

line1 = gfx.Line(
    geometry,
    gfx.LineDashedMaterial(
        thickness=40,
        thickness_space="screen",
        color_mode="uniform",
        dash_pattern=[0, 1.1],
        color=(1.0, 0.0, 0.0),
        opacity=0.5,
    ),
)

line2 = gfx.Line(
    geometry,
    gfx.materials.LineMaterial(
        thickness=40,
        thickness_space="screen",
        color_mode="vertex",
        opacity=1.0,
    ),
)

line3 = gfx.Line(
    geometry,
    gfx.materials._line.LineDebugMaterial(
        thickness=100,
        thickness_space="screen",
        color=(0.0, 1.0, 1.0),
        opacity=0.5,
    ),
)

line1.local.position = 0, 4, 0
line3.local.position = 0, -4, 0

scene = gfx.Scene()
scene.add(line1, line2, line3)  # , middle, bottom)


camera = gfx.PerspectiveCamera(50)
camera.show_object(scene)

controller = gfx.OrbitController(camera, register_events=renderer)


def animate():
    renderer.render(scene, camera)
    for ob in scene.iter():
        if isinstance(ob, gfx.Line):
            if hasattr(ob.material, "dash_offset"):
                ob.material.dash_offset += 0.5


if __name__ == "__main__":
    canvas.request_draw(animate)
    run()
