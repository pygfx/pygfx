"""
Lines in 3D
===========

* This example is to demonstrate lines in 3D, with a perspective camera.
* Dashing and colors should be continuous for non-broken joins (i.e. correct handling of depth and w).
* Lines that are nearly completely in the view direction only have their caps drawn.

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
    gfx.LineDebugMaterial(
        thickness=100,
        thickness_space="screen",
        color=(0.0, 1.0, 1.0),
        opacity=1,
    ),
)

line2 = gfx.Line(
    geometry,
    gfx.LineMaterial(
        thickness=40,
        thickness_space="screen",
        color_mode="vertex",
        opacity=1.0,
    ),
)

line3 = gfx.Line(
    geometry,
    gfx.LineMaterial(
        thickness=40,
        thickness_space="screen",
        color_mode="uniform",
        dash_pattern=[0, 1.1],
        color=(1.0, 0.0, 0.0),
        opacity=0.5,
    ),
)


line1.local.position = 0, 4, 0
line3.local.position = 0, -4, 0

scene = gfx.Scene()
scene.add(line1, line2, line3)  # , middle, bottom)


camera = gfx.PerspectiveCamera(50)
camera.show_object(scene, scale=1.0)

controller = gfx.OrbitController(camera, register_events=renderer)


def animate():
    renderer.render(scene, camera)
    for ob in scene.iter():
        if isinstance(ob, gfx.Line):
            if hasattr(ob.material, "dash_offset"):
                ob.material.dash_offset += 0.02 * line3.material.thickness


if __name__ == "__main__":
    canvas.request_draw(animate)
    run()
