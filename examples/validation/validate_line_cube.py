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

position_pairs = [
    [-1, -1, -1],
    [-1, -1, 1],
    [-1, -1, -1],
    [-1, 1, -1],
    [-1, -1, -1],
    [1, -1, -1],
    [1, 1, -1],
    [1, 1, 1],
    [1, 1, -1],
    [1, -1, -1],
    [1, 1, -1],
    [-1, 1, -1],
    [-1, 1, 1],
    [-1, 1, -1],
    [-1, 1, 1],
    [-1, -1, 1],
    [-1, 1, 1],
    [1, 1, 1],
    [1, -1, 1],
    [1, -1, -1],
    [1, -1, 1],
    [1, 1, 1],
    [1, -1, 1],
    [-1, -1, 1],
]

if True:  # use segment
    Material = gfx.LineSegmentMaterial  # noqa
    positions = np.array(position_pairs, np.float32)
else:  # use nans
    Material = gfx.LineMaterial  # noqa
    positions = np.zeros((3 * len(position_pairs) // 2, 3), np.float32)
    positions[0::3] = position_pairs[0::2]
    positions[1::3] = position_pairs[1::2]
    positions[2::3] = np.nan


def add_three_cubes(parent, material):
    line1 = gfx.Line(
        gfx.Geometry(positions=positions),
        material,
    )
    container1 = gfx.Scene()
    parent.add(container1.add(line1))

    line2 = gfx.Line(
        gfx.Geometry(positions=positions / 2),
        material,
    )
    container2 = gfx.Scene()
    parent.add(container2.add(line2))
    line2.local.scale = 2

    line3 = gfx.Line(
        gfx.Geometry(positions=positions / 3.3333),
        material,
    )
    container3 = gfx.Scene()
    parent.add(container3.add(line3))
    container3.local.scale = 3.3333
    # Positioning
    container1.local.position = -3, 0, 0
    container3.local.position = 3, 0, 0


top = gfx.Scene()  # screen
middle = gfx.Scene()  # world
bottom = gfx.Scene()  # model

top.local.position = 0, 3, 0
bottom.local.position = 0, -3, 0

scene = gfx.Scene()
scene.add(top, middle, bottom)

add_three_cubes(
    top,
    Material(
        thickness=10,
        thickness_space="screen",
        dash_pattern=[0, 2],
        color=(1.0, 0.0, 0.0),
        opacity=0.5,
    ),
)

add_three_cubes(
    middle,
    Material(
        thickness=0.1,
        thickness_space="world",
        dash_pattern=[0, 2],
        color=(0.0, 1.0, 0.0),
        opacity=0.5,
    ),
)

add_three_cubes(
    bottom,
    Material(
        thickness=0.1,
        thickness_space="model",
        dash_pattern=[0, 2],
        color=(0.0, 0.0, 1.0),
        opacity=0.5,
    ),
)

camera = gfx.PerspectiveCamera(90)
camera.show_object(scene)

controller = gfx.OrbitController(camera, register_events=renderer)


if __name__ == "__main__":
    canvas.request_draw(lambda: renderer.render(scene, camera))
    run()
