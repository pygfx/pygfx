"""

Directional Shadow 2
====================

This example demonstrates the effects of directional light shadows (from
DirectionalLight) and omnidirectional shadows (from PointLight).
"""

# sphinx_gallery_pygfx_docs = 'screenshot'
# sphinx_gallery_pygfx_test = 'run'

import time
import math
import pylinalg as la

from wgpu.gui.auto import WgpuCanvas, run
import pygfx as gfx
import numpy as np


renderer = gfx.renderers.WgpuRenderer(WgpuCanvas())
scene = gfx.Scene()

cube = gfx.Mesh(
    gfx.box_geometry(20, 20, 20),
    gfx.MeshPhongMaterial(),
)
cube.local.rotation = la.quat_from_euler((math.pi / 6, math.pi / 6), order="XY")
cube.cast_shadow = True
scene.add(cube)

cube2 = gfx.Mesh(
    gfx.box_geometry(50, 50, 50),
    gfx.MeshPhongMaterial(),
)
cube2.local.rotation = la.quat_from_euler((math.pi / 4, math.pi / 4), order="XY")
cube2.local.position = (0, -150, 0)
cube2.cast_shadow = True
cube2.receive_shadow = True
scene.add(cube2)

cube3 = gfx.Mesh(
    gfx.box_geometry(100, 100, 100),
    gfx.MeshPhongMaterial(),
)

cube3.local.position = (0, -250, 0)
cube3.cast_shadow = True
cube3.receive_shadow = True
scene.add(cube3)

t = np.linspace(0, 10, 100).astype(np.float32)
xyz = 20 * np.sin(2 * t), 20 * np.sin(3 * t) + 50, 20 * np.sin(t)
line1 = gfx.Line(
    gfx.Geometry(positions=np.column_stack(xyz)),
    gfx.LineMaterial(color="#088", thickness=5),
)
line1.cast_shadow = True
scene.add(line1)

box = gfx.Mesh(
    gfx.box_geometry(600, 600, 600),
    gfx.MeshPhongMaterial(color="#808080", side="Back"),
)

box.local.rotation = la.quat_from_euler((-math.pi / 2), order="XY")
box.local.position = (0, 0, 0)
box.receive_shadow = True
box.cast_shadow = False
scene.add(box)

ambient = gfx.AmbientLight()

scene.add(ambient)

light = gfx.PointLight("#4040ff", 500000, decay=2)
light.local.x = 15
light.local.y = 20

light.cast_shadow = True
scene.add(light.add(gfx.PointLightHelper(5)))

light2 = gfx.DirectionalLight("#aaaaaa")
light2.local.position = (-150, 100, 100)
light2.cast_shadow = True

scene.add(light2.add(gfx.DirectionalLightHelper(100)))

camera = gfx.PerspectiveCamera(70, 16 / 9)
camera.local.position = (100, 100, 350)
camera.show_pos((0, 0, 0))

controller = gfx.OrbitController(camera, register_events=renderer)


def animate():
    t = time.time() * 0.5
    scale = 150

    light.local.position = (
        math.cos(t) * math.cos(3 * t) * scale,
        math.cos(3 * t) * math.sin(t) * scale / 2,
        math.sin(3 * t) * scale,
    )

    renderer.render(scene, camera)
    renderer.request_draw()


if __name__ == "__main__":
    renderer.request_draw(animate)
    run()
