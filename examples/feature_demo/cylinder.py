"""
Types of Cylinders
==================

Example showing different types of geometric cylinders.
"""

# sphinx_gallery_pygfx_docs = 'screenshot'
# sphinx_gallery_pygfx_test = 'run'

import numpy as np
from wgpu.gui.auto import WgpuCanvas, run
import pygfx as gfx


canvas = WgpuCanvas()
renderer = gfx.renderers.WgpuRenderer(canvas)
scene = gfx.Scene()

cylinders = [
    ((0, 0, -32.5), (0, 0.65, 0, 1), gfx.cylinder_geometry(10, 10, height=25)),
    (
        (30, 0, 25),
        (1, 0.3, 0.3, 1),
        gfx.cylinder_geometry(
            10,
            10,
            height=12,
            theta_start=np.pi * 1.3,
            theta_length=np.pi * 1.5,
            open_ended=True,
        ),
    ),
    (
        (-50, 0, 0),
        (0, 0, 0.65, 1),
        gfx.cylinder_geometry(
            20, 12, radial_segments=3, height_segments=4, height=10, open_ended=True
        ),
    ),
    ((50, 0, -10), (1, 1, 0.75, 1), gfx.cylinder_geometry(1.5, 1.5, height=20)),
    ((50, 0, 5), (1, 1, 0.75, 1), gfx.cylinder_geometry(4, 0.0, height=10)),
]
for pos, color, geometry in cylinders:
    material = gfx.MeshPhongMaterial(color=color)
    wobject = gfx.Mesh(geometry, material)
    wobject.local.position = pos
    scene.add(wobject)

    material = gfx.MeshNormalLinesMaterial(color=color)
    wobject = gfx.Mesh(geometry, material)
    wobject.local.position = pos
    wobject.cast_shadow = True
    scene.add(wobject)

camera = gfx.PerspectiveCamera(70, 16 / 9)
camera.local.position = (50, 50, 50)
camera.show_pos((0, 0, 0))
controller = gfx.OrbitController(camera, register_events=renderer)

scene.add(gfx.AmbientLight())
light = gfx.PointLight()
light.local.position = (0, 70, 0)
light.add(gfx.PointLightHelper())
light.cast_shadow = True
# since we are shadow mapping open meshes
# disable front face culling to render backfaces to shadow maps
# and set bias to avoid shadow acne
light.shadow.cull_mode = "none"
light.shadow.bias = 0.00001
scene.add(light)

ground = gfx.Mesh(
    gfx.box_geometry(1000, 1, 1000),
    gfx.MeshPhongMaterial(),
)
ground.local.y = -40
ground.receive_shadow = True
scene.add(ground)


if __name__ == "__main__":
    canvas.request_draw(lambda: renderer.render(scene, camera))
    run()
