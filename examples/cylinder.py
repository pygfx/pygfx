"""
Example showing different types of geometric cylinders.
"""

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
        (0.35, 0, 0, 1),
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
    wobject.position.set(*pos)
    scene.add(wobject)

    material = gfx.MeshNormalLinesMaterial(color=color)
    wobject = gfx.Mesh(geometry, material)
    wobject.position.set(*pos)
    scene.add(wobject)

camera = gfx.PerspectiveCamera(70, 16 / 9)
camera.position.set(0, -65, 50)
controller = gfx.OrbitController(camera.position.clone())
controller.add_default_event_handlers(renderer, camera)

camera.add(gfx.DirectionalLight(color=(1, 1, 1, 1)))
camera.add(gfx.AmbientLight())


def animate():
    controller.update_camera(camera)
    renderer.render(scene, camera)


if __name__ == "__main__":
    canvas.request_draw(animate)
    run()
