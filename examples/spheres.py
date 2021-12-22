"""
Example showing different types of geometric cylinders.
"""

import numpy as np
from wgpu.gui.auto import WgpuCanvas, run
import pygfx as gfx


canvas = WgpuCanvas()
renderer = gfx.renderers.WgpuRenderer(canvas)
scene = gfx.Scene()

spheres = [
    (
        (0, 0, -7.5),
        (0, 0.65, 0, 1),
        gfx.sphere_geometry(4.5, phi_length=np.pi * 1.5),
    ),
    ((0, 0, 7.5), (1, 1, 1, 1), gfx.sphere_geometry(4)),
    (
        (15, 0, -7.5),
        (1, 0.3, 0.3, 1),
        gfx.sphere_geometry(4, theta_start=np.pi * 0.25, theta_length=np.pi * 0.50),
    ),
    (
        (15, 0, 7.5),
        (0.35, 0, 0, 1),
        gfx.sphere_geometry(5, width_segments=6),
    ),
    ((-15, 0, -7.5), (1, 1, 0.75, 1), gfx.sphere_geometry(7)),
    ((-15, 0, 7.5), (1, 1, 0.75, 1), gfx.sphere_geometry(5, height_segments=8)),
]
for pos, color, geometry in spheres:
    material = gfx.MeshFlatMaterial(color=color)
    wobject = gfx.Mesh(geometry, material)
    wobject.position.set(*pos)
    scene.add(wobject)

camera = gfx.PerspectiveCamera(70, 16 / 9)
camera.position.set(6, 16, -22)
controls = gfx.OrbitControls(camera.position.clone())
controls.add_default_event_handlers(canvas, camera)


def animate():
    controls.update_camera(camera)
    renderer.render(scene, camera)
    canvas.request_draw()


if __name__ == "__main__":
    canvas.request_draw(animate)
    run()
