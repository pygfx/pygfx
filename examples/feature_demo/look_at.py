"""
Look At
=======

In this example shows how the look_at function can be used.
"""
# sphinx_gallery_pygfx_animate = True
# sphinx_gallery_pygfx_target_name = "disp"
from time import perf_counter_ns
import numpy as np
import pygfx as gfx

scene = gfx.Scene()

sphere = gfx.Mesh(
    gfx.sphere_geometry(100, 20, 20),
    gfx.MeshPhongMaterial(color="#336699"),
)
sphere.receive_shadow = sphere.cast_shadow = True
scene.add(sphere)

plane = gfx.Mesh(
    gfx.plane_geometry(4000, 4000),
    gfx.MeshPhongMaterial(color="#336699"),
)
plane.position.set(0, -2500, 0)
plane.rotation.set_from_euler(gfx.linalg.Euler(-np.pi / 2))
plane.receive_shadow = True
scene.add(plane)

sun = gfx.DirectionalLight(position=(0, 3000, 0))
sun.cast_shadow = True
sun.shadow.camera.depth_range = (1, 6000)
sun.shadow.camera.width = 4000
sun.shadow.camera.height = 4000
scene.add(gfx.AmbientLight())
scene.add(sun)

geometry = gfx.cylinder_geometry(10, 0, 100, 12)
material = gfx.MeshPhongMaterial(color="#336699")
cones = gfx.Group()

for i in range(100):
    cone = gfx.Mesh(geometry, material)
    cone.position.set(*(np.random.rand(3) * 4000 - 2000))
    cone.scale.set_scalar(np.random.rand() * 4 + 2)
    cone.receive_shadow = cone.cast_shadow = True
    cones.add(cone)

scene.add(cones)


def animate():
    t = perf_counter_ns() // 1_000_000 * 0.0005
    sphere.position.set(
        np.sin(t * 0.7) * 2000,
        np.cos(t * 0.5) * 2000,
        np.cos(t * 0.3) * 2000,
    )

    for c in cones.children:
        c.look_at(sphere.position)


if __name__ == "__main__":
    disp = gfx.Display()
    disp.before_render = animate
    disp.stats = True
    disp.show(scene)
