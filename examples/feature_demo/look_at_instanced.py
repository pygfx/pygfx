"""
Look At (Instanced Mesh)
========================

This example shows how the look_at function can be used with instanced mesh.
"""

# sphinx_gallery_pygfx_docs = 'animate 3s'
# sphinx_gallery_pygfx_test = 'run'

from time import perf_counter
import numpy as np
import pygfx as gfx
import pylinalg as la

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
plane.local.position = (0, -2500, 0)
plane.local.rotation = la.quat_from_euler(-np.pi / 2, order="X")
plane.receive_shadow = True
scene.add(plane)

sun = gfx.DirectionalLight()
sun.local.position = (0, 3000, 0)
sun.cast_shadow = True
sun.shadow.camera.depth_range = (1, 6000)
sun.shadow.camera.width = 4000
sun.shadow.camera.height = 4000
scene.add(gfx.AmbientLight())
scene.add(sun)

geometry = gfx.cylinder_geometry(10, 0, 100, 12)
material = gfx.MeshPhongMaterial(color="#336699")
cones = gfx.Group()

instance_count = 100
mesh = gfx.InstancedMesh(geometry, material, instance_count)
mesh.cast_shadow = mesh.receive_shadow = True

for i in range(instance_count):
    mesh.set_matrix_at(
        i,
        la.mat_compose(
            np.random.rand(3) * 4000 - 2000, (0, 0, 0, 1), np.random.rand() * 4 + 2
        ),
    )

scene.add(mesh)

dummy = gfx.objects.WorldObject()  # dummy object to calculate look_at


def animate():
    t = perf_counter() / 0.5
    sphere.local.position = (
        np.sin(t * 0.7) * 2000,
        np.cos(t * 0.5) * 2000,
        np.cos(t * 0.3) * 2000,
    )

    for i in range(instance_count):
        m = mesh.get_matrix_at(i)
        dummy.world.matrix = m
        dummy.look_at(sphere.world.position)
        m = dummy.world.matrix
        mesh.set_matrix_at(i, m)


if __name__ == "__main__":
    disp = gfx.Display()
    disp.before_render = animate
    disp.show(scene)
