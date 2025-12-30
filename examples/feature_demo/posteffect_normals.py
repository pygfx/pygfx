"""
Screen space normal reconstruction
==================================

This example demonstrates how to accurately reconstruct surface normals using the depth buffer.

Implementation is based on https://atyuwen.github.io/posts/normal-reconstruction/
"""

# sphinx_gallery_pygfx_docs = 'screenshot'
# sphinx_gallery_pygfx_test = 'run'

import pygfx as gfx
import numpy as np
import pylinalg as la
from rendercanvas.auto import RenderCanvas, loop
from pygfx.renderers.wgpu import NormalPass

canvas = RenderCanvas(
    size=(800, 600), update_mode="fastest", title="Animations", vsync=False
)

renderer = gfx.WgpuRenderer(canvas)
camera = gfx.PerspectiveCamera(45, 800 / 600, depth_range=(0.1, 1000))
camera.local.position = (3, 4, 1)
scene_center = (-1, 0.5, -2)
camera.look_at(scene_center)
scene = gfx.Scene()

dl = gfx.DirectionalLight()
dl.local.position = (6, 8, 2)
scene.add(gfx.AmbientLight(), dl)

# scene objects
plane = gfx.Mesh(
    gfx.plane_geometry(100, 100),
    gfx.MeshPhongMaterial(color="lightgray"),
)
plane.local.rotation = la.quat_from_axis_angle((1, 0, 0), np.pi / 2)
scene.add(plane)

boxes = gfx.Group()

box = gfx.Mesh(
    gfx.box_geometry(4, 2, 4),
    gfx.MeshPhongMaterial(color="#444"),
)
box.local.position = (-2, 1, -3)
boxes.add(box)

box2 = gfx.Mesh(
    gfx.box_geometry(1, 1, 3),
    gfx.MeshPhongMaterial(color="#666"),
)
box2.local.position = (0.5, 0.5, -1)
boxes.add(box2)

scene.add(boxes)

controller = gfx.OrbitController(camera, target=scene_center, register_events=renderer)

stats = gfx.Stats(viewport=renderer)

normal_pass = NormalPass()
renderer.effect_passes = [normal_pass]


def animate():
    normal_pass.cam_transform_inv = camera.world.matrix.T
    normal_pass.projection_transform_inv = camera.projection_matrix_inverse.T
    normal_pass.width, normal_pass.height = canvas.get_physical_size()
    with stats:
        renderer.render(scene, camera, flush=False)
    stats.render()


if __name__ == "__main__":
    renderer.request_draw(animate)
    loop.run()
