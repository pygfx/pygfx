"""
Post processing effect: normal reconstruction
=============================================

This example demonstrates how to accurately reconstruct normals from the depth buffer.

Implementation is based on https://atyuwen.github.io/posts/normal-reconstruction/
"""

# sphinx_gallery_pygfx_docs = 'animate 4s'
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
# camera = gfx.PerspectiveCamera(45, 800 / 600, depth_range=(0.1, 1000))
camera = gfx.PerspectiveCamera(45, 800 / 600, depth_range=(3, 10))
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
    normal_pass.near = camera.near
    normal_pass.far = camera.far
    normal_pass.width = canvas.get_physical_size()[0]
    normal_pass.height = canvas.get_physical_size()[1]
    with stats:
        renderer.render(scene, camera, flush=False)
    stats.render()

    # vs = la.vec_normalize(la.vec_transform([0, 0, 1.0], camera.projection_matrix_inverse))
    # print("----")
    # print(vs)
    # print(-(camera.near + camera.far) / 2)

    # depth = 0.0  # example depth buffer value
    # z_eye = (camera.near * camera.far) / (camera.far - depth * (camera.far - camera.near))
    # print(z_eye)

    # print(vs * z_eye)



if __name__ == "__main__":
    renderer.request_draw(animate)
    loop.run()
