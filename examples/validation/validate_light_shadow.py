"""
Light and Shadow
================

This example combines MeshPhongMaterial and MeshStandardMaterial with
PointLight, AmbientLight, SpotLight and DirectionalLight to check
that all combinations are working properly.
"""

# sphinx_gallery_pygfx_docs = 'screenshot'
# sphinx_gallery_pygfx_test = 'compare'

from wgpu.gui.auto import WgpuCanvas, run
import pygfx as gfx
import pylinalg as la
import numpy as np

canvas = WgpuCanvas()
renderer = gfx.renderers.WgpuRenderer(canvas)
scene = gfx.Scene()

cube = gfx.Mesh(
    gfx.box_geometry(20, 20, 20),
    gfx.MeshPhongMaterial(),
)
cube.local.rotation = la.quat_from_euler((np.pi / 6, np.pi / 6), order="XY")
cube.cast_shadow = True
scene.add(cube)

cube2 = gfx.Mesh(
    gfx.box_geometry(50, 50, 50),
    gfx.MeshPhongMaterial(),
)
cube2.local.rotation = la.quat_from_euler((np.pi / 4, np.pi / 4), order="XY")
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

box = gfx.Mesh(
    gfx.box_geometry(600, 600, 600),
    gfx.MeshStandardMaterial(color="#808080", side="Back"),
)
box.local.rotation = la.quat_from_euler(-np.pi / 2, order="X")
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
scene.add(light)

light2 = gfx.DirectionalLight("#aaaaaa")
light2.local.position = (-150, 100, 100)
light2.cast_shadow = True
scene.add(light2)

light3 = gfx.SpotLight("#ffffff", 500, angle=0.3, penumbra=0.2, decay=1)
light3.local.position = (0, 0, 100)
light3.cast_shadow = True
scene.add(light3)

camera = gfx.PerspectiveCamera(70, 16 / 9, depth_range=(0.1, 2000))
camera.local.z = 350

canvas.request_draw(lambda: renderer.render(scene, camera))


if __name__ == "__main__":
    run()
