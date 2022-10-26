"""
This example combines all types of lights and all types of materials with shadows.
"""
# test_example = true

import math

from wgpu.gui.auto import WgpuCanvas, run
import pygfx as gfx

canvas = WgpuCanvas()
renderer = gfx.renderers.WgpuRenderer(canvas)
scene = gfx.Scene()

cube = gfx.Mesh(
    gfx.box_geometry(20, 20, 20),
    gfx.MeshPhongMaterial(),
)
cube.rotation.set_from_euler(gfx.linalg.Euler(math.pi / 6, math.pi / 6))
cube.cast_shadow = True
scene.add(cube)

cube2 = gfx.Mesh(
    gfx.box_geometry(50, 50, 50),
    gfx.MeshPhongMaterial(),
)
cube2.rotation.set_from_euler(gfx.linalg.Euler(math.pi / 4, math.pi / 4))
cube2.position.set(0, -150, 0)
cube2.cast_shadow = True
cube2.receive_shadow = True
scene.add(cube2)

cube3 = gfx.Mesh(
    gfx.box_geometry(100, 100, 100),
    gfx.MeshPhongMaterial(),
)

cube3.position.set(0, -250, 0)
cube3.cast_shadow = True
cube3.receive_shadow = True
scene.add(cube3)

box = gfx.Mesh(
    gfx.box_geometry(600, 600, 600),
    gfx.MeshStandardMaterial(color="#808080", side="Back"),
)

box.rotation.set_from_euler(gfx.linalg.Euler(-math.pi / 2))
box.position.set(0, 0, 0)
box.receive_shadow = True
box.cast_shadow = False
scene.add(box)

ambient = gfx.AmbientLight()

scene.add(ambient)

light = gfx.PointLight("#4040ff", 500000, decay=2)
light.position.x = 15
light.position.y = 20

light.cast_shadow = True
scene.add(light)

light2 = gfx.DirectionalLight("#aaaaaa")
light2.position.x = -150
light2.position.y = 100
light2.position.z = 100
light2.cast_shadow = True

scene.add(light2)

camera = gfx.PerspectiveCamera(70, 16 / 9)
camera.position.x = 0
camera.position.y = 0
camera.position.z = 350


canvas.request_draw(lambda: renderer.render(scene, camera))


if __name__ == "__main__":
    run()
