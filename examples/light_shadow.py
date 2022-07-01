import time
import math

from wgpu.gui.auto import WgpuCanvas, run
import pygfx as gfx

renderer = gfx.renderers.WgpuRenderer(WgpuCanvas())
scene = gfx.Scene()

cube = gfx.Mesh(
    gfx.box_geometry(20, 20, 20),
    gfx.MeshStandardMaterial(),
)
cube.rotation.set_from_euler(gfx.linalg.Euler(math.pi / 6, math.pi / 6))
cube.cast_shadow = True
scene.add(cube)

cube2 = gfx.Mesh(
    gfx.box_geometry(50, 50, 50),
    gfx.MeshStandardMaterial(),
)
cube2.rotation.set_from_euler(gfx.linalg.Euler(math.pi / 4, math.pi / 4))
cube2.position.set(0, -150, 0)
cube2.cast_shadow = True
cube2.receive_shadow = True
scene.add(cube2)

cube3 = gfx.Mesh(
    gfx.box_geometry(100, 100, 100),
    gfx.MeshStandardMaterial(),
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

ambient = gfx.AmbientLight("#111111")

scene.add(ambient)

light = gfx.PointLight("#4040ff")
light.position.x = 15
light.position.y = 20

light.cast_shadow = True

h1 = gfx.PointLightHelper(light, 5)

scene.add(light)
scene.add(h1)

light2 = gfx.DirectionalLight("#aaaaaa")
light2.position.x = -150
light2.position.y = 100
light2.position.z = 100
light2.cast_shadow = True

h2 = gfx.DirectionalLightHelper(light2, 100)

scene.add(light2)
scene.add(h2)

camera = gfx.PerspectiveCamera(70, 16 / 9)
camera.position.x = 100
camera.position.y = 100
camera.position.z = 350

controller = gfx.OrbitController(camera.position.clone())
controller.add_default_event_handlers(renderer, camera)


def animate():
    controller.update_camera(camera)
    t = time.time() * 0.5
    scale = 150

    light.position.x = math.cos(t) * math.cos(3 * t) * scale
    light.position.y = math.cos(3 * t) * math.sin(t) * scale / 2
    light.position.z = math.sin(3 * t) * scale

    renderer.render(scene, camera)
    renderer.request_draw()


if __name__ == "__main__":
    renderer.request_draw(animate)
    run()
