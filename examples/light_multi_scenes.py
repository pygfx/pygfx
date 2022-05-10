import time
import math

from wgpu.gui.auto import WgpuCanvas, run
import pygfx as gfx

renderer = gfx.renderers.WgpuRenderer(WgpuCanvas(size=(600, 600)))
scene = gfx.Scene()

cube = gfx.Mesh(
    gfx.box_geometry(20, 20, 20),
    gfx.MeshPhongMaterial(),
)
cube.rotation.set_from_euler(gfx.linalg.Euler(math.pi / 6, math.pi / 6))

light = gfx.PointLight("#0040ff")
light.position.x = 15
light.position.y = 20

light_sp = gfx.sphere_geometry(1)

light_helper = gfx.Mesh(
    light_sp,
    gfx.MeshBasicMaterial(color="#0040ff"),
)
light.add(light_helper)
scene.add(light)


scene2 = gfx.Scene()

light2 = gfx.PointLight("#ffaa00")
light2.position.x = -15
light2.position.y = 20

light_helper2 = gfx.Mesh(
    light_sp,
    gfx.MeshBasicMaterial(color="#ffaa00"),
)
light2.add(light_helper2)
scene2.add(light2)

camera = gfx.PerspectiveCamera(70, 16 / 9)
camera.position.z = 60

controller = gfx.OrbitController(camera.position.clone())
controller.add_default_event_handlers(renderer, camera)


def animate():
    controller.update_camera(camera)
    t = time.time() * 0.2
    scale = 30

    light.position.x = math.cos(t) * math.cos(3 * t) * scale
    light.position.y = math.cos(3 * t) * math.sin(t) * scale
    light.position.z = math.sin(3 * t) * scale

    scene.add(cube)
    renderer.render(scene, camera, rect=(0, 0, 600, 300), flush=False)

    light2.position.x = math.cos(t) * math.cos(3 * t) * scale
    light2.position.y = math.cos(3 * t) * math.sin(t) * scale
    light2.position.z = math.sin(3 * t) * scale

    scene2.add(cube)
    renderer.render(scene2, camera, rect=(0, 300, 600, 300))

    renderer.request_draw()


if __name__ == "__main__":
    renderer.request_draw(animate)
    run()
