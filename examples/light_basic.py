"""
Simple light example.
This example shows a cube with MeshPhongMaterial illuminated by a point light and a directional light.
"""

import time
import math

from wgpu.gui.auto import WgpuCanvas, run
import pygfx as gfx

renderer = gfx.renderers.WgpuRenderer(WgpuCanvas())
scene = gfx.Scene()

cube = gfx.Mesh(
    gfx.box_geometry(20, 20, 20),
    gfx.MeshPhongMaterial(),
)
cube.rotation.set_from_euler(gfx.linalg.Euler(math.pi / 6, math.pi / 6))
scene.add(cube)

light = gfx.DirectionalLight("#0040ff")
light.position.x = 15
light.position.y = 20

light_sp = gfx.sphere_geometry(1)

h1 = gfx.DirectionalLightHelper(light, 10)

scene.add(light)
scene.add(h1)

light2 = gfx.PointLight("#ffaa00")
light2.position.x = -15
light2.position.y = 20

h2 = gfx.PointLightHelper(light2)
scene.add(light2)
scene.add(h2)

scene.add(gfx.AmbientLight())

camera = gfx.PerspectiveCamera(70, 16 / 9)
camera.position.z = 60

controller = gfx.OrbitController(camera.position.clone())
controller.add_default_event_handlers(renderer, camera)


def animate():
    controller.update_camera(camera)
    t = time.time() * 0.1
    scale = 30

    light2.position.x = math.cos(t) * math.cos(3 * t) * scale
    light2.position.y = math.cos(3 * t) * math.sin(t) * scale
    light2.position.z = math.sin(3 * t) * scale

    renderer.render(scene, camera)
    renderer.request_draw()


if __name__ == "__main__":
    renderer.request_draw(animate)
    run()
