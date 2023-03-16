"""

Directional Shadow 1
====================

This example demonstrates directional light shadows and their helper. The cubes
within the view frustum of the shadow camera have complete shadows, the cubes at
the edge of the view frustum of the camera have partial shadows, while the cubes
outside the view frustum of the camera will not cast shadows.
"""
# sphinx_gallery_pygfx_render = True

import math

from wgpu.gui.auto import WgpuCanvas, run
import pygfx as gfx

renderer = gfx.renderers.WgpuRenderer(WgpuCanvas())
scene = gfx.Scene()

floor = gfx.Mesh(
    gfx.plane_geometry(2000, 2000),
    gfx.MeshPhongMaterial(color="#808080", side="Front"),
)

floor.rotation.set_from_euler(gfx.linalg.Euler(-math.pi / 2))
# floor.position.set(0, -0.05, 0)
floor.receive_shadow = True

scene.add(floor)

ambient = gfx.AmbientLight("#fff", 0.1)
scene.add(ambient)

light = gfx.DirectionalLight("#aaaaaa")
light.position.x = -50
light.position.y = 50
light.cast_shadow = True

light.shadow.camera.width = 100
light.shadow.camera.height = 100
light.shadow.camera.update_projection_matrix()

scene.add(light.add(gfx.DirectionalLightHelper(30, show_shadow_extent=True)))

box = gfx.box_geometry(20, 20, 20)
material = gfx.MeshPhongMaterial()

for i in range(5):
    cube = gfx.Mesh(box, material)
    cube.position.set(0, 10, i * 50 - 100)
    cube.cast_shadow = True
    scene.add(cube)


camera = gfx.PerspectiveCamera(70, 16 / 9)
camera.position.y = 100
camera.position.z = 350
camera.show_pos((0, 0, 0))

controller = gfx.OrbitController(camera, register_events=renderer)


def animate():
    renderer.render(scene, camera)
    renderer.request_draw()


if __name__ == "__main__":
    renderer.request_draw(animate)
    run()
