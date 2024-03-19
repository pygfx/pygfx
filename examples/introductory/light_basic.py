"""
Lights
======


Simple light example.
This example shows a cube with MeshPhongMaterial illuminated by a point light and a directional light.
"""

# sphinx_gallery_pygfx_docs = 'screenshot'
# sphinx_gallery_pygfx_test = 'run'

import time
import numpy as np

from wgpu.gui.auto import WgpuCanvas, run
import pygfx as gfx
import pylinalg as la

renderer = gfx.renderers.WgpuRenderer(WgpuCanvas())
scene = gfx.Scene()

cube = gfx.Mesh(
    gfx.box_geometry(20, 20, 20),
    gfx.MeshPhongMaterial(),
)
cube.local.rotation = la.quat_from_euler((np.pi / 6, np.pi / 6), order="XY")
scene.add(cube)

light = gfx.DirectionalLight("#0040ff", 3)
light.local.x = 15
light.local.y = 20

scene.add(light.add(gfx.DirectionalLightHelper(10)))

light2 = gfx.PointLight("#ffaa00", 300, decay=2)  # 300 candela
light2.local.x = -15
light2.local.y = 20

scene.add(light2.add(gfx.PointLightHelper()))

scene.add(gfx.AmbientLight("#fff", 0.2))

camera = gfx.PerspectiveCamera(70, 16 / 9)
camera.show_object(cube, scale=1.5)
controller = gfx.OrbitController(camera, register_events=renderer)


def animate():
    t = time.time() * 0.1
    scale = 30

    light2.local.position = (
        np.cos(t) * np.cos(3 * t) * scale,
        np.cos(3 * t) * np.sin(t) * scale,
        np.sin(3 * t) * scale,
    )

    renderer.render(scene, camera)
    renderer.request_draw()


if __name__ == "__main__":
    renderer.request_draw(animate)
    run()
