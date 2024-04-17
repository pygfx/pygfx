"""
Orbit Camera
============

Example showing orbit camera controller.

Press 's' to save the state, and
press 'l' to load the last saved state.
"""

# sphinx_gallery_pygfx_docs = 'screenshot'
# sphinx_gallery_pygfx_test = 'run'

import imageio.v3 as iio
from wgpu.gui.auto import WgpuCanvas, run
import pygfx as gfx
import numpy as np
import pylinalg as la


canvas = WgpuCanvas()
renderer = gfx.renderers.WgpuRenderer(canvas)
scene = gfx.Scene()

scene.add(gfx.AxesHelper(size=250))

im = iio.imread("imageio:chelsea.png")
tex = gfx.Texture(im, dim=2)

material = gfx.MeshBasicMaterial(map=tex, side="front")
geometry = gfx.box_geometry(100, 100, 100)
cubes = [gfx.Mesh(geometry, material) for i in range(8)]
for i, cube in enumerate(cubes):
    cube.local.position = (350 - i * 100, 150, 0)
    scene.add(cube)

dark_gray = np.array((169, 167, 168, 255)) / 255
light_gray = np.array((100, 100, 100, 255)) / 255
background = gfx.Background.from_color(light_gray, dark_gray)
scene.add(background)

camera = gfx.PerspectiveCamera(70, 16 / 9)
camera.show_object(scene)
controller = gfx.OrbitController(camera, register_events=renderer)

state = {}


def on_key_down(event):
    global state
    if event.key == "s":
        state = camera.get_state()
    elif event.key == "l":
        camera.set_state(state)
    elif event.key == "r":
        camera.show_object(scene)


renderer.add_event_handler(on_key_down, "key_down")


def animate():
    for i, cube in enumerate(cubes):
        rot = la.quat_from_euler((0.005 * i, 0.01 * i), order="XY")
        cube.local.rotation = la.quat_mul(rot, cube.local.rotation)

    renderer.render(scene, camera)
    canvas.request_draw()


if __name__ == "__main__":
    canvas.request_draw(animate)
    run()
