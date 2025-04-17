"""
Panzoom Camera
==============

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


canvas = WgpuCanvas()
renderer = gfx.renderers.WgpuRenderer(canvas)
scene = gfx.Scene()

axes = gfx.AxesHelper(size=250)
scene.add(axes)

dark_gray = np.array((169, 167, 168, 255)) / 255
light_gray = np.array((100, 100, 100, 255)) / 255
background = gfx.Background.from_color(light_gray, dark_gray)
scene.add(background)

im = iio.imread("imageio:astronaut.png")
tex = gfx.Texture(im, dim=2)
geometry = gfx.plane_geometry(512, 512)
material = gfx.MeshBasicMaterial(map=tex)
plane = gfx.Mesh(geometry, material)
scene.add(plane)

camera = gfx.OrthographicCamera(512, 512)
camera.show_object(scene)
controller = gfx.PanZoomController(camera, register_events=renderer)


# initial camera state
camera_state = camera.get_state()


def on_key_down(event):
    if event.key not in ["s", "l", "r"]:
        return

    global camera_state

    # without disabling the controller any in-progress action
    # during this function call will overwrite the camera state
    controller.enabled = False

    if event.key == "s":
        camera_state = camera.get_state()
        print("save")
    elif event.key == "l":
        camera.set_state(camera_state)
        print("load")
    elif event.key == "r":
        camera.show_object(scene)
        print("reset")

    controller.enabled = True

    canvas.request_draw()  # not required if you are continuously rendering


renderer.add_event_handler(on_key_down, "key_down")


if __name__ == "__main__":
    canvas.request_draw(lambda: renderer.render(scene, camera))
    run()
