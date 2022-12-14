"""
Panzoom Camera
==============

Example showing orbit camera controller.

Press 's' to save the state, and
press 'l' to load the last saved state.
"""
# sphinx_gallery_pygfx_render = True

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
background = gfx.Background(None, gfx.BackgroundMaterial(light_gray, dark_gray))
scene.add(background)

im = iio.imread("imageio:astronaut.png")
tex = gfx.Texture(im, dim=2)
geometry = gfx.plane_geometry(512, 512)
material = gfx.MeshBasicMaterial(map=tex.get_view(filter="linear"))
plane = gfx.Mesh(geometry, material)
scene.add(plane)

camera = gfx.OrthographicCamera(512, 512)
camera.position.set(0, 0, 500)
controller = gfx.PanZoomController(camera.position.clone())
controller.add_default_event_handlers(renderer, camera)


def on_key_down(event):
    if event.key == "s":
        controller.save_state()
    elif event.key == "l":
        controller.load_state()
    elif event.key == "r":
        controller.show_object(camera, plane)


renderer.add_event_handler(on_key_down, "key_down")


def render_scene():
    controller.update_camera(camera)
    renderer.render(scene, camera)
    # NOTE: The controller requests new draws automatically
    # so there is no need for an animation loop


if __name__ == "__main__":
    canvas.request_draw(render_scene)
    run()
