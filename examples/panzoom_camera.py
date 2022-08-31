"""
Example showing orbit camera controller.

Press 's' to save the state, and
press 'l' to load the last saved state.
"""

import imageio
from wgpu.gui.auto import WgpuCanvas, run
import pygfx as gfx


canvas = WgpuCanvas()
renderer = gfx.renderers.WgpuRenderer(canvas)
scene = gfx.Scene()

axes = gfx.AxesHelper(size=250)
scene.add(axes)

background = gfx.Background(None, gfx.BackgroundMaterial((0, 1, 0, 1), (0, 1, 1, 1)))
scene.add(background)

im = imageio.imread("imageio:astronaut.png")
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


def animate():
    controller.update_camera(camera)
    renderer.render(scene, camera)
    canvas.request_draw()


if __name__ == "__main__":
    canvas.request_draw(animate)
    run()
