"""
Example showing picking the color from the scene. Depending on the
object being clicked, more detailed picking info is available.
"""

import numpy as np
import imageio
from wgpu.gui.auto import WgpuCanvas, run
import pygfx as gfx


canvas = WgpuCanvas()
renderer = gfx.renderers.WgpuRenderer(canvas)
scene = gfx.Scene()

background = gfx.Background(None, gfx.BackgroundMaterial((0, 1, 0, 1), (0, 0, 1, 1)))
scene.add(background)

im = imageio.imread("imageio:astronaut.png").astype(np.float32) / 255
tex = gfx.Texture(im, dim=2).get_view(filter="linear", address_mode="repeat")

geometry = gfx.box_geometry(200, 200, 200)
material = gfx.MeshBasicMaterial(map=tex)
cube = gfx.Mesh(geometry, material)
scene.add(cube)

# camera = gfx.OrthographicCamera(300, 300)
camera = gfx.PerspectiveCamera(70, 16 / 9)
camera.position.z = 400


@canvas.add_event_handler("pointer_down")
def handle_event(event):
    info = renderer.get_pick_info((event["x"], event["y"]))
    for key, val in info.items():
        print(key, "=", val)


def animate():
    rot = gfx.linalg.Quaternion().set_from_euler(gfx.linalg.Euler(0.005, 0.01))
    cube.rotation.multiply(rot)

    renderer.render(scene, camera)
    canvas.request_draw()


if __name__ == "__main__":
    canvas.request_draw(animate)
    run()
