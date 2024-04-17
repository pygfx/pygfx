"""
Color Picking
=============

Example showing picking the color from the scene. Depending on the
object being clicked, more detailed picking info is available.
"""

# sphinx_gallery_pygfx_docs = 'screenshot'
# sphinx_gallery_pygfx_test = 'run'

import numpy as np
import imageio.v3 as iio
from wgpu.gui.auto import WgpuCanvas, run
import pygfx as gfx
import pylinalg as la


canvas = WgpuCanvas()
renderer = gfx.renderers.WgpuRenderer(canvas)
scene = gfx.Scene()

dark_gray = np.array((169, 167, 168, 255)) / 255
light_gray = np.array((100, 100, 100, 255)) / 255
background = gfx.Background.from_color(light_gray, dark_gray)
scene.add(background)

im = iio.imread("imageio:astronaut.png").astype(np.float32) / 255
tex = gfx.Texture(im, dim=2)

geometry = gfx.box_geometry(200, 200, 200)
material = gfx.MeshBasicMaterial(map=tex, pick_write=True)
cube = gfx.Mesh(geometry, material)
scene.add(cube)

# camera = gfx.OrthographicCamera(300, 300)
camera = gfx.PerspectiveCamera(70, 16 / 9)
camera.show_object(cube, scale=1.5)


@renderer.add_event_handler("pointer_down")
def handle_event(event):
    info = event.pick_info
    for key, val in info.items():
        print(key, "=", val)


def animate():
    rot = la.quat_from_euler((0.005, 0.01), order="XY")
    cube.local.rotation = la.quat_mul(rot, cube.local.rotation)

    renderer.render(scene, camera)
    canvas.request_draw()


if __name__ == "__main__":
    canvas.request_draw(animate)
    run()
