"""
Example showing picking the color from the scene. Depending on the
object being clicked, more detailed picking info is available.
"""

import numpy as np
import imageio
import pygfx as gfx

from PySide6 import QtWidgets
from wgpu.gui.qt import WgpuCanvas


class PickingWgpuCanvas(WgpuCanvas):
    def mousePressEvent(self, event):  # noqa: N802
        # Get a dict with info about the clicked location
        xy = event.position().x(), event.position().y()
        info = renderer.get_pick_info(xy)
        for key, val in info.items():
            print(key, "=", val)


app = QtWidgets.QApplication([])
canvas = PickingWgpuCanvas()
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


def animate():
    rot = gfx.linalg.Quaternion().set_from_euler(gfx.linalg.Euler(0.005, 0.01))
    cube.rotation.multiply(rot)

    renderer.render(scene, camera)
    canvas.request_draw()


if __name__ == "__main__":
    canvas.request_draw(animate)
    app.exec()
