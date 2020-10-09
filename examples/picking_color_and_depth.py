"""
Example showing picking the color and depth from the scene.
"""

import numpy as np
import imageio
import pygfx as gfx

from PyQt5 import QtWidgets
from wgpu.gui.qt import WgpuCanvas


app = QtWidgets.QApplication([])


class PickingWgpuCanvas(WgpuCanvas):
    def mousePressEvent(self, event):  # noqa: N802
        xy = event.x(), self.height() - event.y()  # renderer as origin at bottom left
        print(renderer.get_info_at(xy))


canvas = PickingWgpuCanvas()
renderer = gfx.renderers.WgpuRenderer(canvas)
scene = gfx.Scene()

im = imageio.imread("imageio:astronaut.png").astype(np.float32) / 255
tex = gfx.Texture(im, dim=2, usage="sampled").get_view(
    filter="linear", address_mode="repeat"
)

geometry = gfx.BoxGeometry(200, 200, 200)
material = gfx.MeshBasicMaterial(map=tex)
cube = gfx.Mesh(geometry, material)
scene.add(cube)


camera = gfx.PerspectiveCamera(70, 16 / 9)
camera.position.z = 400


def animate():
    rot = gfx.linalg.Quaternion().set_from_euler(gfx.linalg.Euler(0.005, 0.01))
    cube.rotation.multiply(rot)

    renderer.render(scene, camera)
    canvas.request_draw()


if __name__ == "__main__":
    canvas.request_draw(animate)
    app.exec_()
