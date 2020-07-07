"""
Example showing a Klein Bottle.
"""

import numpy as np
import imageio
import pygfx as gfx

from PyQt5 import QtWidgets
from wgpu.gui.qt import WgpuCanvas


app = QtWidgets.QApplication([])

canvas = WgpuCanvas()
renderer = gfx.renderers.WgpuRenderer(canvas)
scene = gfx.Scene()

im = imageio.imread("imageio:bricks.jpg").astype(np.float32) / 255
im = np.concatenate([im, np.ones(im.shape[:2] + (1,), dtype=im.dtype)], 2)
tex = gfx.Texture(im, dim=2, usage="sampled").get_view(
    filter="linear", address_mode="repeat"
)

geometry = gfx.KleinBottleGeometry(200, 200, 200)
material = gfx.MeshPhongMaterial(color=(1, 1, 0, 1), clim=(0.2, 0.8))
cube = gfx.Mesh(geometry, material)
scene.add(cube)

cube2 = gfx.Mesh(geometry, gfx.MeshNormalLinesMaterial(color=(0, 0, 1, 1)))
cube.add(cube2)

camera = gfx.PerspectiveCamera(70, 1)
camera.position.z = 400


def animate():
    rot = gfx.linalg.Quaternion().set_from_euler(gfx.linalg.Euler(0.0007, 0.001))
    cube.rotation.multiply(rot)

    renderer.render(scene, camera)
    canvas.request_draw()


if __name__ == "__main__":
    canvas.request_draw(animate)
    app.exec_()
