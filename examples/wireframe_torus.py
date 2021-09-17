"""
Example showing a Torus knot, as a wireframe
"""

import imageio
import pygfx as gfx

from PyQt5 import QtWidgets
from wgpu.gui.qt import WgpuCanvas


app = QtWidgets.QApplication([])

canvas = WgpuCanvas()
renderer = gfx.renderers.WgpuRenderer(canvas)
scene = gfx.Scene()

im = imageio.imread("imageio:bricks.jpg")
tex = gfx.Texture(im, dim=2).get_view(filter="linear", address_mode="repeat")

geometry = gfx.TorusKnotGeometry(1, 0.3, 64, 16)
geometry.texcoords.data[:, 0] *= 10  # stretch the texture

material1 = gfx.MeshPhongMaterial(map=tex, clim=(10, 240))
obj1 = gfx.Mesh(geometry, material1)
scene.add(obj1)


material2 = gfx.MeshBasicMaterial(color=(0, 0.5, 0, 1), wireframe=1.5)
obj2 = gfx.Mesh(geometry, material2)
scene.add(obj2)

camera = gfx.PerspectiveCamera(70, 1)
camera.position.z = 4


def animate():
    rot = gfx.linalg.Quaternion().set_from_euler(gfx.linalg.Euler(0.0071, 0.01))
    obj1.rotation.multiply(rot)
    obj2.rotation.multiply(rot)

    renderer.render(scene, camera)
    canvas.request_draw()


if __name__ == "__main__":
    canvas.request_draw(animate)
    app.exec_()
