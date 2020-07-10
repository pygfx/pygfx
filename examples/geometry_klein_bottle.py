"""
Example showing a Klein Bottle.
"""

import pygfx as gfx

from PyQt5 import QtWidgets
from wgpu.gui.qt import WgpuCanvas


app = QtWidgets.QApplication([])

canvas = WgpuCanvas()
renderer = gfx.renderers.WgpuRenderer(canvas)
scene = gfx.Scene()

geometry = gfx.KleinBottleGeometry(10)
geometry.texcoords = None
material = gfx.MeshPhongMaterial(color=(1, 0.5, 0, 1))
obj = gfx.Mesh(geometry, material)
scene.add(obj)

obj2 = gfx.Mesh(geometry, gfx.MeshNormalLinesMaterial(color=(0, 0, 1, 1)))
obj.add(obj2)

camera = gfx.PerspectiveCamera(70, 1)
camera.position.z = 30


def animate():
    rot = gfx.linalg.Quaternion().set_from_euler(gfx.linalg.Euler(0.00071, 0.001))
    obj.rotation.multiply(rot)

    renderer.render(scene, camera)
    canvas.request_draw()


if __name__ == "__main__":
    canvas.request_draw(animate)
    app.exec_()
