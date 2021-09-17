"""
Example showing a Torus knot, as a wireframe. We create two wireframes,
one for the front, bright blue and lit, and one for the back, unlit and
gray.
"""

import pygfx as gfx

from PyQt5 import QtWidgets
from wgpu.gui.qt import WgpuCanvas


app = QtWidgets.QApplication([])

canvas = WgpuCanvas()
renderer = gfx.renderers.WgpuRenderer(canvas)
scene = gfx.Scene()

geometry = gfx.TorusKnotGeometry(1, 0.3, 64, 8)

material1 = gfx.MeshBasicMaterial(color=(0.2, 0.2, 0.2, 1.0), wireframe=3, side="back")
obj1 = gfx.Mesh(geometry, material1)
scene.add(obj1)

material2 = gfx.MeshPhongMaterial(color=(0, 0.8, 0.8, 1), wireframe=3, side="front")
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
