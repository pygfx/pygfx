"""
Example showing off the mesh slice material.
"""

import pygfx as gfx

from PySide6 import QtWidgets
from wgpu.gui.qt import WgpuCanvas


app = QtWidgets.QApplication([])

canvas = WgpuCanvas()
renderer = gfx.renderers.WgpuRenderer(canvas)
scene = gfx.Scene()

geometry = gfx.TorusKnotGeometry(1, 0.3, 128, 16)
material1 = gfx.MeshPhongMaterial(color=(0.5, 0.5, 0.5, 1.0))
material2 = gfx.MeshSliceMaterial(thickness=8, color=(1, 1, 0, 1), plane=(0, 0, 1, 0))
obj1 = gfx.Mesh(geometry, material1)
obj2 = gfx.Mesh(geometry, material2)
scene.add(obj1)
scene.add(obj2)

camera = gfx.PerspectiveCamera(70, 2)
camera.position.z = 4


def animate():

    dist = material2.plane[3]
    dist += 0.02
    if dist > 1:
        dist = -1.5
    material2.plane = 1, 0, 1, dist

    renderer.render(scene, camera)
    canvas.request_draw()


if __name__ == "__main__":
    canvas.request_draw(animate)
    app.exec()
