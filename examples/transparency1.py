"""
Example showing transparency.
"""
import time
import wgpu
import pygfx as gfx

from PyQt5 import QtWidgets
from wgpu.gui.qt import WgpuCanvas


app = QtWidgets.QApplication([])

canvas = WgpuCanvas()
renderer = gfx.renderers.WgpuRenderer(canvas)
scene = gfx.Scene()

geometry = gfx.PlaneGeometry(50, 50)
plane1 = gfx.Mesh(geometry, gfx.MeshBasicMaterial(color=(1, 0, 0, 0.4)))
plane2 = gfx.Mesh(geometry, gfx.MeshBasicMaterial(color=(0, 1, 0, 0.4)))
plane3 = gfx.Mesh(geometry, gfx.MeshBasicMaterial(color=(0, 0, 1, 0.4)))
plane4 = gfx.Mesh(
    gfx.PlaneGeometry(100, 10), gfx.MeshBasicMaterial(color=(0, 1, 0, 0.4))
)

plane1.position.set(-10, -10, 1)
plane2.position.set(0, 0, 2)
plane3.position.set(10, 10, 3)

scene.add(plane2, plane1, plane3, plane4)
# scene.add(plane1, plane2, plane3, plane4)
# scene.add(plane3, plane2, plane1)


camera = gfx.OrthographicCamera(100, 100)


def animate():
    z = time.time() % 4
    plane4.position.x = z
    plane4.position.z = z * 400

    renderer.render(scene, camera)
    canvas.request_draw()


if __name__ == "__main__":
    canvas.request_draw(animate)
    app.exec()
