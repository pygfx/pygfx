# import asyncio

import visvis2 as vv

from PyQt5 import QtWidgets
from wgpu.gui.qt import WgpuCanvas

app = QtWidgets.QApplication([])

canvas = WgpuCanvas()
renderer = vv.WgpuSurfaceRenderer(canvas)

scene = vv.Scene()

t1 = vv.Triangle()
t2 = vv.Triangle()

scene.add(t1)
scene.add(t2)
scene.add(vv.Triangle())
scene.add(vv.Triangle())

camera = vv.Camera()
camera.projection_matrix.identity()


def animate():
    # Actually render the scene
    renderer.render(scene, camera)


if __name__ == "__main__":
    canvas.drawFrame = animate
    app.exec_()
