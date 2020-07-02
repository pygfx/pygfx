import numpy as np
import pygfx as gfx

from PyQt5 import QtWidgets
from wgpu.gui.qt import WgpuCanvas

app = QtWidgets.QApplication([])

canvas = WgpuCanvas()
renderer = gfx.WgpuRenderer(canvas)

scene = gfx.Scene()

positions = np.random.normal(0, 0.5, (100, 2)).astype(np.float32)
geometry = gfx.Geometry(positions=positions)

material = gfx.PointsMaterial(size=10, color=(0, 1, 0.5, 0.7))
points = gfx.Points(geometry, material)
scene.add(points)

camera = gfx.NDCCamera()


def animate():
    # would prefer to do this in a resize event only
    psize = canvas.get_physical_size()
    camera.set_viewport_size(*psize)
    renderer.render(scene, camera)
    canvas.request_draw()


if __name__ == "__main__":
    canvas.draw_frame = animate
    app.exec_()
