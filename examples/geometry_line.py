import numpy as np

import pygfx as gfx

from PyQt5 import QtWidgets
from wgpu.gui.qt import WgpuCanvas

app = QtWidgets.QApplication([])

canvas = WgpuCanvas()
renderer = gfx.WgpuRenderer(canvas)
renderer_svg = gfx.SvgRenderer(640, 480, "~/line.svg")

scene = gfx.Scene()

positions = [
    [200 + np.sin(i) * i * 6, 250 + np.cos(i) * i * 6, 0, 1] for i in range(20)
]
positions += [
    [400 - np.sin(i) * i * 6, 250 + np.cos(i) * i * 6, 0, 1] for i in range(20)
]
geometry = gfx.Geometry(positions=positions)

material = gfx.LineStripMaterial(thickness=10.0, color=(0.8, 0.7, 0.0, 1.0))
line = gfx.Line(geometry, material)
scene.add(line)

camera = gfx.ScreenCoordsCamera()


def animate():
    # would prefer to do this in a resize event only
    lsize = canvas.get_logical_size()
    camera.set_viewport_size(*lsize)
    renderer.render(scene, camera)
    canvas.request_draw()


if __name__ == "__main__":
    renderer_svg.render(scene, camera)
    canvas.draw_frame = animate
    app.exec_()
