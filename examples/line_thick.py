"""
Display very thick lines to show how lines stay pretty on large scales.
"""

import random

import pygfx as gfx

from PyQt5 import QtWidgets
from wgpu.gui.qt import WgpuCanvas

app = QtWidgets.QApplication([])

canvas = WgpuCanvas()
renderer = gfx.WgpuRenderer(canvas)

# A line consisting of two points
line1 = [[100, 100, 0, 1], [400, 100, 0, 1]]

# A line with a 180 degree turn
# todo: fix this adversarial use-case
line2 = [[100, 200, 0, 1], [400, 200, 0, 1], [100, 200, 0, 1]]
# line2 = [[400, 200, 0, 1], [200, 200, 0, 1], [400, 200, 0, 1]]

# A swiggly line
line3 = [[100 + i * 3, 300 + random.randint(-10, 10), 0, 1] for i in range(100)]

# A line with other turns
line4 = [[100, 400, 0, 1], [200, 500, 0, 1], [300, 400, 0, 1], [400, 450, 0, 1]]

scene = gfx.Scene()

material = gfx.LineStripMaterial(thickness=80.0, color=(0.8, 0.7, 0.0, 1.0))

for line in [line1, line2, line3, line4]:
    geometry = gfx.Geometry(positions=line)
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
    canvas.draw_frame = animate
    app.exec_()
