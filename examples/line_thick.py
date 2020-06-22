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

# A straight line
line1 = [[100, 100, 0, 1], [100, 200, 0, 1], [100, 200, 0, 1], [100, 400, 0, 1]]

# A line with a 180 degree turn (a bit of a special case for the implementation)
line2 = [[200, 100, 0, 1], [200, 400, 0, 1], [200, 100, 0, 1]]

# A swiggly line
line3 = [[300 + random.randint(-10, 10), 100 + i * 3, 0, 1] for i in range(100)]

# A line with other turns
line4 = [[400, 100, 0, 1], [500, 200, 0, 1], [400, 300, 0, 1], [450, 400, 0, 1]]

scene = gfx.Scene()

material = gfx.LineMaterial(thickness=80.0, color=(0.8, 0.7, 0.0, 1.0))

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
