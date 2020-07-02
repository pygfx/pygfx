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
line1 = [[100, 100], [100, 200], [100, 200], [100, 400]]

# A line with a 180 degree turn (a bit of a special case for the implementation)
line2 = [[200, 100], [200, 400], [200, 100]]

# A swiggly line
line3 = [[300 + random.randint(-10, 10), 100 + i * 3] for i in range(100)]

# A line with other turns
line4 = [[400, 100], [500, 200], [400, 300], [450, 400]]

scene = gfx.Scene()

material = gfx.LineMaterial(thickness=80.0, color=(0.8, 0.7, 0.0, 1.0))

for line in [line1, line2, line3, line4]:
    line = [pos + [0, 1] for pos in line]  # Make the positions vec4
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
