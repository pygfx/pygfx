"""
Some basic line drawing.
"""

import numpy as np

import pygfx as gfx

from PyQt5 import QtWidgets
from wgpu.gui.qt import WgpuCanvas

app = QtWidgets.QApplication([])

canvas = WgpuCanvas()
renderer = gfx.WgpuRenderer(canvas)
renderer_svg = gfx.SvgRenderer(640, 480, "~/line.svg")

scene = gfx.Scene()

positions = [[200 + np.sin(i) * i * 6, 200 + np.cos(i) * i * 6] for i in range(20)]
positions += [[400 - np.sin(i) * i * 6, 200 + np.cos(i) * i * 6] for i in range(20)]
positions += [
    [450, 400],
    [375, 400],
    [300, 400],
    [400, 370],
    [300, 340],
]

# Our data must be Nx4 (for now?)
positions = [pos + [0, 1] for pos in positions]
geometry = gfx.Geometry(positions=positions)

# Spiral away in z (to make the depth buffer less boring)
for i in range(len(positions)):
    positions[i][2] = i

material = gfx.LineMaterial(thickness=12.0, color=(0.8, 0.7, 0.0, 1.0))
line = gfx.Line(geometry, material)
scene.add(line)

camera = gfx.OrthographicCamera(600, 500)
camera.position.set(300, 250, 0)


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
