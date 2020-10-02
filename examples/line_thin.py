"""
Some thin line drawing.
"""

import numpy as np

import pygfx as gfx

from PyQt5 import QtWidgets
from wgpu.gui.qt import WgpuCanvas

app = QtWidgets.QApplication([])

canvas = WgpuCanvas()
renderer = gfx.WgpuRenderer(canvas)

scene = gfx.Scene()

positions = [[200 + np.sin(i) * i * 6, 200 + np.cos(i) * i * 6, 0] for i in range(20)]
positions += [[400 - np.sin(i) * i * 6, 200 + np.cos(i) * i * 6, 0] for i in range(20)]
positions += [
    [450, 400, 0],
    [375, 400, 0],
    [300, 400, 0],
    [400, 370, 0],
    [300, 340, 0],
]

# Spiral away in z (to make the depth buffer less boring)
for i in range(len(positions)):
    positions[i][2] = i

colors = np.array(positions.copy())
colors /= colors.max()
colors = np.hstack([colors, np.ones((colors.shape[0], 1))])
colors = colors.astype("f4")

geometry = gfx.Geometry(positions=positions, colors=colors)
material = gfx.LineThinMaterial(thickness=12.0, vertex_colors=True)
line = gfx.Line(geometry, material)
scene.add(line)

camera = gfx.OrthographicCamera(600, 500)
camera.position.set(300, 250, 0)


if __name__ == "__main__":
    canvas.request_draw(lambda: renderer.render(scene, camera))
    app.exec_()
