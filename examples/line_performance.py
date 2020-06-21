"""
Display a line depicting a noisy signal consisting of a lot of points.
"""

import numpy as np

import pygfx as gfx

from PyQt5 import QtWidgets
from wgpu.gui.qt import WgpuCanvas

app = QtWidgets.QApplication([])

canvas = WgpuCanvas()
renderer = gfx.WgpuRenderer(canvas)

scene = gfx.Scene()

# todo: crank this to 1M when wgpu allows it :D
x = np.linspace(20, 620, 10_000, dtype=np.float32)
y = np.sin(x / 10) * 100 + 200 + np.random.normal(0, 5, len(x)).astype(np.float32)

positions = np.column_stack([x, y, np.zeros_like(x), np.ones_like(x)])
geometry = gfx.Geometry(positions=positions)

material = gfx.LineStripMaterial(thickness=3.0, color=(0.0, 0.7, 0.3, 1.0))
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
