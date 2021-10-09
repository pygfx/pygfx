"""
Display a line depicting a noisy signal consisting of a lot of points.
"""

import numpy as np

import pygfx as gfx

from PySide6 import QtWidgets
from wgpu.gui.qt import WgpuCanvas

app = QtWidgets.QApplication([])

canvas = WgpuCanvas()
renderer = gfx.WgpuRenderer(canvas)

scene = gfx.Scene()

# todo: crank this to 1M when wgpu allows it :D
x = np.linspace(0, 100, 10_000, dtype=np.float32)
y = np.sin(x) * 30 + np.random.normal(0, 5, len(x)).astype(np.float32)

positions = np.column_stack([x, y, np.zeros_like(x)])
geometry = gfx.Geometry(positions=positions)

material = gfx.LineMaterial(thickness=2.0, color=(0.0, 0.7, 0.3, 1.0))
line = gfx.Line(geometry, material)
scene.add(line)


camera = gfx.OrthographicCamera(110, 110)
camera.position.set(50, 0, 0)


if __name__ == "__main__":
    canvas.request_draw(lambda: renderer.render(scene, camera))
    app.exec()
