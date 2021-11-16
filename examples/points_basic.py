import numpy as np
import pygfx as gfx

from PySide6 import QtWidgets
from wgpu.gui.qt import WgpuCanvas

app = QtWidgets.QApplication([])

canvas = WgpuCanvas()
renderer = gfx.WgpuRenderer(canvas)

scene = gfx.Scene()

positions = np.random.normal(0, 0.5, (100, 3)).astype(np.float32)
sizes = np.random.rand(100).astype(np.float32) * 50
colors = np.random.rand(100, 4).astype(np.float32)
geometry = gfx.Geometry(positions=positions, sizes=sizes, colors=colors)

material = gfx.PointsMaterial(vertex_colors=True, vertex_sizes=True)
points = gfx.Points(geometry, material)
scene.add(points)

scene.add(
    gfx.Background(None, gfx.BackgroundMaterial((0.2, 0.0, 0, 1), (0, 0.0, 0.2, 1)))
)

camera = gfx.NDCCamera()


if __name__ == "__main__":
    canvas.request_draw(lambda: renderer.render(scene, camera))
    app.exec()
