import visvis2 as vv
import numpy as np

from PyQt5 import QtWidgets
from wgpu.gui.qt import WgpuCanvas

app = QtWidgets.QApplication([])

canvas = WgpuCanvas()
renderer = vv.WgpuSurfaceRenderer(canvas)

scene = vv.Scene()

positions = np.array(
    [
        [10, 10, 0, 1],
        [20, 20, 0, 1],
        [30, 40, 0, 1],
        [40, 30, 0, 1],
        [50, 50, 0, 1],
        [60, 50, 0, 1],
        [70, 30, 0, 1],
    ],
    np.float32,
)
geometry = vv.Geometry()
geometry.positions = vv.BufferWrapper(positions * 5, mapped=True)


material = vv.LineStripMaterial()
line = vv.Mesh(geometry, material)  # Mesh??
scene.add(line)

camera = vv.ScreenCoordsCamera()


def animate():
    # would prefer to do this in a resize event only
    width, height, ratio = canvas.get_size_and_pixel_ratio()
    camera.set_viewport_size(width, height)
    renderer.render(scene, camera)
    canvas.update()


if __name__ == "__main__":
    canvas.draw_frame = animate
    app.exec_()
