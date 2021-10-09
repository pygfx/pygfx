"""
Display a lot of line objects. Because of the architecture of wgpu,
this is still performant.
"""

import time  # noqa
import numpy as np

import pygfx as gfx

from PySide6 import QtWidgets, QtCore
from wgpu.gui.qt import WgpuCanvas


class WgpuCanvasWithInputEvents(WgpuCanvas):
    _drag_modes = {QtCore.Qt.RightButton: "pan"}
    _mode = None

    def wheelEvent(self, event):  # noqa: N802
        zoom_multiplier = 2 ** (event.angleDelta().y() * 0.0015)
        controls.zoom_to_point(
            zoom_multiplier,
            (event.position().x(), event.position().y()),
            self.get_logical_size(),
            camera,
        )
        self.request_draw()

    def mousePressEvent(self, event):  # noqa: N802
        mode = self._drag_modes.get(event.button(), None)
        if self._mode or not mode:
            return
        self._mode = mode
        controls.pan_start(
            (event.position().x(), event.position().y()),
            self.get_logical_size(),
            camera,
        )
        app.setOverrideCursor(QtCore.Qt.ClosedHandCursor)

    def mouseReleaseEvent(self, event):  # noqa: N802
        if self._mode and self._mode == self._drag_modes.get(event.button(), None):
            self._mode = None
            controls.pan_stop()
            app.restoreOverrideCursor()

    def mouseMoveEvent(self, event):  # noqa: N802
        if self._mode is not None:
            controls.pan_move((event.position().x(), event.position().y()))
        self.request_draw()


app = QtWidgets.QApplication([])
canvas = WgpuCanvasWithInputEvents()
renderer = gfx.WgpuRenderer(canvas, show_fps=True)

canvas._target_fps = 9999  # don't limit fps

scene = gfx.Scene()

# Define number of vertices
cols = 20
rows = 50
nvertices = 30000
use_thin_lines = True

print(nvertices * rows * cols, "vertices in total")

x = np.linspace(0.05, 0.95, nvertices, dtype=np.float32)

for row in range(rows):
    for col in range(cols):
        y = np.sin(x * 25) * 0.45 + np.random.normal(0, 0.02, len(x)).astype(np.float32)
        positions = np.column_stack([x, y, np.zeros_like(x)])
        geometry = gfx.Geometry(positions=positions)
        if use_thin_lines:
            material = gfx.LineThinMaterial(color=(col / cols, row / rows, 0.5, 1.0))
        else:
            material = gfx.LineMaterial(
                thickness=0.2 + 2 * row / rows, color=(col / cols, row / rows, 0.5, 1.0)
            )
        line = gfx.Line(geometry, material)
        line.position.x = col
        line.position.y = row
        scene.add(line)

camera = gfx.OrthographicCamera(cols, rows)
camera.maintain_aspect = False
controls = gfx.PanZoomControls(camera.position.clone())
controls.pan(gfx.linalg.Vector3(cols / 2, rows / 2, 0))


def animate():
    controls.update_camera(camera)
    t0 = time.perf_counter()  # noqa
    renderer.render(scene, camera)
    # print(time.perf_counter() - t0)
    canvas.request_draw()


if __name__ == "__main__":
    canvas.request_draw(animate)
    app.exec()
