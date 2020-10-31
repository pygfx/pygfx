"""
Example showing a single geometric cylinder.
"""

import numpy as np
import pygfx as gfx

from PyQt5 import QtWidgets, QtCore
from wgpu.gui.qt import WgpuCanvas


class WgpuCanvasWithInputEvents(WgpuCanvas):
    _drag_modes = {QtCore.Qt.RightButton: "pan", QtCore.Qt.LeftButton: "rotate"}
    _mode = None

    def wheelEvent(self, event):  # noqa: N802
        controls.zoom(2 ** (event.angleDelta().y() * 0.0015))

    def mousePressEvent(self, event):  # noqa: N802
        mode = self._drag_modes.get(event.button(), None)
        if self._mode or not mode:
            return
        self._mode = mode
        drag_start = (
            controls.pan_start if self._mode == "pan" else controls.rotate_start
        )
        drag_start((event.x(), event.y()), self.get_logical_size(), camera)
        app.setOverrideCursor(QtCore.Qt.ClosedHandCursor)

    def mouseReleaseEvent(self, event):  # noqa: N802
        if self._mode and self._mode == self._drag_modes.get(event.button(), None):
            self._mode = None
            drag_stop = (
                controls.pan_stop if self._mode == "pan" else controls.rotate_stop
            )
            drag_stop()
            app.restoreOverrideCursor()

    def mouseMoveEvent(self, event):  # noqa: N802
        if self._mode is not None:
            drag_move = (
                controls.pan_move if self._mode == "pan" else controls.rotate_move
            )
            drag_move((event.x(), event.y()))


app = QtWidgets.QApplication([])

canvas = WgpuCanvasWithInputEvents()
renderer = gfx.renderers.WgpuRenderer(canvas)
scene = gfx.Scene()

cylinders = [
    (
        (0, 0, -32.5),
        (0, 0.65, 0, 1),
        gfx.CylinderGeometry(20, 12, radial_segments=16, height=25, open_ended=True),
    ),
    ((0, 0, 25), (1, 1, 1, 1), gfx.CylinderGeometry(10, 10, height=25)),
    (
        (30, 0, 25),
        (1, 0.3, 0.3, 1),
        gfx.CylinderGeometry(
            10,
            10,
            height=12,
            theta_start=np.pi * 1.3,
            theta_length=np.pi * 1.5,
            open_ended=True,
        ),
    ),
    (
        (-50, 0, 0),
        (0.35, 0, 0, 1),
        gfx.CylinderGeometry(
            20, 12, radial_segments=3, height_segments=4, height=10, open_ended=True
        ),
    ),
    ((50, 0, -10), (1, 1, 0.75, 1), gfx.CylinderGeometry(1.5, 1.5, height=20)),
    ((50, 0, 5), (1, 1, 0.75, 1), gfx.CylinderGeometry(4, 0.0, height=10)),
]
for pos, color, geometry in cylinders:
    material = gfx.MeshPhongMaterial(color=color)
    wobject = gfx.Mesh(geometry, material)
    wobject.position.set(*pos)
    scene.add(wobject)

    material = gfx.MeshNormalLinesMaterial(color=color)
    wobject = gfx.Mesh(geometry, material)
    wobject.position.set(*pos)
    scene.add(wobject)

camera = gfx.PerspectiveCamera(70, 16 / 9)
camera.position.set(0, -65, 50)
controls = gfx.OrbitControls(camera.position.clone())


def animate():
    controls.update_camera(camera)

    renderer.render(scene, camera)
    canvas.request_draw()


if __name__ == "__main__":
    canvas.request_draw(animate)
    app.exec_()
