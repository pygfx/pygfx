"""
Example showing different types of geometric cylinders.
"""

import numpy as np
import pygfx as gfx

from PySide6 import QtWidgets, QtCore
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
        drag_start(
            (event.position().x(), event.position().y()),
            self.get_logical_size(),
            camera,
        )
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
            drag_move((event.position().x(), event.position().y()))


app = QtWidgets.QApplication([])

canvas = WgpuCanvasWithInputEvents()
renderer = gfx.renderers.WgpuRenderer(canvas)
scene = gfx.Scene()

spheres = [
    (
        (0, 0, -7.5),
        (0, 0.65, 0, 1),
        gfx.sphere_geometry(4.5, phi_length=np.pi * 1.5),
    ),
    ((0, 0, 7.5), (1, 1, 1, 1), gfx.sphere_geometry(4)),
    (
        (15, 0, -7.5),
        (1, 0.3, 0.3, 1),
        gfx.sphere_geometry(4, theta_start=np.pi * 0.25, theta_length=np.pi * 0.50),
    ),
    (
        (15, 0, 7.5),
        (0.35, 0, 0, 1),
        gfx.sphere_geometry(5, width_segments=6),
    ),
    ((-15, 0, -7.5), (1, 1, 0.75, 1), gfx.sphere_geometry(7)),
    ((-15, 0, 7.5), (1, 1, 0.75, 1), gfx.sphere_geometry(5, height_segments=8)),
]
for pos, color, geometry in spheres:
    material = gfx.MeshFlatMaterial(color=color)
    wobject = gfx.Mesh(geometry, material)
    wobject.position.set(*pos)
    scene.add(wobject)

camera = gfx.PerspectiveCamera(70, 16 / 9)
camera.position.set(6, 16, -22)
controls = gfx.OrbitControls(camera.position.clone())


def animate():
    controls.update_camera(camera)

    renderer.render(scene, camera)
    canvas.request_draw()


if __name__ == "__main__":
    canvas.request_draw(animate)
    app.exec()
