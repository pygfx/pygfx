"""
Example demonstrating clipping planes on a mesh.
"""

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


# Create a canvas and a renderer

app = QtWidgets.QApplication([])
canvas = WgpuCanvasWithInputEvents(size=(800, 400))
renderer = gfx.renderers.WgpuRenderer(canvas)

# Compose two of the same scenes


def create_scene(clipping_planes, clipping_mode):

    maxsize = 221
    scene = gfx.Scene()
    for n in range(20, maxsize, 50):
        material = gfx.MeshPhongMaterial(
            color=(n / maxsize, 1, 0, 1),
            clipping_planes=clipping_planes,
            clipping_mode=clipping_mode,
        )
        geometry = gfx.BoxGeometry(n, n, n)
        cube = gfx.Mesh(geometry, material)
        scene.add(cube)

    return scene


clipping_planes = [(-1, 0, 0, 0), (0, 0, -1, 0)]
scene1 = create_scene(clipping_planes, "any")
scene2 = create_scene(clipping_planes, "all")

camera = gfx.PerspectiveCamera(70, 16 / 9)
camera.position.z = 250

controls = gfx.OrbitControls(camera.position.clone())


def animate():

    controls.update_camera(camera)

    w, h = canvas.get_logical_size()
    renderer.render(scene1, camera, flush=False, viewport=(0, 0, w / 2, h))
    renderer.render(scene2, camera, flush=False, viewport=(w / 2, 0, w / 2, h))
    renderer.flush()

    canvas.request_draw()


if __name__ == "__main__":
    canvas.request_draw(animate)
    app.exec_()
