"""
Example showing orbit camera controls.
"""

import numpy as np
import imageio
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

scene.add(gfx.AxesHelper(size=250))

im = imageio.imread("imageio:chelsea.png").astype(np.float32) / 255
im = np.concatenate([im, np.ones(im.shape[:2] + (1,), dtype=im.dtype)], 2)
tex = gfx.Texture(im, dim=2, usage="sampled").get_view(filter="linear")

material = gfx.MeshBasicMaterial(map=tex, clim=(0.2, 0.8))
geometry = gfx.BoxGeometry(100, 100, 100)
cubes = [gfx.Mesh(geometry, material) for i in range(8)]
for i, cube in enumerate(cubes):
    cube.position.set(350 - i * 100, 150, 0)
    scene.add(cube)

background = gfx.Background(gfx.BackgroundMaterial((0, 1, 0, 1), (0, 1, 1, 1)))
scene.add(background)

camera = gfx.PerspectiveCamera(70, 16 / 9)
camera.position.set(0, 0, 500)
controls = gfx.OrbitControls(camera.position.clone())


def animate():
    for i, cube in enumerate(cubes):
        rot = gfx.linalg.Quaternion().set_from_euler(
            gfx.linalg.Euler(0.005 * i, 0.01 * i)
        )
        cube.rotation.multiply(rot)

    controls.update_camera(camera)

    renderer.render(scene, camera)
    canvas.request_draw()


if __name__ == "__main__":
    canvas.request_draw(animate)
    app.exec_()
