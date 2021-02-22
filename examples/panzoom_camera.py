"""
Example showing orbit camera controls.
"""

import imageio
import pygfx as gfx

from PyQt5 import QtWidgets, QtCore
from wgpu.gui.qt import WgpuCanvas


class WgpuCanvasWithInputEvents(WgpuCanvas):
    _drag_modes = {QtCore.Qt.RightButton: "pan"}
    _mode = None

    def wheelEvent(self, event):  # noqa: N802
        zoom_multiplier = 2 ** (event.angleDelta().y() * 0.0015)
        controls.zoom_to_point(
            zoom_multiplier, (event.x(), event.y()), self.get_logical_size(), camera
        )
        canvas.request_draw()

    def mousePressEvent(self, event):  # noqa: N802
        mode = self._drag_modes.get(event.button(), None)
        if self._mode or not mode:
            return
        self._mode = mode
        controls.pan_start((event.x(), event.y()), self.get_logical_size(), camera)
        app.setOverrideCursor(QtCore.Qt.ClosedHandCursor)

    def mouseReleaseEvent(self, event):  # noqa: N802
        if self._mode and self._mode == self._drag_modes.get(event.button(), None):
            self._mode = None
            controls.pan_stop()
            app.restoreOverrideCursor()
        canvas.request_draw()

    def mouseMoveEvent(self, event):  # noqa: N802
        if self._mode is not None:
            controls.pan_move((event.x(), event.y()))
        canvas.request_draw()


app = QtWidgets.QApplication([])
canvas = WgpuCanvasWithInputEvents()
renderer = gfx.renderers.WgpuRenderer(canvas)
scene = gfx.Scene()

axes = gfx.AxesHelper(length=250)
scene.add(axes)

background = gfx.Background(gfx.BackgroundMaterial((0, 1, 0, 1), (0, 1, 1, 1)))
scene.add(background)

im = imageio.imread("imageio:astronaut.png")
tex = gfx.Texture(im, dim=2, usage="sampled")
geometry = gfx.PlaneGeometry(512, 512)
material = gfx.MeshBasicMaterial(map=tex.get_view(filter="linear"), clim=(0, 255))
plane = gfx.Mesh(geometry, material)
plane.scale.y = -1
scene.add(plane)

camera = gfx.OrthographicCamera(512, 512)
camera.position.set(0, 0, 500)
controls = gfx.PanZoomControls(camera.position.clone())


def animate():
    controls.update_camera(camera)
    renderer.render(scene, camera)
    # canvas.request_draw()  # not needed it we request a draw on user interaction


if __name__ == "__main__":
    canvas.request_draw(animate)
    app.exec_()
