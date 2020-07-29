"""
Example showing orbit camera controls.
"""

import numpy as np
import imageio
import pygfx as gfx

from PyQt5 import QtWidgets, QtCore
from wgpu.gui.qt import WgpuCanvas


class WgpuCanvasWithInputEvents(WgpuCanvas):
    _drag_modes = {QtCore.Qt.RightButton: "pan"}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.drag = None

    def wheelEvent(self, event):  # noqa: N802
        dim = self.width(), self.height()
        pos = event.x(), event.y()
        delta = event.angleDelta().y() * 0.00125
        controls.zoom_to_point(delta, pos, dim, camera)

    def mousePressEvent(self, event):  # noqa: N802
        button = event.button()
        mode = self._drag_modes.get(button, None)
        if self.drag or not mode:
            return
        pos = event.x(), event.y()
        self.drag = {"mode": mode, "button": button, "start": pos}
        app.setOverrideCursor(QtCore.Qt.BlankCursor)

    def mouseReleaseEvent(self, event):  # noqa: N802
        if self.drag and self.drag.get("button") == event.button():
            self.drag = None
            app.restoreOverrideCursor()

    def mouseMoveEvent(self, event):  # noqa: N802
        if not self.drag:
            return
        pos = event.x(), event.y()
        delta = tuple(pos[i] - self.drag["start"][i] for i in range(2))
        getattr(controls, self.drag["mode"])(*delta)
        self.drag["start"] = pos


app = QtWidgets.QApplication([])
canvas = WgpuCanvasWithInputEvents()
renderer = gfx.renderers.WgpuRenderer(canvas)
scene = gfx.Scene()

axes = gfx.AxesHelper(size=250)
scene.add(axes)

background = gfx.Background(gfx.BackgroundMaterial((0, 1, 0, 1), (0, 1, 1, 1)))
scene.add(background)

im = imageio.imread("imageio:astronaut.png")
im = np.concatenate([im, 255 * np.ones(im.shape[:2] + (1,), dtype=im.dtype)], 2)  # yuk!
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
    canvas.request_draw()


if __name__ == "__main__":
    canvas.request_draw(animate)
    app.exec_()
