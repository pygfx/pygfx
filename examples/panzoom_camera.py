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
        # compute current mouse position relative to widget center
        dim = (self.width(), self.height())
        pos = (event.x(), event.y())
        fracpos = tuple((pos[i] - dim[i] * 0.5) / dim[i] for i in range(2))
        view_old = (camera.right - camera.left, camera.top - camera.bottom)
        relpos_old = tuple(fracpos[i] * view_old[i] for i in range(2))
        # apply zoom and compute change in dimensions of camera frustum
        controls.zoom(event.angleDelta().y())
        controls.update_camera(camera)
        camera.update_bounds()
        view = (camera.right - camera.left, camera.top - camera.bottom)
        # compute new mouse position relative to widget center
        relpos = tuple(fracpos[i] * view[i] for i in range(2))
        # compute delta and pan accordingly
        delta = tuple(relpos[i] - relpos_old[i] for i in range(2))
        controls.pan(*delta)

    def mousePressEvent(self, event):  # noqa: N802
        button = event.button()
        mode = self._drag_modes.get(button, None)
        if self.drag or not mode:
            return  # drag is already initiated, or unknown button pressed
        self.drag = {
            "mode": mode,
            "button": button,
            "start": (event.x(), event.y()),
        }
        QtWidgets.QApplication.instance().setOverrideCursor(QtCore.Qt.BlankCursor)

    def mouseReleaseEvent(self, event):  # noqa: N802
        if self.drag and self.drag.get("button") == event.button():
            # stop when the initiating button is released
            self.drag = None
            QtWidgets.QApplication.instance().restoreOverrideCursor()

    def mouseMoveEvent(self, event):  # noqa: N802
        if not self.drag:
            return
        mouse_end = (event.x(), event.y())
        delta = tuple(mouse_end[i] - self.drag["start"][i] for i in range(2))
        getattr(controls, self.drag["mode"])(*delta)
        self.drag["start"] = mouse_end


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
controls = gfx.OrbitControls(camera.position.clone(), zoom="zoom")


def animate():
    controls.update_camera(camera)

    renderer.render(scene, camera)
    canvas.request_draw()


if __name__ == "__main__":
    canvas.request_draw(animate)
    app.exec_()
