"""
Render three volumes using different world transforms.
"""

import imageio
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
        drag_start(
            (event.x(), event.y()),
            self.get_logical_size(),
            camera,
        )
        app.setOverrideCursor(QtCore.Qt.ClosedHandCursor)

        # Picking. Note that this works on both the volume and the slice.
        if event.modifiers() and QtCore.Qt.Key_Shift:
            info = renderer.get_pick_info((event.x(), event.y()))
            if "voxel_index" in info:
                x, y, z = (max(1, int(i)) for i in info["voxel_index"])
                print("Picking", x, y, z)
                tex.data[z - 1 : z + 1, y - 1 : y + 1, x - 1 : x + 1] = 2000
                tex.update_range((x - 1, y - 1, z - 1), (3, 3, 3))

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

voldata = imageio.volread("imageio:stent.npz").astype(np.float32)

tex = gfx.Texture(voldata, dim=3)
vol1 = gfx.Volume(tex, gfx.VolumeRayMaterial(clim=(0, 2000)))
vol2 = gfx.Volume(tex, gfx.VolumeRayMaterial(clim=(0, 2000)))
vol3 = gfx.Volume(tex, gfx.VolumeRayMaterial(clim=(0, 2000)))
scene.add(vol1, vol2, vol3)

vol2.position.x = -150
vol2.scale.z = 0.5

vol3.position.x = 150

camera = gfx.PerspectiveCamera(70, 16 / 9)
camera.position.y = 500
controls = gfx.OrbitControls(camera.position.clone(), up=gfx.linalg.Vector3(0, 0, 1))
controls.rotate(-0.5, -0.5)


def animate():
    controls.update_camera(camera)
    rot = gfx.linalg.Quaternion().set_from_euler(gfx.linalg.Euler(0.005, 0.01))
    vol3.rotation.multiply(rot)

    renderer.render(scene, camera)
    canvas.request_draw()


if __name__ == "__main__":
    canvas.request_draw(animate)
    app.exec()
