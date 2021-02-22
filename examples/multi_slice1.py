"""
Slice a volume and a mesh through the three primary planes (XY, XZ, YZ).
This example uses a mesh object with custom texture coordinates. This
is a generic approach. See multi_slice2.py for a simpler way.
"""

from time import time

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

background = gfx.Background(gfx.BackgroundMaterial((0, 1, 0, 1), (0, 1, 1, 1)))
scene.add(background)

scene.add(gfx.AxesHelper(length=50))

vol = imageio.volread("imageio:stent.npz")
tex = gfx.Texture(vol, dim=3, usage="sampled")
view = tex.get_view(filter="linear")
material = gfx.MeshBasicMaterial(map=view, clim=(0, 255))

# TODO: also add a mesh slice for each plane

planes = []
texcoords = {
    0: [[0.5, 0, 0], [0.5, 1, 0], [0.5, 0, 1], [0.5, 1, 1]],
    1: [[0, 0.5, 0], [1, 0.5, 0], [0, 0.5, 1], [1, 0.5, 1]],
    2: [[0, 0, 0.5], [1, 0, 0.5], [0, 1, 0.5], [1, 1, 0.5]],
}
sizes = {
    0: (vol.shape[1], vol.shape[0]),  # YZ plane
    1: (vol.shape[2], vol.shape[0]),  # XZ plane
    2: (vol.shape[2], vol.shape[1]),  # XY plane (default)
}
for axis in [0, 1, 2]:
    geometry = gfx.PlaneGeometry(*sizes[axis], 1, 1)
    geometry.texcoords = gfx.Buffer(
        np.array(texcoords[axis], dtype="f4"), usage="vertex|storage"
    )
    plane = gfx.Mesh(geometry, material)
    planes.append(plane)
    scene.add(plane)

    if axis == 0:  # YZ plane
        plane.rotation.set_from_euler(gfx.linalg.Euler(0.5 * np.pi, 0.5 * np.pi))
    elif axis == 1:  # XZ plane
        plane.rotation.set_from_euler(gfx.linalg.Euler(0.5 * np.pi))
    # else: XY plane

# camera = gfx.PerspectiveCamera(70, 16 / 9)
camera = gfx.OrthographicCamera(200, 200)
camera.position.set(125, 125, 125)
camera.look_at(gfx.linalg.Vector3())
controls = gfx.OrbitControls(
    camera.position.clone(), up=gfx.linalg.Vector3(0, 0, 1), zoom_changes_distance=False
)


def animate():
    t = np.cos(time() / 2)
    plane = planes[2]
    plane.position.z = t * vol.shape[0] * 0.5
    plane.geometry.texcoords.data[:, 2] = (t + 1) / 2
    plane.geometry.texcoords.update_range(0, plane.geometry.texcoords.nitems)

    controls.update_camera(camera)
    renderer.render(scene, camera)
    canvas.request_draw()


if __name__ == "__main__":
    canvas.request_draw(animate)
    app.exec_()
