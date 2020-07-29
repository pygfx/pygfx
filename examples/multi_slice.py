"""
Slice a volume and a mesh through the three primary planes (XY, XZ, YZ)
"""

import imageio
import numpy as np
import pygfx as gfx

from PyQt5 import QtWidgets, QtCore
from wgpu.gui.qt import WgpuCanvas


class WgpuCanvasWithInputEvents(WgpuCanvas):
    _drag_modes = {QtCore.Qt.RightButton: "pan", QtCore.Qt.LeftButton: "rotate"}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.drag = None

    def wheelEvent(self, event):  # noqa: N802
        controls.zoom(event.angleDelta().y())

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

background = gfx.Background(gfx.BackgroundMaterial((0, 1, 0, 1), (0, 1, 1, 1)))
scene.add(background)

scene.add(gfx.AxesHelper(size=50))

vol = imageio.volread("imageio:stent.npz")
tex = gfx.Texture(vol, dim=3, usage="sampled")
view = tex.get_view(filter="linear")
material = gfx.MeshVolumeSliceMaterial(map=view, clim=(0, 255))

# TODO: also add a mesh slice for each plane

for axis in [0, 1, 2]:
    if axis == 0:  # YZ plane
        geometry = gfx.PlaneGeometry(vol.shape[1], vol.shape[0], 1, 1)
        texcoords = np.array(
            [[0.5, 0, 0], [0.5, 1, 0], [0.5, 0, 1], [0.5, 1, 1]], dtype=np.float32,
        )
    elif axis == 1:  # XZ plane
        geometry = gfx.PlaneGeometry(vol.shape[2], vol.shape[0], 1, 1)
        texcoords = np.array(
            [[0, 0.5, 0], [1, 0.5, 0], [0, 0.5, 1], [1, 0.5, 1]], dtype=np.float32,
        )
    elif axis == 2:  # XY plane (default)
        geometry = gfx.PlaneGeometry(vol.shape[2], vol.shape[1], 1, 1)
        texcoords = np.array(
            [[0, 0, 0.5], [1, 0, 0.5], [0, 1, 0.5], [1, 1, 0.5]], dtype=np.float32,
        )
    geometry.texcoords = gfx.Buffer(texcoords, usage="vertex|storage")

    plane = gfx.Mesh(geometry, material)

    # by default the plane is in XY plane
    if axis == 0:  # YZ plane
        plane.rotation.set_from_euler(gfx.linalg.Euler(0.5 * np.pi, 0.5 * np.pi))
    elif axis == 1:  # XZ plane
        plane.rotation.set_from_euler(gfx.linalg.Euler(0.5 * np.pi))
    scene.add(plane)

camera = gfx.PerspectiveCamera(70, 16 / 9)
camera.position.set(125, 125, 125)
camera.look_at(gfx.linalg.Vector3())
controls = gfx.OrbitControls(camera.position.clone(), up=gfx.linalg.Vector3(0, 0, 1))


def animate():
    # todo: should the control set it directly?
    controls.update_camera(camera)
    renderer.render(scene, camera)
    canvas.request_draw()


if __name__ == "__main__":
    canvas.request_draw(animate)
    app.exec_()
