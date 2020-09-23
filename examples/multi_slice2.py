"""
Slice a volume and a mesh through the three primary planes (XY, XZ, YZ).
This example uses Volume object with a VolumeSliceMaterial, which
produces an implicit geometry defined by the volume data.
See multi_slice1.py for a more generic approach.
"""

from time import time

import imageio
import numpy as np
import pygfx as gfx
from skimage.measure import marching_cubes

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

scene.add(gfx.AxesHelper(size=50))

vol = imageio.volread("imageio:stent.npz")
tex = gfx.Texture(vol, dim=3, usage="sampled")

surface = marching_cubes(vol[0:], 200)
positions = np.fliplr(surface[0])
positions = np.column_stack([positions, np.ones((positions.shape[0], 1), np.float32)])
geo = gfx.Geometry(positions=positions, index=surface[1], normals=surface[2])
mesh = gfx.Mesh(
    geo, gfx.MeshSliceMaterial(plane=(0, 0, -1, vol.shape[0] / 2), color=(1, 1, 0, 1))
)
scene.add(mesh)

planes = []
for dim in [0, 1, 2]:  # xyz
    abcd = [0, 0, 0, 0]
    abcd[dim] = -1
    abcd[-1] = vol.shape[2 - dim] / 2
    material = gfx.VolumeSliceMaterial(map=tex, clim=(0, 255), plane=abcd)
    plane = gfx.Volume(tex.size, material)
    planes.append(plane)
    scene.add(plane)


# camera = gfx.PerspectiveCamera(70, 16 / 9)
camera = gfx.OrthographicCamera(200, 200)
camera.position.set(170, 170, 170)
controls = gfx.OrbitControls(
    camera.position.clone(),
    gfx.linalg.Vector3(64, 64, 128),
    up=gfx.linalg.Vector3(0, 0, 1),
    zoom_changes_distance=False,
)

# Add a slight tilt. This is to show that the slices are still orthogonal
# to the world coordinates.
for ob in planes + [mesh]:
    ob.rotation.set_from_axis_angle(gfx.linalg.Vector3(1, 0, 0), 0.1)


def animate():
    t = np.cos(time() / 2) * 0.5 + 0.5  # 0..1
    planes[2].material.plane = 0, 0, -1, t * vol.shape[0]
    mesh.material.plane = 0, 0, -1, (1 - t) * vol.shape[0]

    controls.update_camera(camera)
    renderer.render(scene, camera)
    canvas.request_draw()


if __name__ == "__main__":
    canvas.request_draw(animate)
    app.exec_()
