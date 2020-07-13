"""
Example showing multiple rotating cubes. This also tests the depth buffer.
"""

import numpy as np
import imageio
import pygfx as gfx

from PyQt5 import QtWidgets, QtCore
from wgpu.gui.qt import WgpuCanvas


class OrbitControls:
    _m = gfx.linalg.Matrix4()
    _v = gfx.linalg.Vector3()
    _q1 = gfx.linalg.Quaternion()
    _q2 = gfx.linalg.Quaternion()
    _e = gfx.linalg.Euler()

    def __init__(
        self,
        eye: gfx.linalg.Vector3 = None,
        target: gfx.linalg.Vector3 = None,
        up: gfx.linalg.Vector3 = None,
    ) -> None:
        self.rotation = gfx.linalg.Quaternion()
        if eye is None:
            eye = gfx.linalg.Vector3(50.0, 50.0, 50.0)
        if target is None:
            target = gfx.linalg.Vector3()
        if up is None:
            up = gfx.linalg.Vector3(0.0, 1.0, 0.0)
        self.look_at(eye, target, up)

    def look_at(
        self,
        eye: gfx.linalg.Vector3,
        target: gfx.linalg.Vector3,
        up: gfx.linalg.Vector3,
    ) -> "OrbitControls":
        self.rotation.set_from_rotation_matrix(self._m.look_at(eye, target, up))
        self.target = target
        self.distance = eye.distance_to(target)
        self.up = up
        return self

    def pan(self, x: float, y: float) -> "OrbitControls":
        self._v.set(-x, y).apply_quaternion(self.rotation)
        self.target.add(self._v)
        return self

    def rotate(
        self, cur_x: float, cur_y: float, prev_x: float, prev_y: float
    ) -> "OrbitControls":
        self._q1.set_from_euler(self._e.set(-cur_x, cur_y, 0))
        self._q2.set_from_euler(self._e.set(-prev_x, prev_y, 0))
        self._q2.inverse()
        self._q1.multiply(self._q2)
        if self._q1.length() < 1e-6:
            return
        self.rotation.multiply(self._q1)
        self.rotation.normalize()
        return self

    def zoom(self, delta: float) -> "OrbitControls":
        self.distance += delta
        if self.distance < 0:
            self.distance = 0
        return self

    def get_view(self) -> (gfx.linalg.Vector3, gfx.linalg.Vector3):
        rot = self.rotation.clone().conjugate()
        pos = (
            gfx.linalg.Vector3(0, 0, -self.distance)
            .apply_quaternion(self.rotation)
            .sub(self.target)
        )
        return rot, pos


class WgpuCanvasWithInputEvents(WgpuCanvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mouse_start = None
        self.pan = False

    def wheelEvent(self, event):  # noqa: N802
        # TODO: emit inputs from canvas
        # TODO: link canvas to controls in glue layer instead of here
        global controls
        degrees = event.angleDelta().y() / 8
        controls.zoom(degrees)

    def mousePressEvent(self, event):  # noqa: N802
        pos = event.windowPos()
        self.pan = event.button() == QtCore.Qt.RightButton
        self.mouse_start = np.array([pos.x(), pos.y()])

    def mouseReleaseEvent(self, event):  # noqa: N802
        self.mouse_start = None

    def mouseMoveEvent(self, event):  # noqa: N802
        global controls
        pos = event.windowPos()
        mouse_end = np.array([pos.x(), pos.y()])
        if self.pan:
            controls.pan(*(mouse_end - self.mouse_start))
        else:
            controls.rotate(*mouse_end, *self.mouse_start)
        self.mouse_start = mouse_end

    def keyPressEvent(self, event):  # noqa: N802
        pass

    def keyReleaseEvent(self, event):  # noqa: N802
        pass


app = QtWidgets.QApplication([])

canvas = WgpuCanvasWithInputEvents()
renderer = gfx.renderers.WgpuRenderer(canvas)
scene = gfx.Scene()

im = imageio.imread("imageio:chelsea.png").astype(np.float32) / 255
im = np.concatenate([im, np.ones(im.shape[:2] + (1,), dtype=im.dtype)], 2)
tex = gfx.Texture(im, dim=2, usage="sampled").get_view(filter="linear")

material = gfx.MeshBasicMaterial(map=tex, clim=(0.2, 0.8))
geometry = gfx.BoxGeometry(100, 100, 100)
cubes = [gfx.Mesh(geometry, material) for i in range(8)]
for i, cube in enumerate(cubes):
    cube.position.set(350 - i * 100, 0, 0)
    scene.add(cube)

background = gfx.Background(gfx.BackgroundMaterial((0, 1, 0, 1), (0, 1, 1, 1)))
scene.add(background)

camera = gfx.PerspectiveCamera(70, 16 / 9)
camera.position.z = 500
controls = OrbitControls(camera.position.clone())


def animate():
    for i, cube in enumerate(cubes):
        rot = gfx.linalg.Quaternion().set_from_euler(
            gfx.linalg.Euler(0.0005 * i, 0.001 * i)
        )
        cube.rotation.multiply(rot)

    rot, pos = controls.get_view()
    camera.rotation.copy(rot)
    camera.position.copy(pos)

    renderer.render(scene, camera)
    canvas.request_draw()


if __name__ == "__main__":
    canvas.request_draw(animate)
    app.exec_()
