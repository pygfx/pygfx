"""
Example showing transparency using three orthogonal planes.
Press space to toggle the order of the planes.
Press 1,2,3 to select the blend mode.
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
        drag_start(
            (event.position().x(), event.position().y()),
            self.get_logical_size(),
            camera,
        )
        app.setOverrideCursor(QtCore.Qt.ClosedHandCursor)

    def keyPressEvent(self, event):  # noqa: N802
        if not event.text():
            pass
        elif event.text() == " ":
            print("Rotating scene element order")
            scene.add(scene.children[0])
        elif event.text() in "0123456789":
            m = [
                None,
                "opaque",
                "simple1",
                "simple2",
                "weighted",
                "weighted_depth",
                "weighted_plus",
            ]
            mode = m[int(event.text())]
            renderer.blend_mode = mode
            print("Selecting blend_mode", mode)

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
            drag_move((event.position().x(), event.position().y()))


app = QtWidgets.QApplication([])

canvas = WgpuCanvasWithInputEvents()
canvas._target_fps = 1000
renderer = gfx.renderers.WgpuRenderer(canvas, show_fps=True)
scene = gfx.Scene()

sphere = gfx.Mesh(
    gfx.sphere_geometry(10),
    gfx.MeshPhongMaterial(),
    render_pass="opaque",
)

geometry = gfx.plane_geometry(50, 50)
plane1 = gfx.Mesh(
    geometry,
    gfx.MeshBasicMaterial(color=(1, 0, 0, 0.3)),
    render_pass="transparent",
)
plane2 = gfx.Mesh(
    geometry,
    gfx.MeshBasicMaterial(color=(0, 1, 0, 0.5)),
    render_pass="transparent",
)
plane3 = gfx.Mesh(
    geometry,
    gfx.MeshBasicMaterial(color=(0, 0, 1, 0.7)),
    render_pass="transparent",
)

plane1.rotation.set_from_axis_angle(gfx.linalg.Vector3(1, 0, 0), 1.571)
plane2.rotation.set_from_axis_angle(gfx.linalg.Vector3(0, 1, 0), 1.571)
plane3.rotation.set_from_axis_angle(gfx.linalg.Vector3(0, 0, 1), 1.571)

scene.add(plane1, plane2, plane3, sphere)

camera = gfx.PerspectiveCamera(70, 16 / 9)
camera.position.z = 70
controls = gfx.OrbitControls(camera.position.clone())


def animate():
    controls.update_camera(camera)
    renderer.render(scene, camera)
    canvas.request_draw()


if __name__ == "__main__":
    print(__doc__)
    canvas.request_draw(animate)
    app.exec()
