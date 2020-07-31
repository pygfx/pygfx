import numpy as np
import pygfx as gfx

from PyQt5 import QtWidgets, QtCore, QtGui
from wgpu.gui.qt import WgpuCanvas


class Overlay(QtWidgets.QWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args, **kwargs,
        )
        # We want a tranlucent background, no window frame, and always on top
        # of the parent widget.
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground, True)
        self.setWindowFlags(QtCore.Qt.FramelessWindowHint | QtCore.Qt.Tool)
        # No background
        self.setAutoFillBackground(False)
        # self.setAttribute(QtCore.Qt.WA_PaintOnScreen, True)

    def paintEvent(self, event):

        painter = QtGui.QPainter()
        if not painter.begin(self):
            return

        features = None
        for label in self.parent().text_labels:
            if features != (label.size, label.color):
                features = label.size, label.color
                painter.setFont(QtGui.QFont("Arial", label.size))
                painter.setPen(QtGui.QColor(label.color))
            x, y = label.ppos
            painter.drawText(QtCore.QPointF(x, y), label.text)

        painter.end()


class TextCanvas(WgpuCanvas):

    overlay = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.text_labels = []

        self.overlay = Overlay(self)
        self.overlay.setGeometry(self.geometry())
        self.overlay.show()
        self.overlay.raise_()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.overlay:
            self.overlay.setGeometry(self.geometry())

    def moveEvent(self, event):
        super().moveEvent(event)
        if self.overlay:
            self.overlay.setGeometry(self.geometry())


import wgpu
from pygfx.renderers import Renderer


class TextLabel(gfx.WorldObject):
    def __init__(self, text, size=12, color="f000000", position=(0, 0, 0)):
        super().__init__()
        self.text = str(text)
        self.size = int(size)
        self.color = color
        self.position = gfx.linalg.Vector3(*position)


class QtTextRenderer(Renderer):
    def __init__(self, canvas):
        self._canvas = canvas

        # todo: allow passing a normal qt canvas?
        # assert isinstance(canvas, wgpu.gui.qt.QtWgpuCanvas)
        assert isinstance(canvas, TextCanvas)

    def render(self, scene: gfx.WorldObject, camera: gfx.Camera):
        """ Main render method, called from the canvas.
        """

        # Ensure that matrices are up-to-date
        # Or not: assume that this is done by the wgpu renderer
        if False:
            scene.update_matrix_world()
            camera.set_viewport_size(*logical_size)
            camera.update_matrix_world()  # camera may not be a member of the scene
            camera.update_projection_matrix()

        # Get the sorted list of objects to render
        def visit(wobject):
            if wobject.visible and isinstance(wobject, TextLabel):
                q.append(wobject)

        q = []
        scene.traverse(visit)

        gfx.linalg.Vector4()
        m = gfx.linalg.Matrix4()
        mul = m.multiply_matrices
        # proj = mul(camera.projection_matrix, camera.matrix_world_inverse)
        logical_size = self._canvas.get_logical_size()

        self._canvas.text_labels = q
        self._canvas.overlay.update()
        for wobject in q:
            # m = gfx.linalg.Matrix4().multiply_matrices(proj, wobject.matrix_world)
            pos = gfx.linalg.Vector4(
                wobject.position.x, wobject.position.y, wobject.position.z, 1
            )
            pos = pos.apply_matrix4(wobject.matrix_world)
            pos = pos.apply_matrix4(camera.matrix_world_inverse)
            pos = pos.apply_matrix4(camera.projection_matrix)

            pos_pix = (
                (pos.x + 1) * logical_size[0] / 2,
                (pos.y + 1) * logical_size[1] / 2,
            )
            wobject.ppos = pos_pix
            # print(pos_pix)


app = QtWidgets.QApplication([])

canvas = TextCanvas()
renderer = gfx.WgpuRenderer(canvas)
text_renderer = QtTextRenderer(canvas)

scene = gfx.Scene()

positions = np.random.normal(0, 0.5, (20, 2)).astype(np.float32)
geometry = gfx.Geometry(positions=positions)

material = gfx.PointsMaterial(size=10, color=(0, 1, 0.5, 0.7))
points = gfx.Points(geometry, material)
scene.add(points)
for point in positions:
    scene.add(TextLabel(f"{point}", position=point, color="#ffffff"))

scene.add(gfx.Background(gfx.BackgroundMaterial((0.04, 0.0, 0, 1), (0, 0.0, 0.04, 1))))

camera = gfx.NDCCamera()


def aninmate():
    renderer.render(scene, camera)
    text_renderer.render(scene, camera)


if __name__ == "__main__":
    canvas.request_draw(aninmate)
    app.exec_()
