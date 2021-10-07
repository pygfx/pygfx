import numpy as np
import pygfx as gfx

from PySide6 import QtWidgets, QtCore, QtGui
from wgpu.gui.qt import WgpuCanvas


# %% Qt logic


class CanvasWithOverlay(WgpuCanvas):
    """A Qt canvas with support for 2D overlay."""

    overlay = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._text_labels = []

        self.overlay = Overlay(self)
        self.overlay.setGeometry(self.geometry())
        self.overlay.show()
        self.overlay.raise_()

    def resizeEvent(self, event):  # noqa: N802
        super().resizeEvent(event)
        if self.overlay:
            self.overlay.setGeometry(self.geometry())

    def moveEvent(self, event):  # noqa: N802
        super().moveEvent(event)
        if self.overlay:
            self.overlay.setGeometry(self.geometry())

    def set_text_labels(self, wobjects):
        """Set text labels to overlay. Must be a list of TextOverlay objects."""
        self._text_labels = wobjects


class Overlay(QtWidgets.QWidget):
    """Overlay that draws 2D featues using the canvas API.

    We cannot draw in the wgpu widget directly, because that widget has
    no paint engine (we have to remove it to prevent Qt from overwriting
    the screen that we draw with wgpu). We can also not use a normal
    widget overlaid over it, because Qt would then do the compositing,
    but the wgpu widget draws directly to screen, so the overlay widget
    would get a black background. Therefore, the overlay widget is a
    toplevel widget (a window) that we keep exactly on top of the actual
    widget. Not pretty, but it seems to work. I am not sure how well
    this holds up on other platforms.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            **kwargs,
        )
        # We want a tranlucent background, and no window frame.
        # Setting the Tool flag make it always on top of the parent widget.
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground, True)
        self.setWindowFlags(QtCore.Qt.FramelessWindowHint | QtCore.Qt.Tool)
        # No background, just in case.
        self.setAutoFillBackground(False)

    def paintEvent(self, event):  # noqa: N802

        painter = QtGui.QPainter()
        if not painter.begin(self):
            return

        # Draw text labels
        text_labels = self.parent()._text_labels
        features = None
        for label in text_labels:
            if features != (label.size, label.color):
                features = label.size, label.color
                painter.setFont(QtGui.QFont("Arial", label.size))
                painter.setPen(QtGui.QColor(label.color))
            painter.drawText(QtCore.QPointF(*label.ppos), label.text)

        painter.end()


# %% wgpu world object and renderer

from pygfx.renderers import Renderer  # noqa: E402


class TextOverlay(gfx.WorldObject):
    """A text label that can get overlaid on the visualization using e.q. Qt."""

    def __init__(self, text, size=12, color="f000000", position=(0, 0, 0)):
        super().__init__()
        self.text = str(text)
        self.size = int(size)
        self.color = color
        self.position = gfx.linalg.Vector3(*position)
        # Other options: font, bold, background, pixel-offset, alignment


class QtOverlayRenderer(Renderer):
    """A special renderer that can draw certain 2D overlays over a Qt canvas.
    Currently only text (TextOverlay objects).
    """

    def __init__(self, canvas):
        self._canvas = canvas

        # todo: allow passing a normal qt canvas?
        # assert isinstance(canvas, wgpu.gui.qt.QtWgpuCanvas)
        assert isinstance(canvas, CanvasWithOverlay)

    def render(self, scene: gfx.WorldObject, camera: gfx.Camera):
        """Main render method, called from the canvas."""

        logical_size = self._canvas.get_logical_size()

        # We assume that this call is preceded by a call to the wgpu renderer,
        # so we don't need to apply any updates.
        # scene.update_matrix_world()
        # camera.set_viewport_size(*logical_size)
        # camera.update_matrix_world()
        # camera.update_projection_matrix()

        # Get the list of objects to render
        def visit(wobject):
            if wobject.visible and isinstance(wobject, TextOverlay):
                q.append(wobject)

        q = []
        scene.traverse(visit)

        # Set the pixel position of each text overlay object
        for wobject in q:
            pos = wobject.position.clone()
            pos = pos.apply_matrix4(wobject.matrix_world).project(camera)
            # I don't understand this 0.25 value. I would expect it to be 0.5
            # but for some reason it needs to be 0.25.
            pos_pix = (
                (+pos.x * 0.25 + 0.5) * logical_size[0],
                (-pos.y * 0.25 + 0.5) * logical_size[1],
            )
            wobject.ppos = pos_pix

        # Store the labels for the overlay, and schedule a draw.
        self._canvas.set_text_labels(q)
        self._canvas.overlay.update()


app = QtWidgets.QApplication([])

canvas = CanvasWithOverlay()
renderer = gfx.WgpuRenderer(canvas)
overlay_renderer = QtOverlayRenderer(canvas)

scene = gfx.Scene()

positions = np.random.normal(0, 1, (20, 3)).astype(np.float32)
geometry = gfx.Geometry(positions=positions)

material = gfx.PointsMaterial(size=10, color=(0, 1, 0.5, 0.7))
points = gfx.Points(geometry, material)
scene.add(points)
for point in positions:
    scene.add(
        TextOverlay(
            f"{point[0]:0.1f}, {point[1]:0.1f}", position=point, color="#ffffff"
        )
    )

scene.add(gfx.Background(gfx.BackgroundMaterial((0.04, 0.0, 0, 1), (0, 0.0, 0.04, 1))))

camera = gfx.OrthographicCamera(3, 3)


def aninmate():
    renderer.render(scene, camera)
    overlay_renderer.render(scene, camera)


if __name__ == "__main__":
    canvas.request_draw(aninmate)
    app.exec_()
