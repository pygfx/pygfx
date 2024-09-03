"""
Integrate Pygfx in Qt
=====================
"""

# sphinx_gallery_pygfx_docs = 'code'
# sphinx_gallery_pygfx_test = 'off'

import random

from PySide6 import QtWidgets
from wgpu.gui.qt import WgpuCanvas

import pygfx as gfx


class Main(QtWidgets.QWidget):
    def __init__(self):
        super().__init__(None)
        self.resize(640, 480)

        # Creat button and hook it up
        self._button = QtWidgets.QPushButton("Add a line", self)
        self._button.clicked.connect(self._on_button_click)

        # Create canvas, renderer and a scene object
        self._canvas = WgpuCanvas(parent=self)
        self._renderer = gfx.WgpuRenderer(self._canvas)
        self._scene = gfx.Scene()
        self._camera = gfx.OrthographicCamera(110, 110)

        # Hook up the animate callback
        self._canvas.request_draw(self.animate)

        layout = QtWidgets.QHBoxLayout()
        self.setLayout(layout)
        layout.addWidget(self._button)
        layout.addWidget(self._canvas)

    def _on_button_click(self):
        positions = [
            [random.uniform(-50, 50), random.uniform(-50, 50), 0] for i in range(8)
        ]
        line = gfx.Line(
            gfx.Geometry(positions=positions), gfx.LineMaterial(thickness=3)
        )
        self._scene.add(line)
        self._canvas.update()

    def animate(self):
        self._renderer.render(self._scene, self._camera)


app = QtWidgets.QApplication([])
m = Main()
m.show()


if __name__ == "__main__":
    app.exec()
