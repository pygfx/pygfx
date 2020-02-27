from PyQt5 import QtWidgets
from wgpu.gui.qt import WgpuCanvas

import visvis2 as vv


class Main(QtWidgets.QWidget):
    def __init__(self):
        super().__init__(None)
        self.resize(640, 480)

        # Creat button and hook it up
        self._button = QtWidgets.QPushButton("add triangle", self)
        self._button.clicked.connect(self._on_button_click)

        # Create canvas, renderer and a scene object
        self._canvas = WgpuCanvas(parent=self)
        self._renderer = vv.WgpuRenderer(self._canvas)
        self._scene = vv.Scene()
        self._camera = vv.PerspectiveCamera(45, 16 / 9, 0.1, 1000)

        # Hook up the animate callback
        self._canvas.draw_frame = self.animate

        layout = QtWidgets.QHBoxLayout()
        self.setLayout(layout)
        layout.addWidget(self._button)
        layout.addWidget(self._canvas)

    def _on_button_click(self):
        t = vv.Mesh(vv.Geometry(), vv.TriangleMaterial())
        self._scene.add(t)
        self._canvas.update()

    def animate(self):
        width, height, ratio = self._canvas.get_size_and_pixel_ratio()
        self._camera.aspect = width / height
        self._renderer.render(self._scene, self._camera)


app = QtWidgets.QApplication([])
m = Main()
m.show()


if __name__ == "__main__":
    app.exec_()
