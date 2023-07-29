# -*- coding: utf-8 -*-
"""
Created on Fri May  5 09:17:14 2023

@author: s.Shaji
"""
import numpy as np
from PyQt6.QtWidgets import QApplication, QMainWindow, QGridLayout, QWidget, QPushButton
from PyQt6.QtGui import QColor
from wgpu.gui.qt import WgpuWidget
import pygfx as gfx


def generateSampleQuads(cols=10):
    pos = np.dstack(np.meshgrid(np.arange(cols), np.arange(2))).reshape(-1, 2)
    z = np.abs([*np.arange(-cols/2, cols/2), *np.arange(-cols/2, cols/2)])
    pos = np.c_[pos, z].astype('f')
    n1 = np.arange(cols)
    n2 = np.full(cols-1, 0)
    n3 = np.full(cols-1, 1)
    idx = np.dstack((n1[:-1], n2, n1[:-1], n3, n1[1:],
                    n3, n1[1:], n2)).reshape(-1, 2)
    indices = (idx[:, 0] + idx[:, 1]*cols).reshape(-1, 4)
    return pos, indices


class Main(QMainWindow):
    def __init__(self):
        super().__init__(None)
        self.resize(640, 480)

        self.statusBar = self.statusBar()
        main_widget = QWidget(self)
        self.setCentralWidget(main_widget)

        # Create a QGridLayout for the main_widget
        main_layout = QGridLayout(main_widget)
        self.setLayout(main_layout)

        for n, i in enumerate(["wireframe", "vertex_color", "face_color"]):
            setattr(self, i, QPushButton(i))
            btn = getattr(self, i)
            main_layout.addWidget(btn, 0, n, 1, 1)
            action = getattr(self, f"make_{i}")
            btn.clicked.connect(action)

        # Create canvas, renderer and a scene object
        self._canvas = WgpuWidget(parent=main_widget)
        self._renderer = gfx.WgpuRenderer(self._canvas)
        self._scene = gfx.Scene()
        self._scene.add(gfx.PointLight())

        clr = QColor.fromRgb(87, 187, 200)

        # Create the contrasting background using the fully saturated red color
        background = gfx.Background(
            None, gfx.BackgroundMaterial(clr.getRgbF(), clr.getRgbF()))

        self._scene.add(background)
        self._camera = gfx.PerspectiveCamera(depth_range=(0.01, 1000000))

        self._controller = gfx.OrbitController(
            camera=self._camera, register_events=self._renderer)
        self._controller.controls['mouse3'] = ('pan', 'drag', (1.0, 1.0))

        # Hook up the animate callback
        self._canvas.request_draw(self.animate)

        main_layout.addWidget(self._canvas, 1, 0, 5, 3)

        self.make_wireframe()

        self._camera.show_object(
            self._scene, view_dir=(0, 0, -1), up=(0, 0, 1))

    def make_wireframe(self):
        if hasattr(self, 'patches'):
            self.patches.material.wireframe = True
            self.patches.material.vertex_colors = False
            self.animate()
            return

        self.statusBar.showMessage('Making patches...')
        pos, indices = generateSampleQuads()
        colors = np.repeat(pos[:, -1]/pos[:, -1].max(), 4).reshape(-1, 4)
        colors[:, -1] = 1
        self.patches = gfx.Mesh(
            gfx.Geometry(indices=indices,
                         positions=pos,
                         colors=colors,
                         texcoords=np.arange(len(indices))),
            gfx.MeshBasicMaterial(wireframe=True)
        )
        self._scene.add(self.patches)
        self.patches.add_event_handler(self.pick_id, "click")
        self.statusBar.showMessage('Ready')

    def make_vertex_color(self):

        self.patches.material.wireframe = False
        self.patches.material.vertex_colors = True
        self.animate()

    def make_face_color(self):
        self.patches.material.vertex_colors = False
        self.patches.material.face_colors = True

    def pick_id(self, event):
        self.pickid = event.pick_info
        print(self.pickid)

    def animate(self):
        self._renderer.render(self._scene, self._camera)
        self._canvas.request_draw()


if __name__ == "__main__":
    app = QApplication([])
    m = Main()
    m.show()
    app.exec()
