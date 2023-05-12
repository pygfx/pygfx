# -*- coding: utf-8 -*-
"""
Created on Fri May  5 09:17:14 2023

@author: s.Shaji
"""
from PyQt6  import uic
from PyQt6.QtWidgets import (QMainWindow,
                             QTableWidgetItem,
                             QApplication)

from wgpu.gui.qt import WgpuWidget
import wgpu.backends.rs  # noqa: F401, Select Rust backend
import numpy as np
import pygfx as gfx
# from MeshParser import parser

ui = '..\data\viewer.ui'

def parser(path=None):
    class mesh:
        nd = np.fromfile('..\data\nodes',dtype=np.float64).reshape(-1,4)
        e3t = np.fromfile('..\data\e3t',dtype=np.uint32).reshape(-1,5)
        e4q = np.fromfile('..\data\e4q',dtype=np.uint32).reshape(-1,6)
    return mesh

class Main(QMainWindow):
    def __init__(self,meshPath = None):
        super().__init__(None)
        uic.loadUi(ui,self)
        self.statusBar = self.statusBar()

        # Create canvas, renderer and a scene object
        self._canvas = WgpuWidget(parent=self)
        self._renderer = gfx.WgpuRenderer(self._canvas)
        self._scene = gfx.Scene()
        self._scene.add(gfx.PointLight())
        # scene.add(gfx.DirectionalLight())

        dark_gray = np.array((169, 167, 168, 255)) / 255
        light_gray = np.array((100, 100, 100, 255)) / 255
        background = gfx.Background(None, gfx.BackgroundMaterial(light_gray, dark_gray))
        self._scene.add(background)
        self._camera = gfx.PerspectiveCamera(depth_range=(0.01,1000000))

        self.gfxCanvas.addWidget(self._canvas)

        self.loadMesh()

        # Hook up the animate callback
        self.camOrbitView()
        self.view_dir = [0,-1,0]
        self._camera.show_object(self._scene,view_dir=self.view_dir)

        self.planView.clicked.connect(self.camPlanView)
        self.orbitView.clicked.connect(self.camOrbitView)
        self.zfak.valueChanged.connect(self.scaleZ)
        self.wireframe.clicked.connect(self.paintWireframe)
        self.normal.clicked.connect(self.paintNormal)
        self.phong.clicked.connect(self.paintPhong)

    def camPlanView(self):
        self._controller = gfx.PanZoomController(camera=self._camera,register_events=self._renderer)
        self._controller.controls['mouse3']= ('pan', 'drag', (1.0, 1.0))
        self._camera.show_object(self._scene,view_dir=(0,-1,0))

    def camOrbitView(self):
        self._controller = gfx.OrbitController(camera=self._camera,register_events=self._renderer)
        self._controller.controls['mouse3']= ('pan', 'drag', (1.0, 1.0))
        self._canvas.request_draw(self.animate)

    def scaleZ(self):
        self._scene.local.scale_y = self.zfak.value()
        self._canvas.request_draw(self.animate)

    def loadMesh(self):
        self.meshObj =  parser()

        # self._scene.world.position = tuple(self.meshObj.nd[:,1:].mean(axis=0))
        self.verts = (self.meshObj.nd[:,1:]-self.meshObj.nd[:,1:].mean(axis=0)).astype('f')
        self.verts = np.ascontiguousarray(self.verts[:,[0,2,1]],dtype='f')
        self.verts[:,2] *= -1
        self.paintWireframe()

    def paintNormal(self):
        self.statusBar.showMessage('Making Triangles...')
        if hasattr(self,'mesh'):
            self._scene.remove(self.mesh)
        self.mesh = gfx.Mesh(
            gfx.Geometry(positions=self.verts, indices=self.meshObj.e3t[:,1:-1]-1),
            gfx.MeshNormalMaterial(clipping_mode='all'),
        )
        self._scene.add(self.mesh)

        if hasattr(self,'patches'):
            self._scene.remove(self.patches)
        self.statusBar.showMessage('Making patches...')
        self.patches = gfx.Mesh(
            gfx.Geometry(indices=  self.meshObj.e4q[:,1:-1]-1, positions=self.verts),
            gfx.MeshNormalMaterial(color='blue',clipping_mode='all')
            )
        self._scene.add(self.patches)
        self.patches.add_event_handler(self.pick_id,"pointer_down")

        self.statusBar.showMessage('Ready')
    
    def paintPhong(self):
        self.statusBar.showMessage('Making Triangles...')
        if hasattr(self,'mesh'):
            self._scene.remove(self.mesh)
        self.mesh = gfx.Mesh(
            gfx.Geometry(positions=self.verts, indices=self.meshObj.e3t[:,1:-1]-1),
            gfx.MeshPhongMaterial(clipping_mode='all'),
        )
        self._scene.add(self.mesh)

        if hasattr(self,'patches'):
            self._scene.remove(self.patches)
        self.statusBar.showMessage('Making patches...')
        self.patches = gfx.Mesh(
            gfx.Geometry(indices=  self.meshObj.e4q[:,1:-1]-1, positions=self.verts),
            gfx.MeshPhongMaterial(color='blue',clipping_mode='all')
            )
        self._scene.add(self.patches)
        self.patches.add_event_handler(self.pick_id,"pointer_down")

        self.statusBar.showMessage('Ready')

    def paintWireframe(self):
        self.statusBar.showMessage('Making Triangles...')
        if hasattr(self,'mesh'):
            self._scene.remove(self.mesh)
        self.mesh = gfx.Mesh(
            gfx.Geometry(positions=self.verts, indices=self.meshObj.e3t[:,1:-1]-1),
            gfx.MeshBasicMaterial(wireframe=True,clipping_mode='all'),
        )
        self._scene.add(self.mesh)

        if hasattr(self,'patches'):
            self._scene.remove(self.patches)
        self.statusBar.showMessage('Making patches...')
        self.patches = gfx.Mesh(
            gfx.Geometry(indices=  self.meshObj.e4q[:,1:-1]-1, positions=self.verts),
            gfx.MeshBasicMaterial(color='blue',wireframe=True,clipping_mode='all')
            )
        self._scene.add(self.patches)
        self.patches.add_event_handler(self.pick_id,"pointer_down")

        self.statusBar.showMessage('Ready')

    def pick_id(self,event):
        self.pickid = event.pick_info
        eid = self.meshObj.e4q[event.pick_info["face_index"]][0]
        self.statusBar.showMessage(f'Element ID: {eid}')
        # print(self.pick_id,event)

    def animate(self):
        self._renderer.render(self._scene, self._camera)
        self._canvas.request_draw()

if __name__ == "__main__":
    app = QApplication([])
    m = Main()
    m.show()
    app.exec()
