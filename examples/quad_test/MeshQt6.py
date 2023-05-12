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

ui = 'viewer.ui'

def parser(path=None):
    class mesh:
        nd = np.fromfile('nodes',dtype=np.float64).reshape(-1,4)
        e3t = np.fromfile('e3t',dtype=np.uint32).reshape(-1,5)
        e4q = np.fromfile('e4q',dtype=np.uint32).reshape(-1,6)
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

        # if meshPath is not None:
        #     self.meshPath = meshPath
        self.loadMesh()

        # Hook up the animate callback
        self.camOrbitView()
        self.view_dir = [0,-1,0]
        self._camera.show_object(self._scene,view_dir=self.view_dir)

        self.planView.clicked.connect(self.camPlanView)
        self.orbitView.clicked.connect(self.camOrbitView)
        self.zfak.valueChanged.connect(self.scaleZ)

    def camPlanView(self):
        self._controller = gfx.PanZoomController(camera=self._camera,register_events=self._renderer)
        self._controller.controls['mouse3']= ('pan', 'drag', (1.0, 1.0))
        # self._camera.show_pos(pos)#[0]-w/2, pos[0]+w/2, pos[1]+h/2, pos[1]-h/2)#,view_dir=(0,0,-1))
        self._camera.show_object(self._scene,view_dir=(0,-1,0))

    def camOrbitView(self):
        self._controller = gfx.OrbitController(camera=self._camera,register_events=self._renderer)
        self._controller.controls['mouse3']= ('pan', 'drag', (1.0, 1.0))
        self._canvas.request_draw(self.animate)

    def scaleZ(self):
        self._scene.local.scale_y = self.zfak.value()
        self._canvas.request_draw(self.animate)

    def loadMesh(self):
        # self.statusBar.showMessage(f'{self.meshPath} wird eingeladen...')
        self.meshObj =  parser()

        self.nsBox.setEnabled(False)
        # if self.meshObj.dimensions.nodestrings>0:
        #     self.nsBox.setEnabled(True)
        #     self.nsFilter.addItems(['None',*[str(t) for t in np.unique(self.meshObj.bcvals['TYP'])]])
        #     self.nodestrings.addItems([str(n) for n in self.meshObj.ns.keys()])
        #     self.nsFilter.currentIndexChanged.connect(self.updateNSFilter)
        #     self.nodestrings.currentIndexChanged.connect(self.updateNS)
        #     self.nsZoom.clicked.connect(self.zoomToNS)
        #     self.updateNS()

        # self._scene.world.position = tuple(self.meshObj.nd[:,1:].mean(axis=0))
        self.verts = (self.meshObj.nd[:,1:]-self.meshObj.nd[:,1:].mean(axis=0)).astype('f')
        self.verts = np.ascontiguousarray(self.verts[:,[0,2,1]],dtype='f')
        self.verts[:,2] *= -1
        # self.verts[:,1] *= self.zfak.value()

        # if self.meshObj.dimensions.triangles>0:
        self.statusBar.showMessage('Making Triangles...')
        mesh = gfx.Mesh(
            gfx.Geometry(positions=self.verts, indices=self.meshObj.e3t[:,1:-1]-1),
            gfx.MeshBasicMaterial(wireframe=True,clipping_mode='all'),
        )
        box_local = gfx.BoxHelper(thickness=2, color="green")
        box_local.set_transform_by_object(mesh, space="local")
        mesh.add(box_local)
        mesh.add(gfx.AxesHelper(size=20))
        self._scene.add(mesh)
        
        
        # if self.meshObj.dimensions.patches>0:
        self.statusBar.showMessage('Making patches...')
        self.patches = gfx.Mesh(
            gfx.Geometry(indices=  self.meshObj.e4q[:,1:-1]-1, positions=self.verts),
            gfx.MeshBasicMaterial(color='blue',wireframe=True,clipping_mode='all')
            )
        self._scene.add(self.patches)
        self.patches.add_event_handler(self.pick_id,"pointer_down")
        

        # if self.meshObj.dimensions.nodestrings>0:
        #     self.statusBar.showMessage('Adding Nodestrings...')
        #     self.nsObj = {}
        #     for ns,arr in self.meshObj.ns.items():
        #         self.nsObj[ns] = gfx.Line(
        #             gfx.Geometry(positions=self.verts[arr-1]),
        #             gfx.LineMaterial(color="red")
        #             )
        #         self._scene.add(self.nsObj[ns])

        # if self.meshObj.dimensions.bcvaln>0:
        #     self.statusBar.showMessage('Adding Nodal Boundary Conditions...')
        #     kuk = self.meshObj.bcvaln[(self.meshObj.bcvaln['CODE']==1).ravel()]
        #     coords = self.verts[kuk['ND'].ravel()-1]
        #     coords[:,1] = kuk['VAL'].ravel() - self.meshObj.nd[:,1:].mean(axis=0)[2]

        #     k = gfx.Points(
        #         gfx.Geometry(positions=coords),
        #         gfx.PointsMaterial(color="red",size=5)
        #         )
        #     self._scene.add(k)

        self.statusBar.showMessage('Ready')

    def updateNSFilter(self):
        f = self.nsFilter.currentText()
        if f != 'None':
            f = int(f)
            ns = np.unique(self.meshObj.bcvals[(self.meshObj.bcvals['TYP']==f).ravel()]['NS'])
        else:
            ns = np.unique(self.meshObj.bcvals['NS'])
        self.nodestrings.blockSignals(True)
        self.nodestrings.clear()
        self.nodestrings.addItems([str(n) for n in ns])
        self.nodestrings.blockSignals(False)

    def updateNS(self):
        ns = int(self.nodestrings.currentText())
        bc = self.meshObj.bcvals[(self.meshObj.bcvals['NS']==ns).ravel()]
        self.nsTable.clearContents()
        self.nsTable.setRowCount(len(bc))
        for i,(n,ns,t,c,v) in enumerate(bc):
           self.nsTable.setItem(i,0, QTableWidgetItem(str(c[0])))
           self.nsTable.setItem(i,1, QTableWidgetItem(str(v[0])))
        self.zoomToNS()

    def zoomToNS(self):
        if hasattr(self,'nsObj'):
            ns = int(self.nodestrings.currentText())
            self._camera.show_object(self.nsObj[ns])


    def pick_id(self,event):
        self.pickid = event.pick_info
        eid = self.meshObj.e4q[event.pick_info["face_index"]][0]
        self.statusBar.showMessage(f'Element ID: {eid}')
        # print(self.pick_id,event)

    def animate(self):
        # if hasattr(self,'state'):
        #     for k,i in self._camera.get_state().items():
        #         if k in self.state.keys():
        #             try:
        #                 if any(self.state[k] != i):
        #                     print(self._camera.get_state())
        #                     break
        #             except:
        #                 if self.state[k] != i:
        #                     print(self._camera.get_state())
        #                     break
        # self.state = self._camera.get_state()
        # print(self._camera.get_state()['rotation'])
        self._renderer.render(self._scene, self._camera)
        self._canvas.request_draw()

if __name__ == "__main__":
    app = QApplication([])
    m = Main()#meshPath=path)
    m.show()
    app.exec()