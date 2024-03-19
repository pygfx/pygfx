"""
Simple Cube with Qt
===================

Example showing a single geometric cube.
"""

# sphinx_gallery_pygfx_docs = 'code'
# sphinx_gallery_pygfx_test = 'off'

import pygfx as gfx

from PySide6 import QtWidgets  # Replace PySide6 with PyQt6, PyQt5 or PySide2
from wgpu.gui.qt import WgpuCanvas
import pylinalg as la


app = QtWidgets.QApplication([])

canvas = WgpuCanvas()
renderer = gfx.renderers.WgpuRenderer(canvas)
scene = gfx.Scene()

cube = gfx.Mesh(
    gfx.box_geometry(200, 200, 200),
    gfx.MeshPhongMaterial(color=(0.2, 0.4, 0.6, 1.0)),
)
scene.add(cube)

scene.add(gfx.AmbientLight())
directional_light = gfx.DirectionalLight()
directional_light.world.z = 1
scene.add(directional_light)

camera = gfx.PerspectiveCamera(70, 16 / 9)
camera.world.z = 400


def animate():
    rot = la.quaternion.quat_from_euler((0.005, 0.01, 0))
    cube.local.rotation = la.quat_mul(rot, cube.local.rotation)

    renderer.render(scene, camera)
    canvas.request_draw()


if __name__ == "__main__":
    canvas.request_draw(animate)
    app.exec()
