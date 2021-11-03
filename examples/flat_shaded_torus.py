"""
Example showing a Torus knot, using flat shading.
"""

import pygfx as gfx

from PySide6 import QtWidgets
from wgpu.gui.qt import WgpuCanvas


app = QtWidgets.QApplication([])

canvas = WgpuCanvas()
renderer = gfx.renderers.WgpuRenderer(canvas)
scene = gfx.Scene()


geometry = gfx.torus_knot_geometry(1, 0.3, 64, 10)
geometry.texcoords.data[:, 0] *= 10  # stretch the texture
material = gfx.MeshFlatMaterial(color=(1, 0, 1, 1))
obj = gfx.Mesh(geometry, material)
scene.add(obj)


camera = gfx.PerspectiveCamera(70, 1)
camera.position.z = 4


def animate():
    rot = gfx.linalg.Quaternion().set_from_euler(gfx.linalg.Euler(0.0071, 0.01))
    obj.rotation.multiply(rot)

    renderer.render(scene, camera)
    canvas.request_draw()


if __name__ == "__main__":
    canvas.request_draw(animate)
    app.exec()
