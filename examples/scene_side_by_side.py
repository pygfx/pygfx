"""
Example showing how to render to a subregion of a canvas.

This is a feature necessary to implement e.g. subplots.
"""

import numpy as np
import pygfx as gfx

from PyQt5 import QtWidgets
from wgpu.gui.qt import WgpuCanvas


# Create a anvas and a renderer

app = QtWidgets.QApplication([])
canvas = WgpuCanvas(size=(500, 300))
renderer = gfx.renderers.WgpuRenderer(canvas)

# Compose a 3D scene

scene1 = gfx.Scene()

geometry1 = gfx.BoxGeometry(200, 200, 200)
material1 = gfx.MeshPhongMaterial(color=(1, 1, 0, 1.0))
cube1 = gfx.Mesh(geometry1, material1)
scene1.add(cube1)

camera1 = gfx.PerspectiveCamera(70, 16 / 9)
camera1.position.z = 400

# Compose another scene

scene2 = gfx.Scene()

positions = np.array(
    [[-1, -1, 0], [-1, +1, 0], [+1, +1, 0], [+1, -1, 0], [-1, -1, 0], [+1, +1, 0]],
    np.float32,
)
geometry2 = gfx.Geometry(positions=positions)
material2 = gfx.LineMaterial(thickness=5.0, color=(0.8, 0.0, 0.2, 1.0))
line2 = gfx.Line(geometry2, material2)
scene2.add(line2)

camera2 = gfx.OrthographicCamera(2.2, 2.2)


def animate():
    rot = gfx.linalg.Quaternion().set_from_euler(gfx.linalg.Euler(0.005, 0.01))
    cube1.rotation.multiply(rot)

    w, h = canvas.get_logical_size()
    renderer.render(scene1, camera1, flush=False, region=(0, 0, w / 2, h))
    renderer.render(scene2, camera2, flush=False, region=(w / 2, 0, w / 2, h))
    renderer.flush()

    canvas.request_draw()


if __name__ == "__main__":
    canvas.request_draw(animate)
    app.exec_()
