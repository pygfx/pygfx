"""
Example showing how two scenes can be visually merged.

The idea is simply to sequentially render the scenes into the same
framebuffer. The result should be the same as when all world-objects
where in one scene rendered at once. Eventually this should hold true
even for semitransparent objects.

Note the similarity with the overlay example, except that we don't clear
the depth in between the two render calls here.

All this should work even with different camera's, but then things get
complicated with the depth buffer values being between 0 and 1.
"""

import pygfx as gfx

from PySide6 import QtWidgets
from wgpu.gui.qt import WgpuCanvas


# Create a canvas and renderer

app = QtWidgets.QApplication([])
canvas = WgpuCanvas(size=(500, 300))
renderer = gfx.renderers.WgpuRenderer(canvas)

# Compose a 3D scene

scene1 = gfx.Scene()

geometry1 = gfx.BoxGeometry(200, 200, 200)
material1 = gfx.MeshPhongMaterial(color=(1, 1, 0, 1.0))
cube1 = gfx.Mesh(geometry1, material1)
cube1.position.set(-50, 0, 0)
scene1.add(cube1)

camera = gfx.OrthographicCamera(400, 400)

# Compose another scene

scene2 = gfx.Scene()

material2 = gfx.MeshPhongMaterial(color=(1, 0, 1, 1.0))
cube2 = gfx.Mesh(geometry1, material2)
cube2.position.set(+50, 0, 0)
scene2.add(cube2)


def animate():
    rot = gfx.linalg.Quaternion().set_from_euler(gfx.linalg.Euler(0.005, 0.01))
    cube1.rotation.multiply(rot)
    cube2.rotation.multiply(rot)

    renderer.render(scene1, camera, flush=False)
    renderer.render(scene2, camera)

    canvas.request_draw()


if __name__ == "__main__":
    canvas.request_draw(animate)
    app.exec_()
