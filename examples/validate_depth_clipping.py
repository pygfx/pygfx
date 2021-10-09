"""
Example (and test) for camera depth clipping planes. This draws four
rectangles near the near and far clipping planes.

* Only the two green ones should be visible.
* The greener square should be in the upper-left.
"""

import pygfx as gfx

from PySide6 import QtWidgets
from wgpu.gui.qt import WgpuCanvas


app = QtWidgets.QApplication([])

canvas = WgpuCanvas()
renderer = gfx.renderers.WgpuRenderer(canvas)
scene = gfx.Scene()


# %% Select near, far, and camera

# With the ortho camera, you can pick any values you like, also negative
# near values. With the perspective camera you'll want to pick a small
# positive near-value, and a relatively small value for the far-value
# as well, otherwise the distant squares become smaller than 1 pixel ;)

# Select camera and matchingclipping planes
if True:
    near, far = -40, 300
    camera = gfx.OrthographicCamera(2.2, 2.2, near, far)
else:
    near, far = 5, 10
    camera = gfx.PerspectiveCamera(50, 1, near, far)


# %% Create four planes near the z-clipping planes

geometry = gfx.PlaneGeometry(1, 1)
green_material = gfx.MeshBasicMaterial(color=(0, 0.8, 0.2, 1))
greener_material = gfx.MeshBasicMaterial(color=(0, 1, 0, 1))
red_material = gfx.MeshBasicMaterial(color=(1, 0, 0, 1))

plane1 = gfx.Mesh(geometry, green_material)
plane2 = gfx.Mesh(geometry, red_material)
plane3 = gfx.Mesh(geometry, greener_material)
plane4 = gfx.Mesh(geometry, red_material)

# Note the negation of near and far in the plane's position. This is
# because the camera looks down the z-axis: more negative means moving
# away from the camera, positive values are behind the camera.
plane1.position = gfx.linalg.Vector3(-0.51, -0.51, -(near + 0.01))  # in range
plane2.position = gfx.linalg.Vector3(+0.51, -0.51, -(near - 0.01))  # out range
plane3.position = gfx.linalg.Vector3(-0.51, +0.51, -(far - 0.01))  # in range
plane4.position = gfx.linalg.Vector3(+0.51, +0.51, -(far + 0.01))  # out range

for plane in (plane1, plane2, plane3, plane4):
    scene.add(plane)


if __name__ == "__main__":
    print(__doc__)
    canvas.request_draw(lambda: renderer.render(scene, camera))
    app.exec()
