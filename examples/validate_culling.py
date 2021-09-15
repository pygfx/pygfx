"""
Example test to validate winding and culling.

* The red knot should look normal and well lit.
* The green know should show the backfaces, well lit.
* The purple and cyan knots are not lit.

"""

import pygfx as gfx

from PyQt5 import QtWidgets
from wgpu.gui.qt import WgpuCanvas


app = QtWidgets.QApplication([])

canvas = WgpuCanvas()
renderer = gfx.renderers.WgpuRenderer(canvas)
scene = gfx.Scene()

# geometry = gfx.BoxGeometry(1, 1, 1)
geometry = gfx.TorusKnotGeometry(1, 0.3, 64, 10)

# Top left
material1 = gfx.MeshPhongMaterial(color=(1, 0, 0, 1))
obj1 = gfx.Mesh(geometry, material1)
obj1.position.set(-2, +2, 0)
obj1.material.winding = "CCW"
obj1.material.side = "FRONT"

# Top right
material2 = gfx.MeshPhongMaterial(color=(0, 1, 0, 1))
obj2 = gfx.Mesh(geometry, material2)
obj2.position.set(+2, +2, 0)
obj2.material.winding = "CCW"
obj2.material.side = "BACK"

# Bottom left
material3 = gfx.MeshPhongMaterial(color=(1, 0, 1, 1))
obj3 = gfx.Mesh(geometry, material3)
obj3.position.set(-2, -2, 0)
obj3.material.winding = "CW"
obj3.material.side = "FRONT"

# Bottom right
material4 = gfx.MeshPhongMaterial(color=(0, 1, 1, 1))
obj4 = gfx.Mesh(geometry, material4)
obj4.position.set(+2, -2, 0)
obj4.material.winding = "CW"
obj4.material.side = "BACK"

# Rotate all of them and add to scene
rot = gfx.linalg.Quaternion().set_from_euler(gfx.linalg.Euler(0.71, 1))
for obj in obj1, obj2, obj3, obj4:
    obj.rotation.multiply(rot)
    scene.add(obj)

camera = gfx.OrthographicCamera(6, 8)
camera.scale.z *= -1

if __name__ == "__main__":
    print(__doc__)
    canvas.request_draw(lambda: renderer.render(scene, camera))
    # canvas.request_draw(animate)
    app.exec_()
