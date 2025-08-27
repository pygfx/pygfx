"""
Simple Cube with Qt
===================

Example showing a single geometric cube.

Alternatively one can import the Qt library of your choice, and then use
``from rendercanvas.qt import RenderCanvas`` to get the corresponding canvas class.
"""

# sphinx_gallery_pygfx_docs = 'code'
# sphinx_gallery_pygfx_test = 'off'

import pygfx as gfx
import pylinalg as la

# Select one
# from rendercanvas.pyqt5 import RenderCanvas
# from rendercanvas.pyqt6 import RenderCanvas
from rendercanvas.pyside6 import RenderCanvas, loop


canvas = RenderCanvas()
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
    loop.run()  # i.e. app.exec()
