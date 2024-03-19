"""
Wireframe 2
===========

Example showing a Torus knot, as a wireframe. We create two wireframes,
one for the front, bright blue and lit, and one for the back, unlit and
gray.
"""

# sphinx_gallery_pygfx_docs = 'screenshot'
# sphinx_gallery_pygfx_test = 'run'

from wgpu.gui.auto import WgpuCanvas, run
import pygfx as gfx
import pylinalg as la


canvas = WgpuCanvas()
renderer = gfx.renderers.WgpuRenderer(canvas)
scene = gfx.Scene()

geometry = gfx.torus_knot_geometry(1, 0.3, 64, 8)

material1 = gfx.MeshBasicMaterial(
    color=(0.2, 0.2, 0.2, 1.0), wireframe=True, wireframe_thickness=3, side="back"
)
obj1 = gfx.Mesh(geometry, material1)
scene.add(obj1)

material2 = gfx.MeshPhongMaterial(
    color=(0, 0.8, 0.8, 1), wireframe=True, wireframe_thickness=3, side="front"
)
obj2 = gfx.Mesh(geometry, material2)
scene.add(obj2)

camera = gfx.PerspectiveCamera(70, 1)
camera.local.z = 4

scene.add(gfx.AmbientLight(0.2), camera.add(gfx.DirectionalLight(0.8)))


def animate():
    rot = la.quat_from_euler((0.0071, 0.01), order="XY")
    obj1.local.rotation = la.quat_mul(rot, obj1.local.rotation)
    obj2.local.rotation = la.quat_mul(rot, obj2.local.rotation)

    renderer.render(scene, camera)
    canvas.request_draw()


if __name__ == "__main__":
    canvas.request_draw(animate)
    run()
