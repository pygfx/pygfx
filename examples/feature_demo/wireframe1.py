"""
Wireframe 1
===========

Example showing a Torus knot, with a wireframe overlay.

In this case the wireframe is lit while the solid mesh is not,
producing a look of a metallic frame around a soft tube.

"""

# sphinx_gallery_pygfx_docs = 'screenshot'
# sphinx_gallery_pygfx_test = 'run'

from wgpu.gui.auto import WgpuCanvas, run
import pygfx as gfx
import pylinalg as la


canvas = WgpuCanvas()
renderer = gfx.renderers.WgpuRenderer(canvas)
scene = gfx.Scene()

geometry = gfx.torus_knot_geometry(1, 0.3, 64, 16)

material1 = gfx.MeshBasicMaterial(color=(0.7, 0, 0, 1))
obj1 = gfx.Mesh(geometry, material1)
scene.add(obj1)

material2 = gfx.MeshPhongMaterial(
    color=(0.7, 0.7, 0.8, 1), wireframe=True, wireframe_thickness=1.5
)
obj2 = gfx.Mesh(geometry, material2)
scene.add(obj2)

camera = gfx.PerspectiveCamera(70, 1)
camera.local.z = 4

scene.add(gfx.AmbientLight(), camera.add(gfx.DirectionalLight()))


def animate():
    rot = la.quat_from_euler((0.0071, 0.01), order="XY")
    obj1.local.rotation = la.quat_mul(rot, obj1.local.rotation)
    obj2.local.rotation = la.quat_mul(rot, obj2.local.rotation)

    renderer.render(scene, camera)
    canvas.request_draw()


if __name__ == "__main__":
    canvas.request_draw(animate)
    run()
