"""
Example showing a Torus knot, with a wireframe overlay.

In this case the wireframe is lit while the solid mesh is not,
producing a look of a metalic frame around a soft tube.

"""

from wgpu.gui.auto import WgpuCanvas, run
import pygfx as gfx


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
camera.position.z = 4


def animate():
    rot = gfx.linalg.Quaternion().set_from_euler(gfx.linalg.Euler(0.0071, 0.01))
    obj1.rotation.multiply(rot)
    obj2.rotation.multiply(rot)

    renderer.render(scene, camera)
    canvas.request_draw()


if __name__ == "__main__":
    canvas.request_draw(animate)
    run()
