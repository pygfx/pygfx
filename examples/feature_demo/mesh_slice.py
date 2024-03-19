"""
Mesh Slice Material
===================

Example showing off the mesh slice material.
"""

# sphinx_gallery_pygfx_docs = 'screenshot'
# sphinx_gallery_pygfx_test = 'run'

from wgpu.gui.auto import WgpuCanvas, run
import pygfx as gfx


canvas = WgpuCanvas()
renderer = gfx.renderers.WgpuRenderer(canvas)
scene = gfx.Scene()

geometry = gfx.torus_knot_geometry(1, 0.3, 128, 16)
material1 = gfx.MeshPhongMaterial(color=(0.5, 0.5, 1.0, 0.5))
material2 = gfx.MeshSliceMaterial(thickness=8, color=(1, 1, 0, 1), plane=(0, 0, 1, 0))
obj1 = gfx.Mesh(geometry, material1)
obj2 = gfx.Mesh(geometry, material2)
scene.add(obj1)
obj2.cast_shadow = True  # It's actually the mesh slice casting a shadow
scene.add(obj2)

container = gfx.Mesh(
    gfx.box_geometry(10, 10, 10),
    gfx.MeshPhongMaterial(),
)
container.receive_shadow = True
scene.add(container)

camera = gfx.PerspectiveCamera(70, 2)
camera.local.position = (0, 4, 4)
camera.look_at((0, 0, 0))

scene.add(gfx.AmbientLight(0.15))

light1 = gfx.PointLight("#fff", 0.7)
light1.local.position = (0, 0, 4)
light1.cast_shadow = True
light2 = gfx.PointLight("#fff", 0.7)
light2.local.position = (0, 4, 0)
light2.cast_shadow = True
scene.add(light1)
scene.add(light2)
light2.visible = True


def animate():
    dist = material2.plane[3]
    dist += 0.02
    if dist > 2.2:
        dist = -2.2
    material2.plane = 1, 0, 1, dist

    renderer.render(scene, camera)
    canvas.request_draw()


if __name__ == "__main__":
    canvas.request_draw(animate)
    run()
