"""
Rotating Polyhedra
==================

Example showing multiple rotating polyhedrons.
"""

from wgpu.gui.auto import WgpuCanvas, run
import pygfx as gfx


canvas = WgpuCanvas()
renderer = gfx.renderers.WgpuRenderer(canvas)
scene = gfx.Scene()

material = gfx.MeshPhongMaterial()
geometries = [
    gfx.tetrahedron_geometry(),
    gfx.octahedron_geometry(),
    gfx.icosahedron_geometry(),
    gfx.dodecahedron_geometry(),
]

polyhedrons = [gfx.Mesh(g, material) for g in geometries]
for i, polyhedron in enumerate(polyhedrons):
    polyhedron.position.set(6 - i * 3, 0, 0)
    scene.add(polyhedron)

background = gfx.Background(None, gfx.BackgroundMaterial((0, 1, 0, 1), (0, 1, 1, 1)))
scene.add(background)

camera = gfx.PerspectiveCamera(70, 16 / 9)
camera.position.z = 10

scene.add(gfx.AmbientLight(), camera.add(gfx.DirectionalLight()))


def animate():
    for polyhedron in polyhedrons:
        rot = gfx.linalg.Quaternion().set_from_euler(gfx.linalg.Euler(0.01, 0.02))
        polyhedron.rotation.multiply(rot)

    renderer.render(scene, camera)
    canvas.request_draw()


if __name__ == "__main__":
    canvas.request_draw(animate)
    run()
