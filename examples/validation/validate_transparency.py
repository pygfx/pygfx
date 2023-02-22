"""
Transparency
============

This example draws a series of semitransparent planes using weighted_plus.
"""
# test_example = true
# sphinx_gallery_pygfx_render = True

from wgpu.gui.auto import WgpuCanvas, run
import pygfx as gfx


canvas = WgpuCanvas(size=(600, 600))
renderer = gfx.renderers.WgpuRenderer(canvas)
renderer.blend_mode = "weighted_plus"

camera = gfx.PerspectiveCamera(70, 16 / 9)
camera.position.set(30, 40, 50)
camera.look_at(gfx.linalg.Vector3(0, 0, 0))


scene = gfx.Scene()

sphere = gfx.Mesh(gfx.sphere_geometry(10), gfx.MeshPhongMaterial())

geometry = gfx.plane_geometry(50, 50)
plane1 = gfx.Mesh(geometry, gfx.MeshBasicMaterial(color=(1, 0, 0, 0.3)))
plane2 = gfx.Mesh(geometry, gfx.MeshBasicMaterial(color=(0, 1, 0, 0.5)))
plane3 = gfx.Mesh(geometry, gfx.MeshBasicMaterial(color=(0, 0, 1, 0.7)))

plane1.rotation.set_from_axis_angle(gfx.linalg.Vector3(1, 0, 0), 1.571)
plane2.rotation.set_from_axis_angle(gfx.linalg.Vector3(0, 1, 0), 1.571)
plane3.rotation.set_from_axis_angle(gfx.linalg.Vector3(0, 0, 1), 1.571)

t = gfx.Text(
    gfx.TextGeometry(renderer.blend_mode, screen_space=True, font_size=20),
    gfx.TextMaterial(),
)
t.position.set(0, 40, 0)

scene.add(plane1, plane2, plane3, sphere, t)

scene.add(gfx.AmbientLight(1, 1))


@canvas.request_draw
def animate():
    renderer.render(scene, camera)


if __name__ == "__main__":
    print(__doc__)
    run()
