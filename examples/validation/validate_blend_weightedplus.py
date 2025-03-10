"""
Validate weighted_plus
======================

This example draws a series of semitransparent planes using weighted_plus.
"""

# sphinx_gallery_pygfx_docs = 'screenshot'
# sphinx_gallery_pygfx_test = 'compare'

from wgpu.gui.auto import WgpuCanvas, run
import pygfx as gfx
import pylinalg as la

canvas = WgpuCanvas(size=(600, 600))
renderer = gfx.renderers.WgpuRenderer(canvas)
renderer.blend_mode = "weighted_plus"

scene = gfx.Scene()

sphere = gfx.Mesh(gfx.sphere_geometry(10), gfx.MeshPhongMaterial())

geometry = gfx.plane_geometry(50, 50)
plane1 = gfx.Mesh(geometry, gfx.MeshBasicMaterial(color=(1, 0, 0, 0.3)))
plane2 = gfx.Mesh(geometry, gfx.MeshBasicMaterial(color=(0, 1, 0, 0.5)))
plane3 = gfx.Mesh(geometry, gfx.MeshBasicMaterial(color=(0, 0, 1, 0.7)))

plane1.local.rotation = la.quat_from_axis_angle((1, 0, 0), 1.571)
plane2.local.rotation = la.quat_from_axis_angle((0, 1, 0), 1.571)
plane3.local.rotation = la.quat_from_axis_angle((0, 0, 1), 1.571)

t = gfx.Text(text=renderer.blend_mode, screen_space=True, font_size=20)
t.local.position = (0, 40, 0)

scene.add(plane1, plane2, plane3, sphere, t)
scene.add(gfx.AmbientLight(1, 1))

camera = gfx.PerspectiveCamera(70, 16 / 9, depth_range=(0.1, 2000))
camera.local.position = (30, 40, 50)
camera.look_at((0, 0, 0))


def animate():
    renderer.render(scene, camera)


canvas.request_draw(animate)


if __name__ == "__main__":
    print(__doc__)
    run()
