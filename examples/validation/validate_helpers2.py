"""
Axis Helper 2
=============

Example showing the axes and grid helpers with a perspective camera.

* The grid spans the x-z plane (red and blue axis).
* The green axis (y) stick up from the plane.
* The red box fits snugly around the grid.
"""

# sphinx_gallery_pygfx_docs = 'screenshot'
# sphinx_gallery_pygfx_test = 'compare'

from wgpu.gui.auto import WgpuCanvas, run
import pygfx as gfx


canvas = WgpuCanvas()
renderer = gfx.renderers.WgpuRenderer(canvas)
scene = gfx.Scene()

axes = gfx.AxesHelper(size=40, thickness=8)
scene.add(axes)

grid = gfx.GridHelper(size=100, thickness=4)
scene.add(grid)

box = gfx.BoxHelper(size=100, thickness=4, color="red")
scene.add(box)

camera = gfx.PerspectiveCamera(70, 16 / 9, depth_range=(0.1, 2000))
camera.local.position = (75, 75, 75)
camera.show_pos((0, 0, 0))
controller = gfx.OrbitController(camera, register_events=renderer)


def animate():
    renderer.render(scene, camera)
    canvas.request_draw()


canvas.request_draw(animate)


if __name__ == "__main__":
    print(__doc__)
    run()
