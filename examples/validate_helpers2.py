"""
Example showing the axes and grid helpers with a perspective camera.

* The grid spans the x-z plane (red and blue axis).
* The green axis (y) stick up from the plane.
* The red box fits snugly around the grid.
"""
# test_example = true

from wgpu.gui.auto import WgpuCanvas, run
import pygfx as gfx


canvas = WgpuCanvas()
renderer = gfx.renderers.WgpuRenderer(canvas)
scene = gfx.Scene()

axes = gfx.AxesHelper(size=40, thickness=8)
scene.add(axes)

grid = gfx.GridHelper(size=100, thickness=4)
scene.add(grid)

box = gfx.BoxHelper(size=100, thickness=4)
scene.add(box)

camera = gfx.PerspectiveCamera(70, 16 / 9)
camera.position.set(75, 75, 75)
camera.look_at(gfx.linalg.Vector3())

controller = gfx.OrbitController(camera.position.clone())
controller.add_default_event_handlers(renderer, camera)


def animate():
    controller.update_camera(camera)
    renderer.render(scene, camera)
    canvas.request_draw()


canvas.request_draw(animate)


if __name__ == "__main__":
    print(__doc__)
    run()
