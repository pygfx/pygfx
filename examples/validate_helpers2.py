"""
Example showing the axes and grid helpers with a perspective camera.

* The grid spans the x-z plane (orange and blue axis).
* The yellow axis (y) stick up from the plane.
* The red box fits snugly around the grid.
"""
# test_example = true

from wgpu.gui.auto import WgpuCanvas, run
import pygfx as gfx


canvas = WgpuCanvas()
renderer = gfx.renderers.WgpuRenderer(canvas)
scene = gfx.Scene()

background = gfx.Background(
    None, gfx.BackgroundMaterial((0, 0.1, 0, 1), (0, 0.1, 0.1, 1))
)
scene.add(background)

axes = gfx.AxesHelper(length=40)
scene.add(axes)

grid = gfx.GridHelper(size=100)
scene.add(grid)

box = gfx.BoxHelper(size=100)
scene.add(box)

camera = gfx.PerspectiveCamera(70, 16 / 9)
camera.position.set(75, 75, 75)
camera.look_at(gfx.linalg.Vector3())

controls = gfx.OrbitControls(camera.position.clone())
controls.add_default_event_handlers(canvas, camera)


def animate():
    controls.update_camera(camera)
    renderer.render(scene, camera)
    canvas.request_draw()


canvas.request_draw(animate)


if __name__ == "__main__":
    print(__doc__)
    run()
