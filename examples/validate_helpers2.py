"""
Example showing the axes and grid helpers with a perspective camera.

* The grid spans the x-z plane (orange and blue axis).
* The yellow axis (y) stick up from the plane.
"""

from wgpu.gui.auto import WgpuCanvas, run
import pygfx as gfx


canvas = WgpuCanvas()
renderer = gfx.renderers.WgpuRenderer(canvas)
scene = gfx.Scene()

background = gfx.Background(None, gfx.BackgroundMaterial((0, 1, 0, 1), (0, 1, 1, 1)))
scene.add(background)

scene.add(gfx.AxesHelper(length=40))
scene.add(gfx.GridHelper(size=100))

camera = gfx.PerspectiveCamera(70, 16 / 9)
camera.position.set(50, 50, 50)
camera.look_at(gfx.linalg.Vector3())


if __name__ == "__main__":
    print(__doc__)
    canvas.request_draw(lambda: renderer.render(scene, camera))
    run()
