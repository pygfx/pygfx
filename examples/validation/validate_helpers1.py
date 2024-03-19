"""
Axes Helper 1
=============

Example showing the axes helper.

* The axes must be centered in the middle.
* The red axes (x) must be to the right.
* The green axes (y) must be to the top.
* The blue axes (z) is not visible.
"""

# sphinx_gallery_pygfx_docs = 'screenshot'
# sphinx_gallery_pygfx_test = 'compare'

from wgpu.gui.auto import WgpuCanvas, run
import pygfx as gfx


canvas = WgpuCanvas()
renderer = gfx.renderers.WgpuRenderer(canvas)

scene = gfx.Scene()
scene.add(gfx.AxesHelper(size=40, thickness=5))
camera = gfx.OrthographicCamera(100, 100)

canvas.request_draw(lambda: renderer.render(scene, camera))

if __name__ == "__main__":
    print(__doc__)
    run()
