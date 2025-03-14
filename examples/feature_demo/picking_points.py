"""
Point Picking
=============

Example showing picking points. When clicking on a point, it's location
is changed. With a small change, a line is shown instead.
"""

# sphinx_gallery_pygfx_docs = 'screenshot'
# sphinx_gallery_pygfx_test = 'run'

import numpy as np
from wgpu.gui.auto import WgpuCanvas, run
import pygfx as gfx


canvas = WgpuCanvas()
renderer = gfx.renderers.WgpuRenderer(canvas)
scene = gfx.Scene()

xx = np.linspace(-50, 50, 10)
yy = np.random.uniform(20, 50, 10)
geometry = gfx.Geometry(positions=[(x, y, 0) for x, y in zip(xx, yy)])
ob = gfx.Points(
    geometry, gfx.PointsMaterial(color=(0, 1, 1, 1), size=20, pick_write=True)
)
scene.add(ob)

camera = gfx.OrthographicCamera(120, 120)


@ob.add_event_handler("pointer_down")
def offset_point(event):
    info = event.pick_info
    if "vertex_index" in info:
        i = round(info["vertex_index"])
        geometry.positions.data[i, 1] *= -1
        geometry.positions.update_range(i)
        canvas.request_draw()


if __name__ == "__main__":
    canvas.request_draw(lambda: renderer.render(scene, camera))
    run()
