"""
Fly controller
==============

Show the fly controller in action.
"""
# sphinx_gallery_pygfx_render = True

from wgpu.gui.auto import WgpuCanvas, run
import pygfx as gfx
import numpy as np


canvas = WgpuCanvas()
renderer = gfx.renderers.WgpuRenderer(canvas)
scene = gfx.Scene()

# Create a bunch of points
n = 1000
positions = np.random.normal(0, 0.5, (n, 3)).astype(np.float32)
sizes = np.random.rand(n).astype(np.float32) * 50
colors = np.random.rand(n, 4).astype(np.float32)
geometry = gfx.Geometry(positions=positions, sizes=sizes, colors=colors)

material = gfx.PointsMaterial(vertex_colors=True, vertex_sizes=True)
points = gfx.Points(geometry, material)
scene.add(points)

camera = gfx.PerspectiveCamera(70)
camera.show_object(scene)
controller = gfx.FlyController(camera, register_events=renderer)


if __name__ == "__main__":
    canvas.request_draw(lambda: renderer.render(scene, camera))
    run()
