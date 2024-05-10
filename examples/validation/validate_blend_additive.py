"""
Additive Blending
=================

This example draws overlapping circles of reference colors with additive blending.
"""

# sphinx_gallery_pygfx_docs = 'screenshot'
# sphinx_gallery_pygfx_test = 'compare'

import numpy as np
from wgpu.gui.auto import WgpuCanvas, run
import pygfx as gfx

colors = ["#ff0000", "#00ff00", "#0000ff"]
distance_from_center = 0.5

canvas = WgpuCanvas()
renderer = gfx.WgpuRenderer(canvas, gamma_correction=1.0)
renderer.blend_mode = "additive"
camera = gfx.OrthographicCamera()
camera.show_rect(-2, 2, -2, 2)
scene = gfx.Scene()

plane = gfx.sphere_geometry(height_segments=64)
N = len(colors)
initial_angle = np.pi / 2
for i, color in enumerate(colors):
    angle = -2 * np.pi * i / N + initial_angle
    x = distance_from_center * np.cos(angle)
    y = distance_from_center * np.sin(angle)

    m = gfx.Mesh(plane, gfx.MeshBasicMaterial(color=color))
    m.local.x = x
    m.local.y = y
    scene.add(m)


canvas.request_draw(lambda: renderer.render(scene, camera))


if __name__ == "__main__":
    print(__doc__)
    run()
