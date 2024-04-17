"""
Subplots 1
==========

Example showing how to render to a subregion of a canvas.

This is a feature necessary to implement e.g. subplots. This example
uses a low-level approach without using the Viewport object. See
scene_subplot2.py for a slightly higher-level approach.
"""

# sphinx_gallery_pygfx_docs = 'screenshot'
# sphinx_gallery_pygfx_test = 'run'

import numpy as np
from wgpu.gui.auto import WgpuCanvas, run
import pygfx as gfx
import pylinalg as la


# Create a anvas and a renderer

canvas = WgpuCanvas(size=(500, 300))
renderer = gfx.renderers.WgpuRenderer(canvas)

# Compose a 3D scene

scene1 = gfx.Scene()

geometry1 = gfx.box_geometry(200, 200, 200)
material1 = gfx.MeshPhongMaterial(color=(1, 1, 0, 1.0))
cube1 = gfx.Mesh(geometry1, material1)
scene1.add(cube1)

camera1 = gfx.PerspectiveCamera(70, 16 / 9, width=200)
camera1.local.z = 400
scene1.add(camera1.add(gfx.DirectionalLight()))

# Compose another scene

scene2 = gfx.Scene()

positions = np.array(
    [[-1, -1, 0], [-1, +1, 0], [+1, +1, 0], [+1, -1, 0], [-1, -1, 0], [+1, +1, 0]],
    np.float32,
)
geometry2 = gfx.Geometry(positions=positions)
material2 = gfx.LineMaterial(thickness=5.0, color=(0.8, 0.0, 0.2, 1.0))
line2 = gfx.Line(geometry2, material2)
scene2.add(line2)

camera2 = gfx.OrthographicCamera(2.2, 2.2)


def animate():
    rot = la.quat_from_euler((0.005, 0.01), order="XY")
    cube1.local.rotation = la.quat_mul(rot, cube1.local.rotation)

    w, h = canvas.get_logical_size()
    renderer.render(scene1, camera1, flush=False, rect=(0, 0, w / 2, h / 2))
    renderer.render(scene2, camera2, flush=False, rect=(w / 2, 0, w / 2, h / 2))
    renderer.render(scene2, camera2, flush=False, rect=(0, h / 2, w / 2, h / 2))
    renderer.render(scene1, camera1, flush=False, rect=(w / 2, h / 2, w / 2, h / 2))
    renderer.flush()

    canvas.request_draw()


if __name__ == "__main__":
    canvas.request_draw(animate)
    run()
