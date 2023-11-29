"""
Depth Overlay 2
===============

Example (and test) for behavior of the depth buffer w.r.t. overlays,
implemented using ``Material.depth_test``. The overlaid object should
always be on top.
"""
# test_example = true

import numpy as np
from wgpu.gui.auto import WgpuCanvas, run
import pygfx as gfx
import pylinalg as la


# Create a canvas and renderer

canvas = WgpuCanvas(size=(500, 300))
renderer = gfx.renderers.WgpuRenderer(canvas)
scene = gfx.Scene()

# Compose a scene with a 3D cube at the origin

cube1 = gfx.Mesh(
    gfx.box_geometry(),
    gfx.MeshPhongMaterial(color="#ff0"),
)
rot = la.quat_from_euler((0.2, 0.3), order="XY")
cube1.local.rotation = la.quat_mul(rot, cube1.local.rotation)
scene.add(cube1)

# Second object

positions = np.array(
    [
        [-1, -1, 0.0],
        [-1, +1, 0.0],
        [+1, +1, 0.0],
        [+1, -1, 0.0],
        [-1, -1, 0.0],
        [+1, +1, 0.0],
    ],
    np.float32,
)
line2 = gfx.Line(
    gfx.Geometry(positions=positions * 0.5),
    gfx.LineMaterial(thickness=5.0, color="#f0f", depth_test=False),
)
scene.add(line2)

# Camera

camera = gfx.OrthographicCamera(2, 2)
scene.add(camera.add(gfx.DirectionalLight()))


if __name__ == "__main__":
    canvas.request_draw(lambda: renderer.render(scene, camera))
    run()
