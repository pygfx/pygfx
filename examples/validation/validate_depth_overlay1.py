"""
Depth Overlay 1
===============

Example (and test) for behavior of the depth buffer w.r.t. overlays,
implemented by multiple render calls. The overlay should always be on
top.
"""

# sphinx_gallery_pygfx_docs = 'screenshot'
# sphinx_gallery_pygfx_test = 'compare'

import numpy as np
from wgpu.gui.auto import WgpuCanvas, run
import pygfx as gfx
import pylinalg as la


# Create a canvas and renderer

canvas = WgpuCanvas(size=(500, 300))
renderer = gfx.renderers.WgpuRenderer(canvas)

# Compose a scene with a 3D cube at the origin

scene1 = gfx.Scene()

cube1 = gfx.Mesh(
    gfx.box_geometry(),
    gfx.MeshPhongMaterial(color="#ff0"),
)
rot = la.quat_from_euler((0.2, 0.3), order="XY")
cube1.local.rotation = la.quat_mul(rot, cube1.local.rotation)
scene1.add(cube1)

# Compose a scene with a 2D square at the origin, in the xy plane

scene2 = gfx.Scene()

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
    gfx.LineMaterial(thickness=5.0, color="#f0f"),
)
scene2.add(line2)

# Camera

camera = gfx.OrthographicCamera(2, 2)
scene1.add(camera.add(gfx.DirectionalLight()))


def draw():
    renderer.render(scene1, camera, flush=False)
    renderer.render(scene2, camera, flush=False)
    renderer.flush()


canvas.request_draw(draw)

if __name__ == "__main__":
    print(__doc__)
    run()
