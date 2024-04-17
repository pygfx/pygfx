"""
Scene Overlay
=============

Example showing a 3D scene, with an object overload via
``Material.depth_test``, and a 2D scene overlay over it all using an
overlay render pass.
"""

# sphinx_gallery_pygfx_docs = 'screenshot'
# sphinx_gallery_pygfx_test = 'run'

import numpy as np
from wgpu.gui.auto import WgpuCanvas, run
import pygfx as gfx
import pylinalg as la


# Create a canvas and renderer

canvas = WgpuCanvas(size=(500, 300))
renderer = gfx.renderers.WgpuRenderer(canvas)

# Compose a 3D scene

scene1 = gfx.Scene()

cube1 = gfx.Mesh(
    gfx.box_geometry(200, 200, 200),
    gfx.MeshPhongMaterial(color="#ff0"),
)
scene1.add(cube1)

camera = gfx.OrthographicCamera(300, 300)
scene1.add(camera.add(gfx.DirectionalLight()))

# Add an object that is drawn on top, even though it's inside the bigger cube

cube2 = gfx.Mesh(
    gfx.box_geometry(75, 75, 75),
    gfx.MeshPhongMaterial(color="#00f", depth_test=False, side="front"),
)
scene1.add(cube2)


# Compose another scene, a 2D overlay

scene2 = gfx.Scene()

positions = np.array(
    [
        [-1, -1, 0.5],
        [-1, +1, 0.5],
        [+1, +1, 0.5],
        [+1, -1, 0.5],
        [-1, -1, 0.5],
        [+1, +1, 0.5],
    ],
    np.float32,
)
line2 = gfx.Line(
    gfx.Geometry(positions=positions * 100),
    gfx.LineMaterial(thickness=5.0, color="#f0f"),
)
scene2.add(line2)


def animate():
    rot = la.quat_from_euler((0.005, 0.01), order="XY")
    cube1.local.rotation = la.quat_mul(rot, cube1.local.rotation)
    cube2.local.rotation = cube1.local.rotation

    renderer.render(scene1, camera, flush=False)
    renderer.render(scene2, camera)

    canvas.request_draw()


if __name__ == "__main__":
    canvas.request_draw(animate)
    run()
