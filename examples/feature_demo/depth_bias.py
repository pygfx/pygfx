"""
Depth bias
==========

Two tilted planes, each with a grid of lines lying exactly on it. On the
left plane the lines z-fight with the plane and are partly hidden. The
right plane is pushed back in the depth test with ``depth_bias`` and
``depth_bias_slope_scale``, so the lines on it are crisp.
"""

# sphinx_gallery_pygfx_docs = 'screenshot'
# sphinx_gallery_pygfx_test = 'run'

import numpy as np
from rendercanvas.auto import RenderCanvas, loop
import pygfx as gfx


canvas = RenderCanvas(size=(500, 300))
renderer = gfx.renderers.WgpuRenderer(canvas)
scene = gfx.Scene()


def add_plane_with_grid(x, depth_bias, depth_bias_slope_scale):
    plane = gfx.Mesh(
        gfx.plane_geometry(2, 2),
        gfx.MeshBasicMaterial(
            color="#fff",
            depth_bias=depth_bias,
            depth_bias_slope_scale=depth_bias_slope_scale,
        ),
    )
    grid = gfx.GridHelper(size=1.6, divisions=8, thickness=3)
    grid.local.euler_x = np.pi / 2  # GridHelper lies in the xz plane
    group = gfx.Group()
    group.add(plane, grid)
    group.local.x = x
    group.local.euler_x = 1.0  # tilt, so the planes have a depth slope
    scene.add(group)


add_plane_with_grid(-1.2, 0, 0.0)
add_plane_with_grid(+1.2, 32, 2.0)

camera = gfx.OrthographicCamera(5, 3)
controller = gfx.OrbitController(camera, register_events=renderer)

canvas.request_draw(lambda: renderer.render(scene, camera))

if __name__ == "__main__":
    print(__doc__)
    loop.run()
