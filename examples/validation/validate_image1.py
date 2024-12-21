"""
Image on Plane Geometry 1
=========================

Show an image using a plane geometry. For historic reasons, image
data (usually) has the first rows representing the top of the image.
But the plane gemeometry is such that it is reversed again.

* The green dots should be at the corners that are NOT darker/brighter.
* The green dots should be on the edge of the image.
* The darker corner is in the top left.
"""

# sphinx_gallery_pygfx_docs = 'screenshot'
# sphinx_gallery_pygfx_test = 'compare'

import numpy as np
from wgpu.gui.auto import WgpuCanvas, run
import pygfx as gfx


canvas = WgpuCanvas()
renderer = gfx.renderers.WgpuRenderer(canvas)
scene = gfx.Scene()

im = (
    np.array(
        [
            [0, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 2],
        ],
        np.float32,
    )
    / 2.0
)

tex = gfx.Texture(im, dim=2)

plane = gfx.Mesh(
    gfx.plane_geometry(4, 4),
    gfx.MeshBasicMaterial(map=gfx.TextureMap(tex, filter="nearest")),
)
plane.local.position = (2, 2, 0)  # put corner at 0, 0
scene.add(plane)

points = gfx.Points(
    gfx.Geometry(positions=[[0, 0, 1], [4, 4, 1]]),
    gfx.PointsMaterial(color=(0, 1, 0, 1), size=20),
)
scene.add(points)

camera = gfx.OrthographicCamera(10, 10)

canvas.request_draw(lambda: renderer.render(scene, camera))

if __name__ == "__main__":
    print(__doc__)
    run()
