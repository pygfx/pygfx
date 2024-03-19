"""
Render a Triangle
=================

Replicating the WGPU triangle example, but with about 10x less code.
"""

# sphinx_gallery_pygfx_docs = 'screenshot'
# sphinx_gallery_pygfx_test = 'run'

from wgpu.gui.auto import WgpuCanvas, run
import pygfx as gfx


canvas = WgpuCanvas()
renderer = gfx.WgpuRenderer(canvas)
camera = gfx.NDCCamera()

triangle = gfx.Mesh(
    gfx.Geometry(
        indices=[(0, 1, 2)],
        positions=[(0.0, -0.5, 0), (0.5, 0.5, 0), (-0.5, 0.75, 0)],
        colors=[(1, 1, 0, 1), (1, 0, 1, 1), (0, 1, 1, 1)],
    ),
    gfx.MeshBasicMaterial(color_mode="vertex"),
)


if __name__ == "__main__":
    canvas.request_draw(lambda: renderer.render(triangle, camera))
    run()
