"""
Replicating the WGPU triangle example, but with about 10x less code.
"""

from wgpu.gui.auto import WgpuCanvas, run
import pygfx as gfx


canvas = WgpuCanvas()
renderer = gfx.WgpuRenderer(canvas)
camera = gfx.NDCCamera()

triangle = gfx.Mesh(
    gfx.Geometry(
        indices=[(0, 1, 2)],
        positions=[(0.0, -0.5, 0), (0.5, 0.5, 0), (-0.5, 0.7, 0)],
        colors=[(0.0, -0.5, 0.5, 1), (0.5, 0.5, 0.5, 1), (-0.5, 0.7, 0.5, 1)],
    ),
    gfx.MeshBasicMaterial(vertex_colors=True),
)


if __name__ == "__main__":
    canvas.request_draw(lambda: renderer.render(triangle, camera))
    run()
