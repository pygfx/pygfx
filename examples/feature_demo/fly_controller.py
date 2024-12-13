"""
Fly controller
==============

Fly through a cloud of cololoured points. This example demonstrates the fly
controller, as well as the GaussianBlob point material, with size_space set to
'world'.

Tip: try using different blend modes.
"""

# sphinx_gallery_pygfx_docs = 'screenshot'
# sphinx_gallery_pygfx_test = 'run'

from wgpu.gui.auto import WgpuCanvas, run
import pygfx as gfx
import numpy as np


canvas = WgpuCanvas()
renderer = gfx.renderers.WgpuRenderer(canvas, blend_mode="weighted")
scene = gfx.Scene()
scene.add(gfx.Background.from_color("#000"))

grid = gfx.Grid(
    None,
    gfx.GridMaterial(
        major_step=100,
        minor_step=10,
        thickness_space="world",
        major_thickness=2,
        minor_thickness=0.1,
        infinite=True,
    ),
    orientation="xz",
)
grid.local.y = -120
scene.add(grid)

# Create a bunch of points
n = 1000
positions = np.random.normal(0, 50, (n, 3)).astype(np.float32)
sizes = np.random.rand(n).astype(np.float32) * 50
colors = np.random.rand(n, 4).astype(np.float32)
geometry = gfx.Geometry(positions=positions, sizes=sizes, colors=colors)

material = gfx.PointsGaussianBlobMaterial(
    color_mode="vertex", size_mode="vertex", size_space="world"
)
points = gfx.Points(geometry, material)
scene.add(points)

camera = gfx.PerspectiveCamera(70)
camera.show_object(scene)
controller = gfx.FlyController(camera, register_events=renderer)


if __name__ == "__main__":
    canvas.request_draw(lambda: renderer.render(scene, camera))
    run()
