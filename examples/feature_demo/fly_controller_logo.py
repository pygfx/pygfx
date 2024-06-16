"""
Fly controller logo
===================

Fly through a cloud of cololoured pygfx logos to inspect the effects of the
renering.
"""

# sphinx_gallery_pygfx_docs = 'screenshot'
# sphinx_gallery_pygfx_test = 'run'

from wgpu.gui.auto import WgpuCanvas, run
import pygfx as gfx
import numpy as np


canvas = WgpuCanvas()
renderer = gfx.renderers.WgpuRenderer(canvas)
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

# Create a bunch of logos
n = 100
positions = np.random.normal(0, 50, (n, 3)).astype(np.float32)
sizes = np.random.rand(n).astype(np.float32) * 50
colors_inner = np.random.rand(n, 4).astype(np.float32)
colors_outer = np.random.rand(n, 4).astype(np.float32)
geometry_inner = gfx.Geometry(
    positions=positions,
    sizes=sizes,
    colors=colors_inner,
)
geometry_outer = gfx.Geometry(
    positions=positions,
    sizes=sizes,
    colors=colors_outer,
)

points_inner = gfx.Points(
    geometry_inner,
    gfx.PointsMarkerMaterial(
        marker="pygfx_inner",
        color_mode="vertex",
        size_mode="vertex",
        size_space="world",
    ),
)
points_outer = gfx.Points(
    geometry_outer,
    gfx.PointsMarkerMaterial(
        marker="pygfx_outer",
        color_mode="vertex",
        size_mode="vertex",
        size_space="world",
    ),
)
scene.add(points_inner, points_outer)

camera = gfx.PerspectiveCamera(70)
camera.show_object(scene)
controller = gfx.FlyController(camera, register_events=renderer)


if __name__ == "__main__":
    canvas.request_draw(lambda: gfx.render_with_logo(renderer, scene, camera))
    run()
