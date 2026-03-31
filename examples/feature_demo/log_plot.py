"""
Line plot
=========

Show a line plot, with axii and a grid.
"""

# sphinx_gallery_pygfx_docs = 'screenshot'
# sphinx_gallery_pygfx_test = 'run'

import numpy as np
from rendercanvas.auto import RenderCanvas, loop
import pygfx as gfx
from pylinalg import vec_transform, vec_unproject


canvas = RenderCanvas()
renderer = gfx.WgpuRenderer(canvas)

scene = gfx.Scene()

background = gfx.Background.from_color("#000")

grid = gfx.Grid(
    None,
    gfx.GridMaterial(
        major_step=1,
        minor_step=0,
        thickness_space="screen",
        major_thickness=2,
        minor_thickness=0.5,
        infinite=True,
    ),
    orientation="xy",
)
grid.local.z = -10

rulerx = gfx.Ruler(tick_side="right", tick_marker="tick_left", min_tick_distance=50)
rulery = gfx.Ruler(tick_side="left", tick_marker="tick_right", min_tick_distance=40)

x = np.linspace(20, 980, 200, dtype=np.float32)
y = np.sin(x / 100) * 4 + 5 + np.random.uniform(-0.8, 0.8, x.shape).astype("f4")
y = 2 ** y
positions = np.column_stack([x, y, np.zeros_like(x)])

line = gfx.Line(
    gfx.Geometry(positions=positions),
    gfx.LineMaterial(thickness=4.0, color="#aaf"),
)
scene.add(background, grid, rulerx, rulery, line)

camera = gfx.OrthographicCamera(maintain_aspect=False,
        nonlinear='ylog10'
)
camera.show_object(scene, match_aspect=True, scale=1.1)


controller = gfx.PanZoomController(camera, register_events=renderer)


# A user may prefer to hard-code the ticks, but the ruler does a pretty good
# job at selection approproate ticks for log scales.s
# rulery.ticks = [1, 5, 10, 50, 100, 500, 1000]

rulerx.start_pos = 0, 1, 10
rulerx.end_pos = x.max(), 1, 10
rulery.start_pos = 0.1, 0, 10
rulery.end_pos = 0.1, y.max(), 10


def animate():

    statsx = rulerx.update(camera, canvas.get_logical_size())
    statsy = rulery.update(camera, canvas.get_logical_size())

    # Note that the grid lives in linear-space, i.e. it does not apply the log scale.
    # So in y we set a step of 1 which means every factor 10.
    major_step_x, major_step_y = statsx["tick_step"], 1
    grid.material.major_step = major_step_x, major_step_y
    grid.material.minor_step = 0.2 * major_step_x, 1

    renderer.render(scene, camera)


if __name__ == "__main__":
    canvas.request_draw(animate)
    loop.run()
