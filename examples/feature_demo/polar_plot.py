"""
Polar plot
==========

Show a polar plot.
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


rulerx = gfx.Ruler(tick_side="right", tick_marker="tick_left", min_tick_distance=50)
rulery = gfx.Ruler(tick_side="left", tick_marker="tick_right", min_tick_distance=40)

radius = np.arange(0, 2, 0.01, dtype=np.float32)
theta = 2 * np.pi * radius
positions = np.column_stack([theta, radius, np.zeros_like(radius)])

line = gfx.Line(
    gfx.Geometry(positions=positions),
    gfx.LineMaterial(thickness=4.0, color="#aaf"),
)
scene.add(background, rulerx, rulery, line)

camera = gfx.OrthographicCamera(maintain_aspect=False, nonlinear="polar")
camera.show_object(scene, match_aspect=True, scale=1.1)


controller = gfx.PanZoomController(camera, register_events=renderer)


# A user may prefer to hard-code the ticks, but the ruler does a pretty good
# job at selection approproate ticks for log scales.s
# rulery.ticks = [1, 5, 10, 50, 100, 500, 1000]

rulerx.start_pos = 0,  radius.max(), 10
rulerx.end_pos = 2*np.pi,  radius.max(), 10
rulery.start_pos = 0, 0, 10
rulery.end_pos = 0,  radius.max(), 10

rulerx.ticks = {r*np.pi/180: f"{r} deg" for r in np.linspace(0, 360, 36)}

def animate():
    statsx = rulerx.update(camera, canvas.get_logical_size())
    statsy = rulery.update(camera, canvas.get_logical_size())

    # Note that the grid lives in linear-space, i.e. it does not apply the log scale.
    # So in y we set a step of 1 which means every factor 10.
    major_step_x, major_step_y = statsx["tick_step"], 1
    # grid.material.major_step = major_step_x, major_step_y
    # grid.material.minor_step = 0.2 * major_step_x, 1

    renderer.render(scene, camera)


if __name__ == "__main__":
    canvas.request_draw(animate)
    loop.run()
