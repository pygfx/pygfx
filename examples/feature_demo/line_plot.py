"""
Line plot
=========

Show a line plot, with axii and a grid.
"""

# sphinx_gallery_pygfx_docs = 'screenshot'
# sphinx_gallery_pygfx_test = 'run'

import numpy as np
from wgpu.gui.auto import WgpuCanvas, run
import pygfx as gfx


canvas = WgpuCanvas()
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
grid.local.z = -1001

rulerx = gfx.Ruler()
rulery = gfx.Ruler()

x = np.linspace(20, 980, 200, dtype=np.float32)
y = np.sin(x / 30) * 4
positions = np.column_stack([x, y, np.zeros_like(x)])

line = gfx.Line(
    gfx.Geometry(positions=positions),
    gfx.LineMaterial(thickness=4.0, color="#aaf"),
)
scene.add(background, grid, rulerx, rulery, line)

camera = gfx.OrthographicCamera(maintain_aspect=False)
camera.show_rect(0, 1000, -5, 5)

controller = gfx.PanZoomController(camera, register_events=renderer)


def animate():
    start_pos = camera.world.x - camera.width / 2, 0, -1000
    end_pos = end_pos = camera.world.x + camera.width / 2, 0, -1000
    start_value = start_pos[0]

    rulerx.configure(start_pos, end_pos, start_value)
    major_step_x = rulerx.calculate_tick_step(camera, canvas.get_logical_size())
    ticks_x = rulerx.get_ticks_uniform(major_step_x)
    rulerx.set_ticks(ticks_x, text_anchor="top-center", text_anchor_offset=5)

    start_pos = 0, camera.world.y - camera.height / 2, -1000
    end_pos = end_pos = 0, camera.world.y + camera.height / 2, -1000
    start_value = start_pos[1]

    rulery.configure(start_pos, end_pos, start_value)
    major_step_y = rulery.calculate_tick_step(camera, canvas.get_logical_size())
    ticks_y = rulery.get_ticks_uniform(major_step_y)
    rulery.set_ticks(ticks_y, text_anchor="middle-right", text_anchor_offset=10)

    grid.material.major_step = major_step_x, major_step_y
    grid.material.minor_step = 0.2 * major_step_x, 0.2 * major_step_y

    # pos = la.vec_transform((1, 0, 0), camera.projection_matrix_inverse)
    # pos = la.vec_transform(pos, camera.world.matrix)
    # print(pos)

    renderer.render(scene, camera)


if __name__ == "__main__":
    canvas.request_draw(animate)
    run()
