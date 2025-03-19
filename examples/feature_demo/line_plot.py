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
from pylinalg import vec_transform, vec_unproject


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

rulerx = gfx.Ruler(tick_side="right")
rulery = gfx.Ruler(tick_side="left", min_tick_distance=40)

x = np.linspace(20, 980, 200, dtype=np.float32)
y = np.sin(x / 30) * 4
positions = np.column_stack([x, y, np.zeros_like(x)])

line = gfx.Line(
    gfx.Geometry(positions=positions),
    gfx.LineMaterial(thickness=4.0, color="#aaf"),
)
scene.add(background, grid, rulerx, rulery, line)

camera = gfx.OrthographicCamera(maintain_aspect=False)
camera.show_object(scene, match_aspect=True, scale=1.1)

controller = gfx.PanZoomController(camera, register_events=renderer)


def map_screen_to_world(pos, viewport_size):
    # first convert position to NDC
    x = pos[0] / viewport_size[0] * 2 - 1
    y = -(pos[1] / viewport_size[1] * 2 - 1)
    pos_ndc = (x, y, 0)

    pos_ndc += vec_transform(camera.world.position, camera.camera_matrix)
    # unproject to world space
    pos_world = vec_unproject(pos_ndc[:2], camera.camera_matrix)

    return pos_world


def animate():
    # get range of screen space
    xmin, ymin = 0, renderer.logical_size[1]
    xmax, ymax = renderer.logical_size[0], 0

    world_xmin, world_ymin, _ = map_screen_to_world((xmin, ymin), renderer.logical_size)
    world_xmax, world_ymax, _ = map_screen_to_world((xmax, ymax), renderer.logical_size)

    # set start and end positions of rulers
    rulerx.start_pos = world_xmin, 0, -1000
    rulerx.end_pos = world_xmax, 0, -1000

    rulerx.start_value = rulerx.start_pos[0]

    statsx = rulerx.update(camera, canvas.get_logical_size())

    rulery.start_pos = 0, world_ymin, -1000
    rulery.end_pos = 0, world_ymax, -1000

    rulery.start_value = rulery.start_pos[1]
    statsy = rulery.update(camera, canvas.get_logical_size())

    major_step_x, major_step_y = statsx["tick_step"], statsy["tick_step"]
    grid.material.major_step = major_step_x, major_step_y
    grid.material.minor_step = 0.2 * major_step_x, 0.2 * major_step_y

    # print(statsx)

    renderer.render(scene, camera)


if __name__ == "__main__":
    canvas.request_draw(animate)
    run()
