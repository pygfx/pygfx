"""
Line Drawing
============


Some basic line drawing.
"""
# sphinx_gallery_pygfx_render = True
# sphinx_gallery_pygfx_target_name = "canvas"

import numpy as np
from wgpu.gui.auto import WgpuCanvas, run
import pygfx as gfx
import pylinalg as la
import numpy as np

canvas = WgpuCanvas(size=(1000, 800))
renderer = gfx.WgpuRenderer(canvas)
renderer_svg = gfx.SvgRenderer(640, 480, "~/line.svg")

renderer.blend_mode = "weighted"

scene = gfx.Scene()
positions = [[200 + np.sin(i) * i * 6, 200 + np.cos(i) * i * 6, 0] for i in range(20)]
positions += [[np.nan, np.nan, np.nan]]
positions += [[400 - np.sin(i) * i * 6, 200 + np.cos(i) * i * 6, 0] for i in range(20)]
positions += [[np.nan, np.nan, np.nan]]
positions += [
    [100, 440, 0],
    [100, 450, 0],
    [200, 450, 0],
    [200, 445, 0],
    [400, 440, 0],
    [300, 400, 0],
    [300, 390, 0],
    [400, 370, 0],
    # [300, 380, 0],
    [350, 350, 0],
]

# Spiral away in z (to make the depth buffer less boring)
for i in range(len(positions)):
    positions[i][2] = i

line = gfx.Line(
    gfx.Geometry(positions=positions),
    gfx.LineMaterial(thickness=22.0, color=(0.8, 0.7, 0.0), opacity=0.5),
)
scene.add(line)

camera = gfx.OrthographicCamera(600, 500)
camera.local.position = (300, 250, 0)

controller = gfx.PanZoomController(camera, register_events=renderer)

alpha = 0
d_alpha = 0.05


@renderer.add_event_handler("key_down")
def change_material(event):
    print(event.key)
    if event.key == "1":
        line.material = gfx.LineMaterial(
            thickness=22.0, color=(0.8, 0.7, 0.0), opacity=0.5
        )
    elif event.key == "2":
        line.material = gfx.LineDashedMaterial(
            thickness=22.0,
            color=(0.8, 0.7, 0.0),
            dash_size=40,
            dash_ratio=0.2,
            dash_is_screen_space=True,
        )
    elif event.key == "o":
        line.material.dash_offset += 4
    renderer.request_draw()


@renderer.add_event_handler("pointer_move", "pointer_down")
def set_node(event):
    if event.modifiers:
        return
    if 3 in event.buttons or event.button == 3:
        w, h = canvas.get_logical_size()
        ndcx, ndcy = 2 * event.x / w - 1, 1 - 2 * event.y / h
        pos = la.vec_transform((ndcx, ndcy, 0), np.linalg.inv(camera.camera_matrix))
        line.geometry.positions.data[-1, :2] = pos[0], pos[1]
        line.geometry.positions.update_range(len(positions) - 1, 1)
        renderer.request_draw()


def animate():
    global alpha
    alpha += d_alpha
    # line.geometry.positions.data[-1, :2] = 400 + 40 * np.sin(alpha), 370 + 40 * np.cos(
    #     alpha
    # )
    # line.geometry.positions.update_range(len(positions) - 1, 1)
    # canvas.request_draw()

    renderer.render(scene, camera)


if __name__ == "__main__":
    renderer_svg.render(scene, camera)
    canvas.request_draw(animate)
    run()
