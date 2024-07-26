"""
Validate ruler
==============

Showing off some ruler tricks.
"""

# sphinx_gallery_pygfx_docs = 'screenshot'
# sphinx_gallery_pygfx_test = 'compare'

import numpy as np
from wgpu.gui.auto import WgpuCanvas, run
import pygfx as gfx


canvas = WgpuCanvas()
renderer = gfx.WgpuRenderer(canvas)

scene = gfx.Scene()
background = gfx.Background.from_color("#000")


x = np.linspace(20, 980, 200, dtype=np.float32)
y = np.sin(x / 30) * 4
positions = np.column_stack([x, y, np.ones_like(x)])

line = gfx.Line(
    gfx.Geometry(positions=positions),
    gfx.LineMaterial(thickness=4.0, color="#aaf"),
)
scene.add(background, line)


# Normal horizontal ruler
rulerx1 = gfx.Ruler(
    tick_side="right",
    start_value=0,
    start_pos=(0, 0, 0),
    end_pos=(1000, 0, 0),
)

# Normal vertical ruler
rulery1 = gfx.Ruler(
    tick_side="left",
    start_value=-5,
    start_pos=(0, -5, 0),
    end_pos=(0, 5, 0),
)

# Alternative formatting and tick distance
rulerx2 = gfx.Ruler(
    tick_side="right",
    start_pos=(0, -6, 0),
    end_pos=(1000, -6, 0),
    tick_format="0.1f",
    min_tick_distance=100,
)

# Ticks specified using list, and formatting via a function
rulerx3 = gfx.Ruler(
    tick_side="right",
    start_pos=(0, -8, 0),
    end_pos=(1000, -8, 0),
    ticks=[0, 250, 500, 750, 950],
    tick_format=lambda v, mi, ma: str(v / 1000) + "K",
)

# Ticks specified using a dict, and showing ticks on other side.
# Note how the dict can contain both strings and floats.
rulerx4 = gfx.Ruler(
    tick_side="left",
    start_pos=(0, -10, 0),
    end_pos=(1000, -10, 0),
    ticks={250: "25%", 500: "50%", 750: "75%", 0: 0, 1000: 1000, 400: "POI"},
    tick_format="0.1f",
)

scene.add(rulerx1, rulerx2, rulerx3, rulerx4, rulery1)


camera = gfx.OrthographicCamera(maintain_aspect=False)
camera.show_rect(-100, 1100, -12, 6)
controller = gfx.PanZoomController(camera, register_events=renderer)


def animate():
    for ob in scene.children:
        if isinstance(ob, gfx.Ruler):
            ob.update(camera, canvas.get_logical_size())

    renderer.render(scene, camera)


canvas.request_draw(animate)

if __name__ == "__main__":
    print(__doc__)
    run()
