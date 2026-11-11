"""
Stencil priority layers
=======================

A bunch of squares, all in the same plane, in three "levels": level 0 has
many small squares, level 1 has fewer squares that are twice as large, and
level 2 is a single square twice as large again. Squares of level 0 and 1
partly overlap.

The squares are drawn with additive blending and no depth test. The stencil
buffer is used to give the levels priority: where a level 0 square is
drawn, no level 1 or level 2 square is drawn, and where a level 1 square is
drawn, no level 2 square is drawn. So each pixel shows the highest level
that covers it.

This works by drawing the levels from high to low priority, each marking
the stencil buffer where it is drawn.
"""

# sphinx_gallery_pygfx_docs = 'screenshot'
# sphinx_gallery_pygfx_test = 'run'

import numpy as np
import wgpu
from rendercanvas.auto import RenderCanvas, loop
import pygfx as gfx


canvas = RenderCanvas()
renderer = gfx.WgpuRenderer(canvas, enable_depth=False, enable_stencil=True)
scene = gfx.Scene()

rng = np.random.default_rng(42)

LEVEL1_POSITIONS = [(-0.4, 0.69), (0.4, 0.731), (-0.764, -0.623), (0.741, -0.654)]

# The stencil buffer is cleared to the highest possible number,
# and we write numbers corresponding to the level.
renderer.set_clear_value(stencil=255, color=(0.0, 0.0, 0.0, 1.0))


def add_square(size, x, y, color, level):
    material = gfx.MeshBasicMaterial(
        color=color,
        opacity=0.3,
        alpha_mode="add",  # additive blending
        depth_test=False,  # all squares are in the same plane
        depth_write=False,
    )
    # Stencil: draw only where no higher-priority level was drawn yet
    material.stencil_value = level
    state = wgpu.StencilFaceState(compare="less-equal", pass_op="replace")
    material.stencil_front_state = state
    material.stencil_back_state = state

    square = gfx.Mesh(gfx.plane_geometry(size, size), material)
    square.local.position = (x, y, 0)
    square.render_order = level  # draw level 0 first, then 1, then 2
    scene.add(square)


# level 0: many small squares, randomly placed so they partly overlap
for _ in range(16):
    x, y = rng.uniform(-1.0, 1.0, 2)
    add_square(0.5, x, y, "#ff0000", 0)
# level 1: twice as large, fewer, partly overlapping the level 0 squares
for x, y in LEVEL1_POSITIONS:
    add_square(1.0, x, y, "#00ff00", 1)
# level 2: twice as large again, exactly one
add_square(2.0, 0.0, 0.0, "#0000ff", 2)

camera = gfx.OrthographicCamera(16 / 9)
camera.show_rect(-1.4, 1.4, -1.4, 1.4)
controller = gfx.PanZoomController(camera, register_events=renderer)


if __name__ == "__main__":
    print(__doc__)
    canvas.request_draw(lambda: renderer.render(scene, camera))
    loop.run()
