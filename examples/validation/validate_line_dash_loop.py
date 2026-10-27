"""
Dashed line loops
=================

Dashing combined with material.loop.

* The dashes have the same size and spacing all the way around each loop,
  including in the segment that closes the loop (see gh-1103).
* The bottom-right square has a perimeter that is an exact multiple of the
  dash period, so its dashes wrap around seamlessly: all four of its corners
  look the same, including the bottom-left one where the loop closes.
* Multiple loops (separated by nans) each close on themselves.
* Every loop starts its dash pattern from scratch, so each one has a dash
  beginning at its first node (12 o'clock here) at every zoom level. Zoom in
  and out to check: the number of dashes changes, but they stay anchored.
"""

# sphinx_gallery_pygfx_docs = 'screenshot'
# sphinx_gallery_pygfx_test = 'compare'

import numpy as np
from rendercanvas.auto import RenderCanvas, loop
import pygfx as gfx


canvas = RenderCanvas(size=(1000, 800))
renderer = gfx.WgpuRenderer(canvas)
scene = gfx.Scene()
scene.add(gfx.Background.from_color("#000"))


def circle(n, x=0, y=0, r=100):
    """A regular n-gon, as a loop (i.e. the last node is not repeated)."""
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack(
        [x + r * np.sin(t), y + r * np.cos(t), np.zeros_like(t)], axis=1
    ).astype(np.float32)


def rectangle(w, h, x=0, y=0):
    return np.array(
        [
            [x - 0.5 * w, y - 0.5 * h, 0],
            [x - 0.5 * w, y + 0.5 * h, 0],
            [x + 0.5 * w, y + 0.5 * h, 0],
            [x + 0.5 * w, y - 0.5 * h, 0],
        ],
        np.float32,
    )


nanpoint = np.full((1, 3), np.nan, np.float32)

# A bunch of loops in one buffer, separated by nans. With increasingly
# more nodes, so that both sharp corners and smooth curves are covered.
positions = np.vstack(
    [
        rectangle(200, 200, -350, 150),
        nanpoint,
        circle(3, -100, 150),
        nanpoint,
        circle(5, 150, 150),
        nanpoint,
        circle(64, 400, 150),
        nanpoint,
        circle(24, -350, -150),
        nanpoint,
        circle(8, -100, -150),
        nanpoint,
        circle(192, 150, -150),
    ],
    dtype=np.float32,
)

line1 = gfx.Line(
    gfx.Geometry(positions=positions),
    gfx.LineMaterial(
        thickness=10,
        color="#f55",
        loop=True,
        aa=True,
        dash_pattern=[2, 2],
        thickness_space="screen",
    ),
)
scene.add(line1)

# This square's perimeter (4 * 200) is an exact multiple of the dash
# period (thickness 10 * pattern-sum 4), so the dashes fit exactly and
# the corner that closes the loop is indistinguishable from the others.
line2 = gfx.Line(
    gfx.Geometry(positions=rectangle(200, 200, 400, -150)),
    gfx.LineMaterial(
        thickness=10,
        color="#5f5",
        loop=True,
        aa=True,
        dash_pattern=[2, 2],
        dash_offset=1,
        thickness_space="screen",
    ),
)
scene.add(line2)

camera = gfx.OrthographicCamera(1120, 896)
controller = gfx.PanZoomController(camera, register_events=renderer)

canvas.request_draw(lambda: renderer.render(scene, camera))

if __name__ == "__main__":
    print(__doc__)
    loop.run()
