"""
Dash scaling
============

How the dash pattern responds to zoom. Scroll to zoom, drag to pan.

All four rows are the same geometry with the same ``dash_pattern``, and differ
only in ``dash_scale_step``: the factor by which the dash size is allowed to
jump as the view scale changes.

* ``dash_scale_step=1`` (top) is the old behaviour, and is what
  ``dash_scaling="continuous"`` does. The size follows the view exactly, so the
  dashes always have the size asked for -- but their number along the line has
  to change continuously, so they slide. A dash drifts in proportion to its
  distance from the start of its line piece, which is why the far end of the
  long line races while the near end barely moves.

* ``dash_scale_step=2`` (bottom) snaps the size to powers of two. Nothing
  slides: each time the level changes, every dash splits in two (or pairs
  merge), and no dash moves. The cost is that the size is only right to within
  a factor of two.

The two rows between are the same mechanism at 1.25 and 1.6, to show that this
is one continuous dial rather than two modes. Only a whole-number step gives
dashes that truly never move, because the dash starts of one level are a subset
of the next level's only when the levels differ by a whole factor; at 1.25 and
1.6 the dashes still jump rather than slide, but they land somewhere new when
they do. The smaller the step, the smaller and more frequent the jumps, until
at 1 they merge into a continuous slide.

A second parameter, ``dash_max_scale``, chooses where the size range sits: at
``dash_scale_step`` the dashes are never finer than asked, at 1 never coarser,
and the default (None) centres it.

Watch the far end of the long line for the sliding, and watch any one dash on
the bottom row to see it split rather than move.
"""

# sphinx_gallery_pygfx_docs = 'screenshot'
# sphinx_gallery_pygfx_test = 'run'

import numpy as np
from rendercanvas.auto import RenderCanvas, loop
import pygfx as gfx


canvas = RenderCanvas(size=(1000, 700))
renderer = gfx.WgpuRenderer(canvas)
scene = gfx.Scene()
scene.add(gfx.Background.from_color("#000"))


def circle(n, x=0.0, y=0.0, r=55.0):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack(
        [x + r * np.sin(t), y + r * np.cos(t), np.zeros_like(t)], axis=1
    ).astype(np.float32)


nanpoint = np.full((1, 3), np.nan, np.float32)


def geometry(y):
    """A long straight line plus two loops, so drift shows up at both ends."""
    straight = np.array([[-430, y + 70, 0], [430, y + 70, 0]], np.float32)
    return gfx.Geometry(
        positions=np.vstack(
            [
                straight,
                nanpoint,
                circle(96, -150, y),
                nanpoint,
                circle(6, 150, y),
            ]
        ).astype(np.float32)
    )


# dash_scale_step, colour, label
VARIANTS = [
    (1.0, "#f55", "dash_scale_step=1: the old behaviour, exact size but slides"),
    (1.25, "#5f5", "dash_scale_step=1.25: small, frequent jumps"),
    (1.6, "#59f", "dash_scale_step=1.6: larger, rarer jumps"),
    (2.0, "#fd5", "dash_scale_step=2: dashes split in two, and never move"),
]

for i, (dash_scale_step, color, label) in enumerate(VARIANTS):
    y = 240 - i * 170
    material = gfx.LineMaterial(
        thickness=8,
        color=color,
        loop=True,
        aa=True,
        dash_pattern=[2, 2],
        thickness_space="screen",
        dash_scaling="quantized",
    )
    material.dash_scale_step = dash_scale_step
    scene.add(gfx.Line(geometry(y), material))

    text = gfx.Text(
        text=label,
        font_size=15,
        screen_space=True,
        anchor="middle-left",
        material=gfx.TextMaterial(color=color),
    )
    text.local.position = (-430, y + 95, 0)
    scene.add(text)

camera = gfx.OrthographicCamera(1000, 700)
controller = gfx.PanZoomController(camera, register_events=renderer)

canvas.request_draw(lambda: renderer.render(scene, camera))

if __name__ == "__main__":
    print(__doc__)
    loop.run()
