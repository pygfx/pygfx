"""
Line loops of fixed size
========================

Drawing many closed shapes with ``material.loop = n``, i.e. without having to
separate (or close) the shapes with nan-positions.

The top row uses ``loop=n``, with just the corners of each shape. The bottom row
draws the same shapes the classic way: with an explicit nan-position between the
shapes. The two rows should look identical.
"""

# sphinx_gallery_pygfx_docs = 'screenshot'
# sphinx_gallery_pygfx_test = 'run'

import numpy as np
from rendercanvas.auto import RenderCanvas, loop
import pygfx as gfx


canvas = RenderCanvas(size=(800, 900))
renderer = gfx.WgpuRenderer(canvas)
scene = gfx.Scene()

# The lines are semi-transparent, so that any overlap at the point that closes
# the loop shows up as a brighter spot. Add a background to make that visible.
scene.add(gfx.Background.from_color("#fff", "#000"))


def shape(n, x, y):
    """The n corners of a regular n-gon, without repeating the first one."""
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([x + np.sin(t), y + np.cos(t), np.zeros_like(t)], axis=1)


def shapes(n, y):
    """A row of m shapes of n corners each, as one (m * n, 3) array."""
    return np.concatenate([shape(n, 3 * i - 6, y) for i in range(5)], axis=0)


nanpoint = np.full((1, 3), np.nan, dtype="f4")


def with_nans(positions, n):
    """The same positions, but with a nan-position between each shape."""
    pieces = positions.reshape(-1, n, 3)
    return np.concatenate(
        [x for piece in pieces for x in (piece, nanpoint)], axis=0
    ).astype("f4")


for i, n in enumerate([3, 4, 5, 12]):
    positions = shapes(n, -5.5 * i).astype("f4")

    # Closed with loop=n: n positions per shape
    material = gfx.LineMaterial(
        thickness=12,
        color="red",
        alpha_mode="blend",
        opacity=0.6,
        loop=n,
        aa=True,
        dash_pattern=(2, 1) if i % 2 else (),
    )
    scene.add(gfx.Line(gfx.Geometry(positions=positions), material))

    # Closed with loop=True: n + 1 positions per shape, the last one nan
    material = gfx.LineMaterial(
        thickness=12,
        color="cyan",
        alpha_mode="blend",
        opacity=0.6,
        loop=True,
        aa=True,
        dash_pattern=(2, 1) if i % 2 else (),
    )
    positions_nan = with_nans(positions, n)
    positions_nan[:, 1] -= 2.5
    scene.add(gfx.Line(gfx.Geometry(positions=positions_nan), material))


camera = gfx.OrthographicCamera(500, 500)
camera.show_object(scene, match_aspect=True, scale=1.1)
controller = gfx.PanZoomController(camera, register_events=renderer)


canvas.request_draw(lambda: renderer.render(scene, camera))


if __name__ == "__main__":
    print(__doc__)
    loop.run()
